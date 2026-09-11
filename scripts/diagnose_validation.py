"""Diagnose a pooled correlation against client labels.

A single pooled Pearson r conflates several very different situations, and the
remedy for each is different -- so the first job after getting one is to take it
apart rather than to act on it.

What this separates:

* **Range restriction.** r is a ratio to the spread in the labels. If graders put
  almost everything at 3 or 4 on a 0-5 scale, r is attenuated no matter how good
  the scores are, and the number understates the model.
* **Between- vs within-question variance.** Our own validation was per question
  (relevance ranked a graded fixture at rho 1.000 within a prompt). Pooling mixes
  in between-question differences, which can inflate r -- when both sides agree
  some prompts are harder -- or depress it, when our per-question offsets do not
  match the graders'. Both are plausible here and they point opposite ways, which
  is exactly why it has to be measured. Centring each side on its question mean
  removes the between-question part and leaves the agreement that actually
  matters for ranking candidates who answered the same prompt.
* **One latent factor wearing four hats.** Four scores landing within 0.07 of each
  other is what you would see if all four mostly track general proficiency. If our
  four scores inter-correlate at 0.9 we are selling one number four times. The
  same check on the labels matters just as much: graders show halo effects, and if
  *their* four columns inter-correlate at 0.9 then four separable dimensions are
  not recoverable from this data at all, by us or by anyone.
* **Length.** Words spoken predicts human ratings of speech strongly. If our scores
  correlate with transcript length about as well as they correlate with the
  labels, length is doing the work.
* **Abstentions and unscorable items.** An abstained channel is a constant, and
  constants drag a pooled r toward zero while looking like poor accuracy.

Usage:

    python scripts/diagnose_validation.py --file merged.xlsx \\
        --pred relevance,fluency,lexical,grammar \\
        --true human_rel,human_flu,human_lex,human_gram \\
        --question question_id --length duration_s

Column names are matched case-insensitively and forgivingly, so the typo'd
headers in a hand-merged sheet ("lexcial", "grammae") still resolve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CATS = ("relevance", "fluency", "lexical", "grammar")


def _read(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {path}")


def _resolve(df: pd.DataFrame, names: list[str], what: str) -> list[str]:
    """Map requested column names onto real ones, tolerating typos and case."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    out = []
    for n in names:
        key = n.strip().lower()
        if key in lowered:
            out.append(lowered[key])
            continue
        # Fall back to nearest-neighbour matching. A prefix rule is not enough:
        # the transposition in "lexcial" corrupts the fourth character, so any
        # prefix long enough to be unambiguous is already long enough to miss.
        import difflib

        hits = difflib.get_close_matches(key, list(lowered), n=2, cutoff=0.7)
        if len(hits) == 1 or (hits and hits[0] != (hits[1] if len(hits) > 1 else None)
                              and difflib.SequenceMatcher(None, key, hits[0]).ratio()
                              > difflib.SequenceMatcher(None, key, hits[1]).ratio()):
            print(f"  note: {what} column '{n}' matched '{lowered[hits[0]]}'")
            out.append(lowered[hits[0]])
        else:
            raise SystemExit(
                f"cannot find {what} column '{n}'. Available: {list(df.columns)}")
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Pearson, Spearman and n over the rows where both sides are present."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    xs, ys = x[ok], y[ok]
    if np.std(xs) == 0 or np.std(ys) == 0:
        # A constant column -- an abstained channel, or a grader who used one
        # value. Correlation is undefined, not zero; reporting 0 would read as
        # "measured and unrelated", which is a different and much worse claim.
        return float("nan"), float("nan"), int(ok.sum())
    return (float(stats.pearsonr(xs, ys)[0]),
            float(stats.spearmanr(xs, ys)[0]),
            int(ok.sum()))


def _centre(df: pd.DataFrame, col: str, by: str) -> np.ndarray:
    """Subtract the per-question mean, leaving within-question variation."""
    v = pd.to_numeric(df[col], errors="coerce")
    return (v - v.groupby(df[by]).transform("mean")).to_numpy(dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="one sheet holding both sides")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--pred", required=True, help="our 4 columns, in CATS order")
    ap.add_argument("--true", required=True, dest="true_cols",
                    help="the 4 human columns, same order")
    ap.add_argument("--question", default=None,
                    help="question id column (enables the within-question split)")
    ap.add_argument("--length", default=None,
                    help="a word-count or duration column, for the length check")
    ap.add_argument("--confidence-suffix", default="_confidence")
    args = ap.parse_args()

    df = _read(Path(args.file), args.sheet)
    pred = _resolve(df, args.pred.split(","), "prediction")
    true = _resolve(df, args.true_cols.split(","), "label")
    qcol = _resolve(df, [args.question], "question")[0] if args.question else None
    lcol = _resolve(df, [args.length], "length")[0] if args.length else None

    print(f"\nrows in file: {len(df)}")

    # ---- label spread: is r being squeezed by the graders' own range? ----
    print("\nLABEL DISTRIBUTION  (r scales with this spread; a narrow band caps it)")
    print(f"  {'category':<12} {'n':>4} {'mean':>7} {'sd':>6} {'min':>6} {'max':>6}  values used")
    for cat, tc in zip(CATS, true):
        v = pd.to_numeric(df[tc], errors="coerce").dropna()
        uniq = sorted(v.unique())
        shown = ", ".join(f"{u:g}" for u in uniq[:8]) + (" ..." if len(uniq) > 8 else "")
        print(f"  {cat:<12} {len(v):>4} {v.mean():>7.2f} {v.std():>6.2f} "
              f"{v.min():>6.2f} {v.max():>6.2f}  {shown}")

    # ---- abstentions: constants masquerading as bad accuracy ----
    print("\nOUR SCORES  (a channel that abstained is a constant, not a weak signal)")
    print(f"  {'category':<12} {'n':>4} {'mean':>7} {'sd':>6} {'zeros':>6} {'conf':>6}")
    for cat, pc in zip(CATS, pred):
        v = pd.to_numeric(df[pc], errors="coerce")
        conf_col = next((c for c in df.columns
                         if str(c).lower() == f"{cat}{args.confidence_suffix}"), None)
        conf = pd.to_numeric(df[conf_col], errors="coerce").mean() if conf_col else float("nan")
        print(f"  {cat:<12} {v.notna().sum():>4} {v.mean():>7.2f} {v.std():>6.2f} "
              f"{int((v == 0).sum()):>6} {conf:>6.2f}")

    # ---- the headline, taken apart ----
    print("\nAGREEMENT")
    hdr = f"  {'category':<12} {'pearson':>8} {'spearman':>9} {'n':>5}"
    if qcol:
        hdr += f" | {'within-q r':>10} {'within-q rho':>12}"
    print(hdr)
    for cat, pc, tc in zip(CATS, pred, true):
        x = pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[tc], errors="coerce").to_numpy(dtype=float)
        r, rho, n = _corr(x, y)
        line = f"  {cat:<12} {r:>8.3f} {rho:>9.3f} {n:>5}"
        if qcol:
            wr, wrho, _ = _corr(_centre(df, pc, qcol), _centre(df, tc, qcol))
            line += f" | {wr:>10.3f} {wrho:>12.3f}"
        print(line)
    if qcol:
        print("  within-q removes each question's mean from both sides: agreement on")
        print("  ranking candidates who answered the SAME prompt, which is the")
        print("  decision the score is actually used for.")

    # ---- per question: a single bad rubric can sink a pooled number ----
    if qcol:
        print("\nPER QUESTION  (pearson; '--' means one side had no spread)")
        print(f"  {'question':<14} {'n':>4} " + " ".join(f"{c[:8]:>9}" for c in CATS))
        for q, sub in df.groupby(df[qcol]):
            if len(sub) < 5:
                continue
            cells = []
            for pc, tc in zip(pred, true):
                r, _, _ = _corr(pd.to_numeric(sub[pc], errors="coerce").to_numpy(float),
                                pd.to_numeric(sub[tc], errors="coerce").to_numpy(float))
                cells.append(f"{r:>9.3f}" if np.isfinite(r) else f"{'--':>9}")
            print(f"  {str(q)[:14]:<14} {len(sub):>4} " + " ".join(cells))

    # ---- are four scores really four things? ----
    for label, cols in (("OURS", pred), ("THEIRS", true)):
        print(f"\nINTER-CORRELATION, {label}  (high everywhere = one factor, not four)")
        m = df[cols].apply(pd.to_numeric, errors="coerce")
        cm = m.corr(method="spearman")
        print("  " + " " * 12 + " ".join(f"{c[:8]:>9}" for c in CATS))
        for i, cat in enumerate(CATS):
            print(f"  {cat:<12} " + " ".join(f"{cm.iloc[i, j]:>9.3f}"
                                             for j in range(len(CATS))))

    # ---- is length doing the work? ----
    if lcol:
        print("\nLENGTH CONFOUND  (compare each pair against the agreement above)")
        length = pd.to_numeric(df[lcol], errors="coerce").to_numpy(dtype=float)
        print(f"  {'category':<12} {'ours~len':>9} {'theirs~len':>11}")
        for cat, pc, tc in zip(CATS, pred, true):
            a, _, _ = _corr(pd.to_numeric(df[pc], errors="coerce").to_numpy(float), length)
            b, _, _ = _corr(pd.to_numeric(df[tc], errors="coerce").to_numpy(float), length)
            print(f"  {cat:<12} {a:>9.3f} {b:>11.3f}")
        print("  If ours~len is close to our agreement with the labels, the score is")
        print("  mostly counting words. If theirs~len is also high, so are the graders.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
