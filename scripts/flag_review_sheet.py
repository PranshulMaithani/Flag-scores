"""Summarise the flag columns, and build a blind review sheet from them.

Two situations produce "no flag output", and they are not the same problem:

* The **Flags sheet is empty or absent** -- a pipeline or export fault, and
  nothing downstream is trustworthy until it is fixed.
* The sheet is full of numbers and **nothing crossed a threshold**. On a few
  hundred ordinary responses that is the expected result: reading the prompt
  aloud, padding by repetition, going off topic and switching language are all
  rare. A screen of zeros in the `*_fired` columns looks like no output.

The second case is not reassurance. A flag that never fires has not been shown
to be right; it has been shown to be quiet, and a detector wired to a constant
zero is also quiet. Distinguishing the two needs a human to look at the items the
flag ranked highest, whether or not they crossed the line.

So the review samples on the **score**, not on `fired`: the top of each flag's
distribution, plus a random draw from the rest as controls. Rows are shuffled and
the flag's own score is withheld, so the reviewer cannot tell which side an item
came from. Without that, knowing an item is a "suspected" one is enough to make
it look suspicious, and the precision number measures the labelling setup rather
than the detector.

    python scripts/flag_review_sheet.py --file voxscore_results.xlsx --top 25 --controls 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FLAGS = ("prompt_read", "repetition", "off_topic", "foreign_language")
THRESHOLDS = {"prompt_read": 55, "repetition": 55, "off_topic": 35,
              "foreign_language": 30}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="the scored workbook")
    ap.add_argument("--sheet", default="Flags")
    ap.add_argument("--scores-sheet", default="Scores",
                    help="used only to attach the transcript for the reviewer")
    ap.add_argument("--top", type=int, default=25, help="highest-scoring per flag")
    ap.add_argument("--controls", type=int, default=25, help="random draws per flag")
    ap.add_argument("--out", default="flag_review.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.file)
    try:
        df = pd.read_excel(path, sheet_name=args.sheet)
    except ValueError:
        book = pd.ExcelFile(path)
        raise SystemExit(
            f"no '{args.sheet}' sheet in {path.name}. Sheets present: "
            f"{book.sheet_names}")

    print(f"\n'{args.sheet}' sheet: {len(df)} rows, {len(df.columns)} columns")
    if df.empty:
        raise SystemExit(
            "the sheet is empty. That is an export fault, not a quiet detector: "
            "the pipeline builds all four flags for every item unconditionally, "
            "so an empty sheet means results carried no 'flags' key -- most "
            "likely scoring ran on a build older than the flag code, with cached "
            "per-item JSON reused. Rerun with --rescore.")

    present = [f for f in FLAGS if f in df.columns]
    missing = [f for f in FLAGS if f not in df.columns]
    if missing:
        print(f"  MISSING flag columns: {missing}")
        print(f"  columns found: {list(df.columns)}")
    if not present:
        raise SystemExit("none of the four flag columns are present")

    print("\nFLAG SCORE DISTRIBUTIONS  (0-100, higher = more suspicious)")
    print(f"  {'flag':<18} {'n':>5} {'mean':>7} {'sd':>6} {'p90':>6} {'max':>6} "
          f"{'thresh':>7} {'fired':>6}")
    for f in present:
        v = pd.to_numeric(df[f], errors="coerce").dropna()
        if v.empty:
            print(f"  {f:<18} {'all values non-numeric or blank'}")
            continue
        th = THRESHOLDS[f]
        fired_col = f"{f}_fired"
        fired = (int(pd.to_numeric(df[fired_col], errors="coerce").sum())
                 if fired_col in df.columns else int((v >= th).sum()))
        print(f"  {f:<18} {len(v):>5} {v.mean():>7.1f} {v.std():>6.1f} "
              f"{np.percentile(v, 90):>6.1f} {v.max():>6.1f} {th:>7} {fired:>6}")
        if v.std() == 0:
            print(f"      ^ constant. Not a quiet detector -- a dead one.")
        elif v.max() < th:
            print(f"      ^ nothing reached the threshold; review the top of the "
                  f"range anyway, since the ranking can be right while the cut "
                  f"is set too high")

    # Attach transcripts so the reviewer can judge without opening the audio.
    try:
        sc = pd.read_excel(path, sheet_name=args.scores_sheet)
        keep = [c for c in ("item_id", "transcript", "duration_s") if c in sc.columns]
        df = df.merge(sc[keep], on="item_id", how="left")
    except Exception as exc:
        print(f"\n  note: could not attach transcripts ({exc}); "
              f"the reviewer will need the audio")

    rng = np.random.default_rng(args.seed)
    rows = []
    for f in present:
        v = pd.to_numeric(df[f], errors="coerce")
        ranked = df.assign(_v=v).dropna(subset=["_v"]).sort_values("_v", ascending=False)
        if ranked.empty:
            continue
        top = ranked.head(args.top)
        rest = ranked.iloc[args.top:]
        n_ctrl = min(args.controls, len(rest))
        ctrl = rest.iloc[rng.choice(len(rest), n_ctrl, replace=False)] if n_ctrl else rest

        for group, part in (("top", top), ("control", ctrl)):
            for _, r in part.iterrows():
                rows.append({
                    "review_id": "",                      # filled after shuffling
                    "flag": f,
                    "item_id": r.get("item_id", ""),
                    # The reviewer must not see the score or the group; both are
                    # kept for scoring the review afterwards and dropped from the
                    # sheet the reviewer opens.
                    "_score": round(float(r["_v"]), 2),
                    "_group": group,
                    "evidence": str(r.get(f"{f}_why", ""))[:220],
                    "duration_s": r.get("duration_s", ""),
                    "transcript": str(r.get("transcript", ""))[:1200],
                    "verdict": "",                        # yes / no / unsure
                    "reviewer_note": "",
                })

    if not rows:
        raise SystemExit("nothing to review")

    out = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out["review_id"] = np.arange(1, len(out) + 1)

    key_path = Path(args.out).with_suffix(".key.csv")
    out[["review_id", "flag", "item_id", "_score", "_group"]].to_csv(key_path, index=False)
    out.drop(columns=["_score", "_group"]).to_csv(args.out, index=False)

    print(f"\nwrote {args.out}  ({len(out)} rows to review)")
    print(f"wrote {key_path}  -- keep this away from the reviewer until they finish")
    print("""
For the reviewer, one question per row, answered from the transcript:
  prompt_read        did they mostly recite or paraphrase the question to fill time?
  repetition         did they pad by repeating themselves, beyond normal speech?
  off_topic          did they answer a different question than the one asked?
  foreign_language   did they speak something other than English for a real stretch?
Answer yes / no / unsure. 'unsure' is a real answer; forcing a call on an
ambiguous item adds noise to the estimate rather than removing it.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
