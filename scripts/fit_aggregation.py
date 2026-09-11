"""Test whether the unfitted scorecard, not the features, is the bottleneck.

The scorecard maps features to a 0-100 score through hand-chosen piecewise-linear
breakpoints. That was forced: with no labels, there was nothing to fit to, and a
transparent unfitted rule is defensible in a way a guessed one is not. It is also
the obvious suspect when the features look sound and the scores correlate weakly.

This fits a regularised linear model from the same features to the human labels
and reports **out-of-fold** correlation -- every item predicted by a model that
never saw it -- against the unfitted score as baseline. The gap between the two
is what the aggregation is costing.

Three things it does that a naive fit would get wrong:

* **Groups folds by candidate.** One person's three answers share an accent, a
  microphone and a proficiency. Splitting them across folds lets the model
  recognise the speaker rather than the speech, and inflates the result.
* **Fits each category on its own block first, then on everything.** If grammar
  labels are predicted just as well by fluency features as by grammar features,
  the graders were not scoring four things, and no amount of feature work will
  separate them. That comparison is the point, not the headline number.
* **Reports the baseline on the identical folds.** An out-of-fold number compared
  against an in-sample one is not a comparison.

A caveat that has to travel with the output: fitting on these items spends them.
They were held as a clean test set precisely so the reported numbers would not be
optimistic. Out-of-fold CV is the honest way to spend them -- but "honest" here
means an unbiased estimate of what this model does on new data from the same
questions and graders, not a guarantee for new questions.

    python scripts/fit_aggregation.py --file scored.xlsx --sheet features \\
        --labels labels.csv --group anon_id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CATS = ("relevance", "fluency", "lexical", "grammar")

# Which feature block belongs to which score. Blocks are the prefixes written by
# write_excel(): "grammar__err_preposition_per100" and so on.
BLOCKS = {
    "relevance": ("relevance",),
    "fluency": ("fluency", "prosody"),
    "lexical": ("lexical",),
    "grammar": ("grammar",),
}


def _read(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"cannot decode {path}")


def _numeric_block(df: pd.DataFrame, prefixes: tuple[str, ...] | None) -> pd.DataFrame:
    """Numeric feature columns, optionally restricted to given blocks."""
    cols = []
    for c in df.columns:
        name = str(c)
        if "__" not in name:
            continue
        # Columns the merge had to rename are the same feature arriving twice.
        # Feeding both halves of a duplicated pair to a ridge splits one
        # coefficient across two collinear columns -- harmless for the
        # prediction, misleading in the feature count printed beside it.
        if name.endswith(".lab"):
            continue
        if prefixes and not name.startswith(tuple(p + "__" for p in prefixes)):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # A feature that never varies carries nothing and destabilises the fit.
        if s.notna().sum() > 0 and s.std(skipna=True) > 0:
            cols.append(c)
    return df[cols].apply(pd.to_numeric, errors="coerce")


def _oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int) -> np.ndarray:
    """Out-of-fold predictions from a ridge fit, grouped so a speaker never splits."""
    pred = np.full(len(y), np.nan)
    ok = np.isfinite(y) & X.notna().all(axis=1).to_numpy()
    Xo, yo, go = X[ok], y[ok], groups[ok]
    n_splits = min(folds, len(np.unique(go)))
    if n_splits < 2:
        return pred

    idx = np.flatnonzero(ok)
    for tr, te in GroupKFold(n_splits=n_splits).split(Xo, yo, go):
        model = make_pipeline(
            StandardScaler(),
            # A wide alpha grid matters more than the model class here: with ~350
            # items and dozens of features, how hard it is regularised dominates.
            RidgeCV(alphas=np.logspace(-2, 4, 25)),
        )
        model.fit(Xo.iloc[tr], yo[tr])
        pred[idx[te]] = model.predict(Xo.iloc[te])
    return pred


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), float("nan"), int(ok.sum())
    return (float(stats.pearsonr(x[ok], y[ok])[0]),
            float(stats.spearmanr(x[ok], y[ok])[0]), int(ok.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="the features table")
    ap.add_argument("--sheet", default="features")
    ap.add_argument("--labels", required=True,
                    help="file with the human labels; joined on --key")
    ap.add_argument("--labels-sheet", default=None)
    ap.add_argument("--key", default="item_id")
    ap.add_argument("--group", default="anon_id",
                    help="fold grouping column; use the candidate, not the item")
    ap.add_argument("--true", default=None,
                    help="4 label columns in relevance,fluency,lexical,grammar "
                         "order (default: look for the category names)")
    ap.add_argument("--scores", default=None,
                    help="optional sheet/file holding the unfitted 0-100 scores, "
                         "if they are not already in --labels")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    feats = _read(Path(args.file), args.sheet)
    labels = _read(Path(args.labels), args.labels_sheet)

    for frame, what in ((feats, "features"), (labels, "labels")):
        if args.key not in frame.columns:
            raise SystemExit(f"{what} table has no '{args.key}' column; "
                             f"columns are {list(frame.columns)}")

    df = feats.merge(labels, on=args.key, how="inner", suffixes=("", ".lab"))
    if df.empty:
        raise SystemExit(f"no rows matched on '{args.key}' between the two files")

    if args.group in df.columns:
        groups = df[args.group].astype(str).to_numpy()
    else:
        # No candidate column: fall back to per-item folds and say so, because a
        # reader would otherwise assume the stronger guarantee.
        print(f"  warning: no '{args.group}' column; folds are per item, so a "
              f"candidate's answers can straddle a split and the numbers below "
              f"are optimistic")
        groups = np.arange(len(df)).astype(str)

    true_cols = args.true.split(",") if args.true else None
    lowered = {str(c).strip().lower(): c for c in df.columns}

    print(f"\nrows matched: {len(df)}   candidates: {len(np.unique(groups))}   "
          f"folds: {args.folds}")
    print("\nOUT-OF-FOLD CORRELATION WITH THE HUMAN LABELS")
    print(f"  {'category':<12} {'unfitted':>9} {'own block':>10} {'all feats':>10} "
          f"{'n feat':>7} {'n':>5}")

    for i, cat in enumerate(CATS):
        if true_cols:
            tcol = lowered.get(true_cols[i].strip().lower())
        else:
            tcol = next((lowered[k] for k in lowered
                         if k in {cat, f"human_{cat}", f"{cat}_label",
                                  f"{cat}_human"}), None)
        if tcol is None:
            print(f"  {cat:<12} no label column found")
            continue

        y = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float)

        # Baseline: the unfitted score, on the same rows.
        base = np.full(len(df), np.nan)
        scol = lowered.get(f"{cat}_score") or lowered.get(f"{cat}_unfitted")
        if scol is None and not true_cols:
            scol = None                       # the label took the plain name
        if scol is not None:
            base = pd.to_numeric(df[scol], errors="coerce").to_numpy(dtype=float)
        br, _, _ = _corr(base, y)

        own = _numeric_block(df, BLOCKS[cat])
        allf = _numeric_block(df, None)
        own = own.fillna(own.median(numeric_only=True))
        allf = allf.fillna(allf.median(numeric_only=True))

        own_r, _, n = _corr(_oof(own, y, groups, args.folds), y)
        all_r, _, _ = _corr(_oof(allf, y, groups, args.folds), y)

        def fmt(v: float, w: int) -> str:
            return f"{v:>{w}.3f}" if np.isfinite(v) else f"{'--':>{w}}"

        print(f"  {cat:<12} {fmt(br, 9)} {fmt(own_r, 10)} {fmt(all_r, 10)} "
              f"{own.shape[1]:>7} {n:>5}")

    print("""
Reading this:
  own block > unfitted        the scorecard was the bottleneck, not the features
  own block ~ unfitted        the features are the bottleneck; the curves are fine
  all feats >> own block      that category's label is being predicted by other
                              categories' features, i.e. the graders scored one
                              thing and split it four ways after the fact
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
