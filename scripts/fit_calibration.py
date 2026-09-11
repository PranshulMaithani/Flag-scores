"""Fit a calibration model on labelled data, and report honestly whether it helps.

The default scorer is an unfitted scorecard (`voxscore/scoring/aggregate.py`),
chosen because with zero client training data any fitted head would be
unvalidatable. This script is what you run **if and when** you decide to fit.

It deliberately makes overfitting hard to hide:

* **Grouped cross-validation by question.** Items sharing a question are not
  split across folds. Without this the model can learn per-question quirks and
  report a correlation it will not reproduce on a new question -- which is the
  failure mode that matters, since new prompts are added over time.
* **The unfitted scorecard is the baseline**, reported side by side. A fitted
  model that does not beat it is not worth the loss of explainability.
* **Ridge by default**, with few features and strong regularisation. On a few
  hundred items, gradient boosting will look better in-fold and generalise worse.
* **Spearman is the headline**, because that is what the client measures.

Usage
-----
    python scripts/fit_calibration.py --features features.csv --labels labels.csv

``features.csv``  one row per item: ``item_id``, ``question_id``, then feature columns
                  (produce it with ``scripts/extract_features.py``)
``labels.csv``    ``item_id`` plus one column per category, any numeric scale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import REPORTS_DIR
from voxscore.scoring.aggregate import CATEGORIES, score_category

# Features offered to each category's model. Kept small on purpose: with a few
# hundred labelled items, feature count is the main overfitting risk.
CANDIDATES = {
    "grammar": ["errors_per_100_words", "error_free_sentence_ratio",
                "clauses_per_sentence", "mean_dependency_distance",
                "mean_sentence_len", "err_article_determiner_per100",
                "err_preposition_per100", "err_subject_verb_agreement_per100"],
    "lexical": ["mtld", "mattr_50", "hdd", "mean_log_freq", "pct_beyond_zipf4",
                "lexical_density", "rare_word_rate", "content_types"],
    "fluency": ["speech_rate_wpm", "articulation_rate_wps", "phonation_time_ratio",
                "mean_length_of_run", "silent_pause_rate", "long_pause_rate",
                "within_clause_pause_ratio", "filled_pause_rate"],
    "relevance": ["sim_q", "element_coverage", "move_coverage", "specificity",
                  "content_novelty_vs_q", "profile_match", "distinct_content_rate",
                  "pct_windows_offtopic", "min_window_sim", "nli_move_entailment"],
}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def grouped_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx) with every question confined to one fold."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, min(n_splits, len(uniq)))
    for held in chunks:
        test = np.isin(groups, held)
        if test.any() and (~test).any():
            yield np.where(~test)[0], np.where(test)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="ridge regularisation; higher is safer on small data")
    ap.add_argument("--out", default=str(REPORTS_DIR / "calibration.json"))
    args = ap.parse_args()

    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    feats = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    df = feats.merge(labels, on="item_id", suffixes=("", "_label"))
    print(f"{len(df)} items, {df['question_id'].nunique()} distinct questions\n")

    if df["question_id"].nunique() < 3:
        print("WARNING: fewer than 3 distinct questions. Grouped CV cannot estimate "
              "generalisation to a NEW question, which is the number that matters.\n")

    groups = df["question_id"].to_numpy()
    results = {}

    print(f"{'category':11s}{'n':>5s}{'baseline r':>12s}{'baseline rho':>14s}"
          f"{'fitted r':>11s}{'fitted rho':>12s}{'verdict':>12s}")
    print("-" * 78)

    for cat in CATEGORIES:
        if cat not in labels.columns:
            continue
        cols = [c for c in CANDIDATES[cat] if c in df.columns]
        if not cols:
            print(f"{cat:11s}  no feature columns present; skipped")
            continue

        sub = df.dropna(subset=cols + [cat])
        if len(sub) < 20:
            print(f"{cat:11s}  only {len(sub)} usable rows; skipped")
            continue

        X = sub[cols].to_numpy(float)
        y = sub[cat].to_numpy(float)
        g = sub["question_id"].to_numpy()

        # Baseline: the unfitted scorecard, scored from the same rows.
        base = np.array([
            score_category(cat, {c: r[c] for c in cols}).score
            for _, r in sub.iterrows()
        ])

        # Fitted, out-of-fold only -- never in-sample.
        pred = np.full(len(sub), np.nan)
        for tr, te in grouped_folds(g):
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=args.alpha).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.predict(sc.transform(X[te]))

        b_r, b_rho = pearson(base, y), spearman(base, y)
        f_r, f_rho = pearson(pred, y), spearman(pred, y)
        better = (f_rho - b_rho) > 0.03
        verdict = "fitted" if better else "keep scorecard"

        results[cat] = {
            "n": int(len(sub)), "features": cols,
            "baseline": {"pearson": b_r, "spearman": b_rho},
            "fitted_oof": {"pearson": f_r, "spearman": f_rho},
            "recommendation": verdict,
        }
        print(f"{cat:11s}{len(sub):5d}{b_r:12.3f}{b_rho:14.3f}"
              f"{f_r:11.3f}{f_rho:12.3f}{verdict:>12s}")

    print("\nNotes")
    print("  * Fitted figures are out-of-fold with questions held out whole, so they")
    print("    estimate generalisation to a NEW question rather than a new candidate.")
    print("  * 'keep scorecard' means fitting bought less than 0.03 Spearman, which")
    print("    does not justify losing per-feature explainability.")
    print("  * Compare both columns against the incumbent's per-category correlation")
    print("    on the SAME items; anything else is not a like-for-like comparison.")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
