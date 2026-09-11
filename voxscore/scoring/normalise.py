"""Per-question score centring, for pooled correlation.

Measured on the graded fixture: relevance ranks **perfectly within each question**
(Spearman +1.000 on both a personal and an opinion prompt) but only +0.964 when
the two questions are pooled. The residue is not mis-ranking — it is that the
same true relevance level lands on slightly different score values depending on
the prompt, so cross-question comparisons blur.

That matters because the client is likely to pool all ~350 items into one
correlation against their labels. If they do, every cross-question pair carries
that blur.

Whether centring *helps* depends on something we cannot know without their data:
whether their human raters scored each response **against its own prompt**
(in which case centring matches what the raters did) or on one absolute scale
across all prompts (in which case centring removes real signal). Rather than
guess, both columns are emitted and the client can compute the correlation each
way in one line. The comparison is itself informative — if centring improves
correlation, their raters were grading relative to the prompt.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

CATEGORIES = ("grammar", "lexical", "fluency", "relevance")

# Below this many responses for a question, its mean is too noisy to centre on
# and centring would inject noise rather than remove it.
MIN_PER_QUESTION = 5


def centre_by_question(
    rows: list[dict],
    categories: tuple[str, ...] = CATEGORIES,
    target_mean: float = 50.0,
    target_sd: float = 15.0,
) -> list[dict]:
    """Add ``<category>_centred`` to each row, z-scored within its question.

    Rows are dicts with at least ``question_id`` and the category score keys.
    Questions with fewer than :data:`MIN_PER_QUESTION` scorable responses are
    passed through uncentred, and marked so, rather than centred on a mean
    estimated from three items.
    """
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("scorable", 1):
            by_q[r.get("question_id") or "?"].append(i)

    for cat in categories:
        for qid, idxs in by_q.items():
            vals = np.array(
                [rows[i].get(cat, np.nan) for i in idxs], dtype=float
            )
            ok = np.isfinite(vals)
            if ok.sum() < MIN_PER_QUESTION or vals[ok].std() < 1e-6:
                for i in idxs:
                    rows[i][f"{cat}_centred"] = rows[i].get(cat)
                    rows[i]["centring"] = f"skipped (n={int(ok.sum())})"
                continue
            mu, sd = float(vals[ok].mean()), float(vals[ok].std())
            for i in idxs:
                v = rows[i].get(cat)
                rows[i][f"{cat}_centred"] = (
                    float(np.clip(target_mean + target_sd * (v - mu) / sd, 0, 100))
                    if v is not None and np.isfinite(v) else None
                )
                rows[i]["centring"] = f"z within {qid} (n={int(ok.sum())})"

    # Rows for unscorable items never get centred values; make that explicit
    # rather than leaving the key missing.
    for r in rows:
        for cat in categories:
            r.setdefault(f"{cat}_centred", None)
        r.setdefault("centring", "not scorable")
    return rows


def summarise(rows: list[dict], categories: tuple[str, ...] = CATEGORIES) -> str:
    """One-line-per-category report of what centring changed."""
    out = [f"{'category':12s}{'n':>6s}{'raw mean':>11s}{'raw sd':>9s}"
           f"{'centred mean':>14s}{'centred sd':>12s}"]
    out.append("-" * 64)
    for cat in categories:
        raw = np.array([r[cat] for r in rows
                        if r.get(cat) is not None and np.isfinite(r.get(cat, np.nan))], float)
        cen = np.array([r[f"{cat}_centred"] for r in rows
                        if r.get(f"{cat}_centred") is not None], float)
        if raw.size:
            out.append(f"{cat:12s}{raw.size:6d}{raw.mean():11.1f}{raw.std():9.1f}"
                       f"{(cen.mean() if cen.size else float('nan')):14.1f}"
                       f"{(cen.std() if cen.size else float('nan')):12.1f}")
    return "\n".join(out)
