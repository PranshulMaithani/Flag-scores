"""Does relevance RANK, not just discriminate?

The open risk. We can show the system separates on-topic from off-topic
perfectly, but the client's 0-5 relevance labels encode something finer: whether
an on-topic answer is a *good* one. The proficiency fixture could not test this
because it holds relevance constant while varying proficiency.

`data/synthetic/graded_relevance.json` is the complement: five responses per
question at graded relevance, written at a similar proficiency level, so
relevance is the only thing moving.

Run with --no-ideals (default) to score exactly as production will, with no ideal
answers at all.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_relevance_ranking.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import REPORTS_DIR
from voxscore.features.embed import NLI, Embedder
from voxscore.features.qtype import profile_question
from voxscore.features.relevance import build_rubric, relevance_features
from voxscore.flags.detectors import off_topic_flag
from voxscore.scoring.aggregate import score_relevance
from voxscore.utils import textproc as tp


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise(scores, truth) -> tuple[int, int]:
    ok = tot = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if truth[i] == truth[j]:
                continue
            tot += 1
            hi, lo = (i, j) if truth[i] > truth[j] else (j, i)
            ok += scores[hi] > scores[lo]
    return ok, tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideals", action="store_true",
                    help="also score with our own authored ideal answers, to measure "
                         "what the optional path adds")
    args = ap.parse_args()

    data = json.loads(Path("data/synthetic/graded_relevance.json").read_text(encoding="utf-8"))
    own = {}
    if args.ideals:
        own = json.loads(
            Path("data/synthetic/own_ideal_answers.json").read_text(encoding="utf-8"))
    nlp, emb, nli = tp.get_nlp(), Embedder(), NLI()

    print("=" * 92)
    print("RELEVANCE RANKING - proficiency held constant, relevance varied")
    print("=" * 92)

    all_s, all_t, per_q = [], [], {}
    for qid, block in data.items():
        if qid.startswith("_"):
            continue
        qtext = block["question"]
        qp = profile_question(qtext, nlp)
        rub = build_rubric(qid, qtext, None, emb, nlp)
        rub_i = None
        if args.ideals and qid in own:
            rub_i = build_rubric(qid, qtext, own[qid]["ideal_answers"], emb, nlp)

        print(f"\n{qid}  [{qp.family}  arg={qp.argumentativeness:.2f} "
              f"narr={qp.narrativity:.2f} expl={int(qp.wants_explanation)}]")
        print(f"  {qtext}")
        print(f"\n  {'truth':>6s}{'relevance':>11s}{'off_topic':>11s}   "
              f"{'stance':>7s}{'reasons':>8s}{'specif':>8s}  first words")
        print("  " + "-" * 88)

        s_list, t_list = [], []
        for r in block["responses"]:
            p = tp.parse(r["text"], nlp)
            f = relevance_features(p, rub, emb, nli)
            score = score_relevance(f).score
            ot = off_topic_flag(f).score
            s_list.append(score); t_list.append(r["relevance"])
            print(f"  {r['relevance']:>6d}{score:>11.1f}{ot:>11.1f}   "
                  f"{f.get('stance_clarity', 0):>7.2f}{f.get('reason_count', 0):>8.1f}"
                  f"{f.get('specificity', 0):>8.2f}  {r['text'][:34]}...")

        rho = spearman(s_list, t_list)
        ok, tot = pairwise(s_list, t_list)
        per_q[qid] = {"spearman": rho, "pairwise": ok / max(tot, 1),
                      "scores": s_list, "truth": t_list}
        print(f"\n  Spearman {rho:+.3f}   pairwise {ok}/{tot} = {ok / max(tot, 1):.0%}")
        all_s += s_list; all_t += t_list

    rho = spearman(all_s, all_t)
    ok, tot = pairwise(all_s, all_t)
    print("\n" + "=" * 92)
    print(f"POOLED   Spearman {rho:+.3f}   pairwise {ok}/{tot} = {ok / max(tot, 1):.0%}")
    print("=" * 92)
    print("\nScored with NO ideal answers - exactly the production configuration.")

    out = REPORTS_DIR / "relevance_ranking.json"
    out.write_text(json.dumps(
        {"per_question": per_q, "pooled_spearman": rho,
         "pooled_pairwise": ok / max(tot, 1)}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
