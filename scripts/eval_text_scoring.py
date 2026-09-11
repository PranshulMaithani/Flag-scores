"""Text-level validation: ordinal score agreement, and flag ROC curves.

Two things are measured, neither of which needs client labels.

**1. Ordinal validity.** Each question has three authored responses at weak / mid
/ strong proficiency. A valid scorer must rank them in that order. Since the
client evaluates by *correlation*, ordinal agreement is direct evidence of the
thing they will measure, not a proxy for it. Reported as Spearman rho against the
intended level and as pairwise accuracy.

**2. Flag ROC.** Gaming variants are generated from the genuine responses --
prompt echo, padded repetition, and mismatched questions -- giving labelled
positives whose *only* difference from the negatives is the gaming behaviour.
Recommended thresholds come from the resulting curves.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_text_scoring.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import PROCESSED_DIR, REPORTS_DIR
from voxscore.eval.synth import make_prompt_echo, make_text_repetition
from voxscore.features.embed import NLI, Embedder
from voxscore.features.grammar import GrammarScorer, grammar_features
from voxscore.features.lexical import lexical_features
from voxscore.features.relevance import build_rubric, relevance_features
from voxscore.flags.detectors import off_topic_flag, prompt_read_flag, repetition_flag
from voxscore.scoring.aggregate import score_category
from voxscore.utils import textproc as tp

LEVELS = {"weak": 1, "mid": 2, "strong": 3}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def roc_auc(scores, labels) -> float:
    """AUC via the rank formulation; no sklearn dependency for one number."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_threshold(scores, labels) -> tuple[float, float, float, float]:
    """Threshold maximising Youden's J. Returns (threshold, tpr, fpr, J)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    best = (55.0, 0.0, 0.0, -1.0)
    for t in np.arange(0, 100.5, 1.0):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fn = int((~pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        tn = int((~pred & (y == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best[3]:
            best = (float(t), tpr, fpr, j)
    return best


def main() -> int:
    qs = json.loads(Path("data/raw/questions_sample.json").read_text(encoding="utf-8"))
    data = json.loads(Path("data/synthetic/authored_responses.json").read_text(encoding="utf-8"))
    responses = {k: v for k, v in data.items() if not k.startswith("_")}

    nlp = tp.get_nlp()
    emb = Embedder()
    nli = NLI()
    gec = GrammarScorer()
    rng = random.Random(7)

    rubrics = {
        qid: build_rubric(qid, qs[qid]["text"], qs[qid].get("ideal_answers"), emb, nlp)
        for qid in responses if qid in qs
    }

    # ---------------- 1. ordinal validity ----------------
    print("=" * 92)
    print("ORDINAL VALIDITY - does the scorer rank weak < mid < strong?")
    print("=" * 92)
    print(f"\n{'question':22s}{'level':8s}{'gram':>7s}{'lex':>7s}{'flu*':>7s}{'rel':>7s}")
    print("-" * 92)

    per_cat: dict[str, list] = {c: [] for c in ("grammar", "lexical", "relevance")}
    intended: list[int] = []
    rows = []

    for qid, levels in responses.items():
        if qid not in rubrics:
            continue
        for level in ("weak", "mid", "strong"):
            text = levels[level]
            p = tp.parse(text, nlp)
            gf = grammar_features(p, gec, variant="spoken")
            lf = lexical_features(p)
            rf = relevance_features(p, rubrics[qid], emb, nli)

            g = score_category("grammar", gf).score
            l = score_category("lexical", lf).score
            r = score_category("relevance", rf).score

            per_cat["grammar"].append(g)
            per_cat["lexical"].append(l)
            per_cat["relevance"].append(r)
            intended.append(LEVELS[level])
            rows.append({"qid": qid, "level": level, "grammar": g, "lexical": l,
                         "relevance": r, "text": text})
            print(f"{qid[:21]:22s}{level:8s}{g:7.1f}{l:7.1f}{'-':>7s}{r:7.1f}")

    print("\n* fluency needs audio; excluded from this text-only evaluation.\n")
    print(f"{'category':14s}{'spearman':>10s}{'pairwise acc':>15s}")
    print("-" * 40)
    summary = {}
    for cat, scores in per_cat.items():
        rho = spearman(scores, intended)
        # Pairwise accuracy within each question, which is the cleaner measure:
        # it never compares across questions of differing difficulty.
        correct = total = 0
        for i in range(0, len(scores), 3):
            trio = scores[i:i + 3]
            if len(trio) < 3:
                continue
            for a in range(3):
                for b in range(a + 1, 3):
                    total += 1
                    correct += int(trio[b] > trio[a])
        acc = correct / max(total, 1)
        summary[cat] = {"spearman": rho, "pairwise_accuracy": acc}
        print(f"{cat:14s}{rho:10.3f}{acc:15.1%}")

    # ---------------- 2. flag ROC ----------------
    print("\n" + "=" * 92)
    print("FLAG ROC - gaming variants generated from the same genuine responses")
    print("=" * 92)

    pr_s, pr_y, rep_s, rep_y, ot_s, ot_y = [], [], [], [], [], []
    qids = list(rubrics)

    for qid, levels in responses.items():
        if qid not in rubrics:
            continue
        qtext = qs[qid]["text"]
        for level, text in levels.items():
            # --- negatives: the genuine response, correctly paired ---
            p = tp.parse(text, nlp)
            rf = relevance_features(p, rubrics[qid], emb, nli)
            pr_s.append(prompt_read_flag(p, qtext, rf).score); pr_y.append(0)
            rep_s.append(repetition_flag(p, emb).score); rep_y.append(0)
            ot_s.append(off_topic_flag(rf).score); ot_y.append(0)

            # --- positive: prompt echo ---
            pe = tp.parse(make_prompt_echo(qtext), nlp)
            pe_rf = relevance_features(pe, rubrics[qid], emb, nli)
            pr_s.append(prompt_read_flag(pe, qtext, pe_rf).score); pr_y.append(1)

            # --- positive: padded repetition ---
            rp = tp.parse(make_text_repetition(text, 2), nlp)
            rep_s.append(repetition_flag(rp, emb).score); rep_y.append(1)

            # --- positive: genuine answer, wrong question ---
            other = rng.choice([q for q in qids if q != qid])
            o_rf = relevance_features(p, rubrics[other], emb, nli)
            ot_s.append(off_topic_flag(o_rf).score); ot_y.append(1)

    print(f"\n{'flag':16s}{'n':>5s}{'AUC':>8s}{'best thr':>10s}{'TPR':>8s}{'FPR':>8s}"
          f"{'TPR@55':>9s}{'FPR@55':>9s}")
    print("-" * 92)
    flag_summary = {}
    for name, s, y in [("prompt_read", pr_s, pr_y), ("repetition", rep_s, rep_y),
                       ("off_topic", ot_s, ot_y)]:
        auc = roc_auc(s, y)
        thr, tpr, fpr, _ = best_threshold(s, y)
        sa, ya = np.asarray(s), np.asarray(y)
        t55 = float((sa[ya == 1] >= 55).mean()) if (ya == 1).any() else float("nan")
        f55 = float((sa[ya == 0] >= 55).mean()) if (ya == 0).any() else float("nan")
        flag_summary[name] = {"auc": auc, "best_threshold": thr, "tpr": tpr,
                              "fpr": fpr, "tpr_at_55": t55, "fpr_at_55": f55}
        print(f"{name:16s}{len(s):5d}{auc:8.3f}{thr:10.0f}{tpr:8.1%}{fpr:8.1%}"
              f"{t55:9.1%}{f55:9.1%}")

    out = REPORTS_DIR / "text_eval.json"
    out.write_text(json.dumps(
        {"ordinal": summary, "flags": flag_summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
