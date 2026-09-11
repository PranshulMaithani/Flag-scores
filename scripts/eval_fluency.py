"""Validate the fluency score by controlled degradation of real speech.

Fluency is the one category with no validation. This manufactures a known
severity variable instead of hunting for proficiency-labelled audio, exactly as
the foreign-language dose-response did.

Three experiments, each testing a specific claim the scorecard makes:

  A. pause burden   -> score must fall monotonically
  B. pause placement -> mid-clause must score below clause-boundary at equal count
  C. speaking rate  -> plateau, so both extremes fall below natural pace

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_fluency.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.asr.align import align_words
from voxscore.asr.whisper_asr import WhisperASR
from voxscore.config import RAW_DIR, REPORTS_DIR, SAMPLE_RATE, device_report
from voxscore.eval.fluency_synth import (
    change_rate,
    clause_boundary_word_indices,
    inject_pauses,
)
from voxscore.features.fluency import fluency_features
from voxscore.scoring.aggregate import score_category
from voxscore.utils import textproc as tp

N_SOURCES = 10
PAUSE_DOSES = [(0, 0.0), (2, 0.6), (4, 0.6), (8, 0.6), (4, 1.2), (8, 1.2)]
RATES = [0.60, 0.75, 1.00, 1.25, 1.50]


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def score_fluency(asr, nlp, audio) -> tuple[float, dict]:
    res = asr.transcribe(audio, language="en")
    if not res.text.strip():
        return float("nan"), {}
    p = tp.parse(res.text, nlp)
    words = align_words(audio, res.text)
    f = fluency_features(words, p, len(audio) / SAMPLE_RATE)
    return score_category("fluency", f).score, f


def main() -> int:
    print(device_report(), "\n", flush=True)
    rng = random.Random(17)
    asr = WhisperASR()
    nlp = tp.get_nlp()

    # Fluent native readers, so the starting point is genuinely fluent speech.
    srcs = sorted((RAW_DIR / "eval" / "saa_native_en").glob("*.npy"))[:N_SOURCES]
    if not srcs:
        print("no source audio; run scripts/fetch_accent_corpora.py first")
        return 1

    prepared = []
    for f in srcs:
        a = np.load(f)
        res = asr.transcribe(a, language="en")
        if not res.text.strip():
            continue
        p = tp.parse(res.text, nlp)
        w = align_words(a, res.text)
        if len(w) < 20:
            continue
        prepared.append((a, p, w, clause_boundary_word_indices(p, len(w))))
    print(f"prepared {len(prepared)} fluent source recordings\n", flush=True)

    out: dict = {}

    # ---- A. pause burden -------------------------------------------------
    print("=" * 78)
    print("A. PAUSE BURDEN - score must fall as pauses are added")
    print("=" * 78)
    print(f"\n{'injected':>22s}{'added s':>10s}{'fluency':>10s}{'MLR':>8s}"
          f"{'pause/min':>11s}{'phon':>8s}")
    print("-" * 70)
    sev_all, sc_all = [], []
    for n_p, dur in PAUSE_DOSES:
        scores, mlr, rate, phon = [], [], [], []
        for a, p, w, cb in prepared:
            item = inject_pauses(a, w, n_p, dur, "random", cb, rng)
            s, f = score_fluency(asr, nlp, item.audio)
            if np.isnan(s):
                continue
            scores.append(s); mlr.append(f.get("mean_length_of_run", 0))
            rate.append(f.get("silent_pause_rate", 0)); phon.append(f.get("phonation_time_ratio", 0))
            sev_all.append(n_p * dur); sc_all.append(s)
        label = "clean" if n_p == 0 else f"{n_p} x {dur:.1f}s"
        print(f"{label:>22s}{n_p * dur:>10.1f}{np.mean(scores):>10.1f}"
              f"{np.mean(mlr):>8.1f}{np.mean(rate):>11.1f}{np.mean(phon):>8.2f}", flush=True)
    rho_a = spearman(sc_all, [-s for s in sev_all])
    out["pause_burden"] = {"spearman_vs_negative_severity": rho_a}
    print(f"\n  Spearman(score, -severity) = {rho_a:+.3f}   "
          f"{'PASS' if rho_a > 0.6 else 'FAIL'}  (want strongly positive)")

    # ---- B. pause placement ---------------------------------------------
    print("\n" + "=" * 78)
    print("B. PAUSE PLACEMENT - same count, clause boundary vs mid-clause")
    print("=" * 78)
    print(f"\n{'n pauses':>10s}{'at clause':>12s}{'mid-clause':>13s}{'gap':>9s}"
          f"{'within-clause ratio':>22s}")
    print("-" * 68)
    gaps = []
    for n_p in (4, 8):
        cl_s, mid_s, cl_r, mid_r = [], [], [], []
        for a, p, w, cb in prepared:
            if len(cb) < n_p:
                continue
            ic = inject_pauses(a, w, n_p, 0.6, "clause", cb, rng)
            im = inject_pauses(a, w, n_p, 0.6, "mid", cb, rng)
            s1, f1 = score_fluency(asr, nlp, ic.audio)
            s2, f2 = score_fluency(asr, nlp, im.audio)
            if np.isnan(s1) or np.isnan(s2):
                continue
            cl_s.append(s1); mid_s.append(s2)
            cl_r.append(f1.get("within_clause_pause_ratio", 0))
            mid_r.append(f2.get("within_clause_pause_ratio", 0))
        if not cl_s:
            continue
        gap = np.mean(cl_s) - np.mean(mid_s)
        gaps.append(gap)
        print(f"{n_p:>10d}{np.mean(cl_s):>12.1f}{np.mean(mid_s):>13.1f}{gap:>+9.1f}"
              f"{np.mean(cl_r):>11.2f}{np.mean(mid_r):>11.2f}", flush=True)
    out["placement_gap"] = [float(g) for g in gaps]
    mean_gap = float(np.mean(gaps)) if gaps else float("nan")
    print(f"\n  mean gap = {mean_gap:+.1f}   "
          f"{'PASS' if mean_gap > 1.0 else 'INCONCLUSIVE'}  "
          f"(clause-boundary pausing should score higher)")

    # ---- C. speaking rate ------------------------------------------------
    print("\n" + "=" * 78)
    print("C. SPEAKING RATE - plateau, so both extremes should fall")
    print("=" * 78)
    print(f"\n{'stretch':>10s}{'fluency':>10s}{'wpm':>9s}{'artic rate':>12s}")
    print("-" * 42)
    rate_scores = {}
    for r in RATES:
        scores, wpm, ar = [], [], []
        for a, p, w, cb in prepared:
            item = change_rate(a, r)
            s, f = score_fluency(asr, nlp, item.audio)
            if np.isnan(s):
                continue
            scores.append(s); wpm.append(f.get("speech_rate_wpm", 0))
            ar.append(f.get("articulation_rate_wps", 0))
        rate_scores[r] = float(np.mean(scores))
        print(f"{r:>10.2f}{np.mean(scores):>10.1f}{np.mean(wpm):>9.0f}"
              f"{np.mean(ar):>12.2f}", flush=True)
    out["rate"] = rate_scores
    nat = rate_scores.get(1.00, 0)
    slowest, fastest = rate_scores.get(0.60, 0), rate_scores.get(1.50, 0)
    plateau = nat >= slowest and nat >= fastest
    print(f"\n  natural {nat:.1f} vs slowest {slowest:.1f} and fastest {fastest:.1f}   "
          f"{'PASS' if plateau else 'FAIL'}  (natural should not be beaten by either extreme)")

    p = REPORTS_DIR / "fluency_validation.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
