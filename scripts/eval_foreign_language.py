"""Evaluate the foreign-language flag, including its fairness on accented English.

Two experiments:

**1. Dose-response.** Splice controlled proportions of non-English speech into
English responses and plot flag score against the true proportion. This is worth
more than a positive/negative set: it tells the client what a threshold *means*
in terms of how much foreign speech triggers it, which is exactly the decision
they said they wanted to keep.

**2. Fairness.** Measure the false-positive rate on **Indian-accented English**
(Svarah) against US English (FLEURS en_us). This is the experiment that decides
whether the flag is safe to deploy. Language-ID models systematically misread
accented English as the speaker's L1, and the client's population is entirely L2
speakers -- a threshold tuned on native English would fire on exactly the
candidates it must never fire on.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_foreign_language.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.asr.whisper_asr import WhisperASR
from voxscore.config import RAW_DIR, REPORTS_DIR, SAMPLE_RATE, device_report
from voxscore.eval.synth import make_code_switch, make_foreign_mix
from voxscore.flags.detectors import foreign_language_flag
from voxscore.pipeline import AUTO_TRANSCRIBE_THRESHOLD

PROPORTIONS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
FOREIGN_LANGS = ["hi", "ta", "tl", "sw", "yo"]
TARGET_S = 45.0


def load_clips(name: str, limit: int = 100) -> list[np.ndarray]:
    d = RAW_DIR / "eval" / name
    if not d.exists():
        return []
    return [np.load(f) for f in sorted(d.glob("*.npy"))[:limit]]


def build_response(clips: list[np.ndarray], rng: random.Random, target_s: float = TARGET_S):
    """Concatenate short corpus clips into a response of realistic length."""
    if not clips:
        return None
    want = int(target_s * SAMPLE_RATE)
    out, guard = [], 0
    while sum(len(c) for c in out) < want and guard < 40:
        out.append(clips[rng.randrange(len(clips))])
        guard += 1
    return np.concatenate(out)[:want].astype(np.float32) if out else None


def score_item(asr: WhisperASR, audio: np.ndarray) -> tuple[float, dict]:
    """Score one item exactly as the pipeline would, corroborating pass included."""
    res = asr.transcribe(audio, language="en")
    lw = asr.language_windows(audio)
    auto = None
    if lw:
        mean_non_en = float(np.mean([w.p_non_english for w in lw]))
        if mean_non_en > AUTO_TRANSCRIBE_THRESHOLD:
            auto = asr.transcribe_suspect_spans(audio, lw) or None
    flag = foreign_language_flag(lw, res.text, res.avg_logprob, transcript_auto=auto)
    return flag.score, flag.features


def main() -> int:
    print(device_report(), "\n", flush=True)
    rng = random.Random(11)
    asr = WhisperASR()

    en_us = load_clips("fleurs_en")
    en_in = load_clips("svarah_en_in")
    foreign = {l: load_clips(f"fleurs_{l}") for l in FOREIGN_LANGS}
    foreign = {k: v for k, v in foreign.items() if v}

    print(f"corpora: en_us={len(en_us)} svarah_en_in={len(en_in)} "
          + " ".join(f"{k}={len(v)}" for k, v in foreign.items()), flush=True)
    if not en_us or not foreign:
        print("\nMissing corpora. Run scripts/fetch_eval_corpora.py first.")
        return 1

    results = {"dose_response": [], "fairness": [], "code_switch": []}
    t0 = time.perf_counter()

    # ---- 1. dose-response -------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT 1 - dose-response: flag score vs true proportion of non-English")
    print("=" * 78)
    print(f"\n{'proportion':>11s}" + "".join(f"{l:>9s}" for l in foreign) + f"{'mean':>9s}")
    print("-" * (11 + 9 * (len(foreign) + 1)))

    for prop in PROPORTIONS:
        row, per_lang = [], {}
        for lang, clips in foreign.items():
            scores = []
            for _ in range(3):
                base = build_response(en_us, rng)
                fgn = build_response(clips, rng, target_s=TARGET_S)
                if base is None or fgn is None:
                    continue
                item = make_foreign_mix(base, fgn, prop, lang, rng)
                s, _ = score_item(asr, item.audio)
                scores.append(s)
                results["dose_response"].append(
                    {"lang": lang, "proportion": prop, "score": s, "detail": item.detail}
                )
            m = float(np.mean(scores)) if scores else float("nan")
            per_lang[lang] = m
            row.append(m)
        print(f"{prop:>10.0%} " + "".join(f"{v:>9.1f}" for v in row)
              + f"{np.nanmean(row):>9.1f}", flush=True)

    # ---- 2. fairness ------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 - FAIRNESS: false positives on accented English (all truly English)")
    print("=" * 78)

    for name, clips in [("US English (FLEURS en_us)", en_us),
                        ("Indian English (Svarah)", en_in)]:
        if not clips:
            print(f"\n  {name}: NO DATA - fairness check incomplete")
            continue
        scores = []
        for _ in range(12):
            base = build_response(clips, rng)
            if base is None:
                continue
            s, feats = score_item(asr, base)
            scores.append(s)
            results["fairness"].append({"group": name, "score": s,
                                        "mean_p_non_en": feats.get("mean_p_non_english", 0.0)})
        if scores:
            a = np.array(scores)
            print(f"\n  {name}  (n={len(a)})")
            print(f"    mean {a.mean():6.1f} | median {np.median(a):6.1f} | "
                  f"p90 {np.percentile(a, 90):6.1f} | max {a.max():6.1f}")
            for t in (15, 25, 40, 55):
                print(f"    false-positive rate at threshold {t}: {(a >= t).mean():6.1%}")

    # ---- 3. code-switching ------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT 3 - code-switching: a few foreign words, not a language change")
    print("=" * 78)
    print(f"\n{'switches':>9s}{'approx share':>14s}{'mean score':>12s}")
    print("-" * 35)
    for n_sw in (1, 2, 4, 6):
        scores, shares = [], []
        for lang, clips in list(foreign.items())[:3]:
            base = build_response(en_us, rng)
            fgn = build_response(clips, rng, target_s=10.0)
            if base is None or fgn is None:
                continue
            item = make_code_switch(base, fgn, n_sw, lang=lang, rng=rng)
            s, _ = score_item(asr, item.audio)
            scores.append(s)
            shares.append(item.label)
            results["code_switch"].append({"lang": lang, "n_switches": n_sw,
                                           "share": item.label, "score": s})
        if scores:
            print(f"{n_sw:>9d}{np.mean(shares):>13.1%}{np.mean(scores):>12.1f}", flush=True)

    out = REPORTS_DIR / "foreign_language_eval.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nelapsed {time.perf_counter() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
