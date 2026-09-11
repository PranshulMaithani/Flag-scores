"""Why does a fully Tamil response score 29.7 when Swahili scores 99.5?

The dose-response run used only 2 samples per language per proportion, so the
first job is confirming the Tamil failure is real rather than a small-sample
artefact. The second is locating which channel breaks: the acoustic language ID,
the corroborating text pass, or the fusion between them.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/diag_tamil.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.asr.whisper_asr import WhisperASR
from voxscore.config import RAW_DIR, SAMPLE_RATE, device_report
from voxscore.flags.detectors import foreign_language_flag
from voxscore.pipeline import AUTO_TRANSCRIBE_THRESHOLD

LANGS = ["ta", "hi", "sw", "tl", "yo"]
N = 8
TARGET_S = 45.0


def load(name: str):
    d = RAW_DIR / "eval" / name
    return [np.load(f) for f in sorted(d.glob("*.npy"))] if d.exists() else []


def build(clips, rng, target_s=TARGET_S):
    want = int(target_s * SAMPLE_RATE)
    out = []
    while sum(len(c) for c in out) < want:
        out.append(clips[rng.randrange(len(clips))])
    return np.concatenate(out)[:want].astype(np.float32)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(device_report(), "\n", flush=True)
    rng = random.Random(5)
    asr = WhisperASR()

    print("=" * 100)
    print(f"100% NON-ENGLISH, n={N} per language - every one of these SHOULD fire")
    print("=" * 100)
    print(f"\n{'lang':6s}{'score':>8s}{'mean_p_non':>12s}{'max_run':>9s}"
          f"{'text_p_non':>12s}{'corrob':>8s}{'degen':>7s}  LID top guesses")
    print("-" * 100)

    summary = {}
    for lang in LANGS:
        clips = load(f"fleurs_{lang}")
        if not clips:
            print(f"{lang:6s}  no data")
            continue
        scores, rows = [], []
        for i in range(N):
            audio = build(clips, rng)
            res = asr.transcribe(audio, language="en")
            lw = asr.language_windows(audio)
            mean_non = float(np.mean([w.p_non_english for w in lw])) if lw else 0.0
            auto = None
            if mean_non > AUTO_TRANSCRIBE_THRESHOLD:
                auto = asr.transcribe_suspect_spans(audio, lw) or None
            fl = foreign_language_flag(lw, res.text, res.avg_logprob, transcript_auto=auto)
            scores.append(fl.score)
            guesses: dict[str, int] = {}
            for w in lw:
                if w.p_non_english > 0.5:
                    guesses[w.top_lang] = guesses.get(w.top_lang, 0) + 1
            rows.append((fl, guesses, auto))

        arr = np.array(scores)
        summary[lang] = arr
        f0, g0, a0 = rows[0][0], rows[0][1], rows[0][2]
        top = ",".join(f"{k}:{v}" for k, v in sorted(g0.items(), key=lambda x: -x[1])[:3])
        print(f"{lang:6s}{arr.mean():8.1f}{f0.features['mean_p_non_english']:12.3f}"
              f"{f0.features['max_run_non_english']:9.0f}"
              f"{f0.features['text_p_non_english']:12.3f}"
              f"{f0.features.get('corroboration', 0):8.2f}"
              f"{f0.features.get('auto_transcript_degenerate', 0):7.0f}  {top}", flush=True)
        print(f"      scores: {[round(s) for s in scores]}   "
              f"fires@30: {int((arr >= 30).sum())}/{N}", flush=True)
        if a0:
            print(f"      suspect-span transcript: {a0[:78]!r}", flush=True)
        print(flush=True)

    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    for lang, arr in summary.items():
        miss = int((arr < 30).sum())
        verdict = "OK" if miss == 0 else f"MISSES {miss}/{N} at threshold 30"
        print(f"  {lang:4s} mean {arr.mean():5.1f}  min {arr.min():5.1f}  "
              f"max {arr.max():5.1f}   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
