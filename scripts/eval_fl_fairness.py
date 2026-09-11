"""Re-measure the foreign-language flag after the corroboration change.

Focused rerun: the fairness groups at higher n, plus enough dose points to
confirm that suppressing accent false-positives has not blunted real detection.
Both must hold — a flag that never fires is fair and useless.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_fl_fairness.py
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
from voxscore.eval.synth import make_foreign_mix
from voxscore.flags.detectors import foreign_language_flag
from voxscore.pipeline import AUTO_TRANSCRIBE_THRESHOLD

THRESHOLDS = (15, 25, 35, 45, 55)


def load(name: str, limit: int = 120):
    d = RAW_DIR / "eval" / name
    return [np.load(f) for f in sorted(d.glob("*.npy"))[:limit]] if d.exists() else []


def build(clips, rng, target_s=45.0):
    if not clips:
        return None
    want = int(target_s * SAMPLE_RATE)
    out, guard = [], 0
    while sum(len(c) for c in out) < want and guard < 40:
        out.append(clips[rng.randrange(len(clips))])
        guard += 1
    return np.concatenate(out)[:want].astype(np.float32)


def score(asr, audio):
    res = asr.transcribe(audio, language="en")
    lw = asr.language_windows(audio)
    auto = None
    if lw and float(np.mean([w.p_non_english for w in lw])) > AUTO_TRANSCRIBE_THRESHOLD:
        auto = asr.transcribe_suspect_spans(audio, lw) or None
    f = foreign_language_flag(lw, res.text, res.avg_logprob, transcript_auto=auto)
    return f.score, f.features


def main() -> int:
    print(device_report(), "\n", flush=True)
    rng = random.Random(23)
    asr = WhisperASR()
    out: dict = {"fairness": [], "dose": []}

    print("=" * 74)
    print("FAIRNESS - all of these are genuinely English")
    print("=" * 74)
    groups = [("US English (FLEURS)", load("fleurs_en")),
              ("Indian English (Svarah)", load("svarah_en_in"))]

    summary = {}
    for name, clips in groups:
        if not clips:
            print(f"\n  {name}: NO DATA")
            continue
        scores = []
        for _ in range(20):
            a = build(clips, rng)
            if a is None:
                continue
            s, feats = score(asr, a)
            scores.append(s)
            out["fairness"].append({"group": name, "score": s,
                                    "corroboration": feats.get("corroboration"),
                                    "mean_p_non_en": feats.get("mean_p_non_english")})
        arr = np.array(scores)
        summary[name] = {t: float((arr >= t).mean()) for t in THRESHOLDS}
        print(f"\n  {name}  (n={len(arr)})")
        print(f"    mean {arr.mean():5.1f} | median {np.median(arr):5.1f} | "
              f"p90 {np.percentile(arr, 90):5.1f} | max {arr.max():5.1f}")
        print("    false-positive rate: " + "  ".join(
            f"@{t}={float((arr >= t).mean()):.1%}" for t in THRESHOLDS), flush=True)

    print("\n" + "=" * 74)
    print("DOSE - detection must survive the fairness fix")
    print("=" * 74)
    en = load("fleurs_en")
    langs = {l: load(f"fleurs_{l}") for l in ("hi", "ta", "sw")}
    langs = {k: v for k, v in langs.items() if v}
    print(f"\n{'proportion':>11s}" + "".join(f"{l:>8s}" for l in langs) + f"{'mean':>8s}")
    print("-" * (11 + 8 * (len(langs) + 1)))
    for prop in (0.0, 0.10, 0.20, 0.35, 0.50, 1.0):
        row = []
        for lang, clips in langs.items():
            ss = []
            for _ in range(2):
                base, fgn = build(en, rng), build(clips, rng)
                if base is None or fgn is None:
                    continue
                item = make_foreign_mix(base, fgn, prop, lang, rng)
                s, _ = score(asr, item.audio)
                ss.append(s)
                out["dose"].append({"lang": lang, "proportion": prop, "score": s})
            row.append(float(np.mean(ss)) if ss else float("nan"))
        print(f"{prop:>10.0%} " + "".join(f"{v:>8.1f}" for v in row)
              + f"{np.nanmean(row):>8.1f}", flush=True)

    p = REPORTS_DIR / "fl_fairness_eval.json"
    p.write_text(json.dumps({"results": out, "fpr": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
