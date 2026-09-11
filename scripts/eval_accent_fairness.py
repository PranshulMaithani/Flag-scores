"""Accent fairness audit, with the spoken content held constant.

Every speaker in the Speech Accent Archive reads the *same* elicitation
paragraph. That removes the confound that makes accent-fairness claims hard to
support: any difference between groups cannot be "they said different things",
because they said identical things.

Two measurements, both of which must come out flat:

1. **foreign_language flag score by accent group.** Everyone is speaking English.
   Any group scoring higher than the native control is being penalised for its
   accent, which is the failure this project has been chasing since day one.
2. **ASR word-error rate by accent group.** Everything downstream inherits ASR
   error, so an accent that transcribes worse is disadvantaged on *every*
   category, not just this flag.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_accent_fairness.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.asr.whisper_asr import WhisperASR
from voxscore.config import RAW_DIR, REPORTS_DIR, device_report
from voxscore.flags.detectors import foreign_language_flag
from voxscore.pipeline import AUTO_TRANSCRIBE_THRESHOLD

# The standard Speech Accent Archive elicitation paragraph, read by every speaker.
REFERENCE = (
    "please call stella ask her to bring these things with her from the store "
    "six spoons of fresh snow peas five thick slabs of blue cheese and maybe a "
    "snack for her brother bob we also need a small plastic snake and a big toy "
    "frog for the kids she can scoop these things into three red bags and we "
    "will go meet her wednesday at the train station"
)

GROUPS = ["native_en", "indian", "filipino", "african"]
THRESHOLDS = (15, 25, 30, 45)


def norm(t: str) -> str:
    t = re.sub(r"[^a-z0-9' ]", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    print(device_report(), "\n", flush=True)
    asr = WhisperASR()
    import jiwer

    results: dict[str, dict] = {}

    print("=" * 84)
    print("ACCENT FAIRNESS - same paragraph, different first languages, all English")
    print("=" * 84)

    for grp in GROUPS:
        d = RAW_DIR / "eval" / f"saa_{grp}"
        files = sorted(d.glob("*.npy"))
        if not files:
            print(f"\n  {grp}: NO DATA")
            continue

        scores, wers, hyps = [], [], []
        for f in files:
            audio = np.load(f)
            res = asr.transcribe(audio, language="en")
            lw = asr.language_windows(audio)
            auto = None
            if lw and float(np.mean([w.p_non_english for w in lw])) > AUTO_TRANSCRIBE_THRESHOLD:
                auto = asr.transcribe_suspect_spans(audio, lw) or None
            flag = foreign_language_flag(lw, res.text, res.avg_logprob, transcript_auto=auto)
            scores.append(flag.score)

            hyp = norm(res.text)
            if hyp:
                hyps.append(hyp)
                wers.append(float(jiwer.wer(REFERENCE, hyp)))

        s = np.array(scores)
        w = np.array(wers) if wers else np.array([np.nan])
        results[grp] = {
            "n": len(s),
            "flag_mean": float(s.mean()), "flag_p90": float(np.percentile(s, 90)),
            "flag_max": float(s.max()),
            "fpr": {t: float((s >= t).mean()) for t in THRESHOLDS},
            "wer_mean": float(np.nanmean(w)), "wer_median": float(np.nanmedian(w)),
        }
        print(f"\n  {grp:12s} n={len(s)}")
        print(f"    foreign_language  mean {s.mean():5.1f}  p90 {np.percentile(s, 90):5.1f}  "
              f"max {s.max():5.1f}")
        print(f"    false positives   " + "  ".join(
            f"@{t}={float((s >= t).mean()):.1%}" for t in THRESHOLDS), flush=True)
        print(f"    ASR WER           mean {np.nanmean(w):5.1%}  median {np.nanmedian(w):5.1%}")

    if len(results) > 1:
        print("\n" + "=" * 84)
        print("GAP vs the native-English control")
        print("=" * 84)
        base = results.get("native_en")
        if base:
            print(f"\n{'group':14s}{'flag mean':>12s}{'gap':>9s}{'FPR@30':>10s}"
                  f"{'WER':>9s}{'WER gap':>10s}")
            print("-" * 64)
            for grp in GROUPS:
                r = results.get(grp)
                if not r:
                    continue
                print(f"{grp:14s}{r['flag_mean']:12.1f}"
                      f"{r['flag_mean'] - base['flag_mean']:+9.1f}"
                      f"{r['fpr'][30]:10.1%}{r['wer_mean']:9.1%}"
                      f"{r['wer_mean'] - base['wer_mean']:+10.1%}")
        print("\nEvery group is speaking English. A positive flag gap means that accent "
              "is being penalised;\na positive WER gap means it is transcribed worse, "
              "which disadvantages it on every category.")

    out = REPORTS_DIR / "accent_fairness.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
