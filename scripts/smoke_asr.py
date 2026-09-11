"""End-to-end smoke test: ASR -> forced alignment -> pauses -> windowed LID.

Run:  .venv/Scripts/python.exe scripts/smoke_asr.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import PipelineConfig, device_report
from voxscore.asr.align import align_words, extract_pauses, phonation_time
from voxscore.asr.whisper_asr import WhisperASR


def main() -> int:
    print(device_report())
    print()

    files = sorted(Path("data/raw/smoke").glob("*.npy"))
    if not files:
        print("no smoke audio found; run the loader first")
        return 1

    cfg = PipelineConfig()
    asr = WhisperASR(cfg)

    for f in files[:3]:
        audio = np.load(f)
        dur = len(audio) / 16000
        print("=" * 72)
        print(f"{f.name}  ({dur:.1f}s)")

        t0 = time.perf_counter()
        res = asr.transcribe(audio)
        t_asr = time.perf_counter() - t0

        print(f"  ASR {t_asr:.1f}s ({dur / max(t_asr, 1e-9):.1f}x realtime)")
        print(f"  text: {res.text[:150]}")
        print(f"  avg_logprob: {res.avg_logprob:.3f}  words: {res.word_count}")

        t0 = time.perf_counter()
        words = align_words(audio, res.text)
        t_align = time.perf_counter() - t0
        print(f"  alignment {t_align:.1f}s -> {len(words)} words")

        if words:
            print("  first 8:", ", ".join(f"{w.text}[{w.start:.2f}-{w.end:.2f}]" for w in words[:8]))
            pauses = extract_pauses(words, dur, cfg.short_pause_s)
            pt = phonation_time(words)
            print(f"  pauses>={cfg.short_pause_s}s: {len(pauses)}  "
                  f"phonation {pt:.1f}s / {dur:.1f}s = {pt / dur:.2f}")
            if pauses:
                longest = max(pauses, key=lambda p: p.duration)
                print(f"  longest pause: {longest.duration:.2f}s at {longest.start:.2f}s")
            # Sanity: alignment must not run past the audio.
            assert words[-1].end <= dur + 0.1, f"alignment overran audio: {words[-1].end} > {dur}"

        t0 = time.perf_counter()
        lw = asr.language_windows(audio)
        t_lid = time.perf_counter() - t0
        if lw:
            pen = np.mean([w.p_english for w in lw])
            print(f"  LID {t_lid:.1f}s -> {len(lw)} windows, mean p(en)={pen:.3f}, "
                  f"top={lw[0].top_lang}({lw[0].top_prob:.2f})")
        print()

    print("smoke test complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
