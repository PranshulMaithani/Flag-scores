"""End-to-end pipeline smoke test: audio in, XML out.

Uses concatenated LibriSpeech as a stand-in response. The content is nonsense as
an answer -- that is fine and in fact useful, since it should score badly on
relevance while every other code path still runs. The purpose is to prove the
wiring, measure runtime, and produce a real XML document to inspect.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/smoke_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import SAMPLE_RATE, device_report
from voxscore.pipeline import Pipeline
from voxscore.utils.audio_io import AudioItem
from voxscore.xml_out import write_json, write_xml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    print(device_report(), flush=True)

    files = sorted(Path("data/raw/smoke").glob("ls*.npy"))
    if not files:
        print("no smoke audio; run the loader first")
        return 1

    clips = [np.load(f) for f in files]
    # ~40 s, which crosses Whisper's 30 s boundary and exercises the long-form path.
    audio = np.concatenate(clips * 2)[: int(40 * SAMPLE_RATE)].astype(np.float32)

    qs = json.loads(Path("data/raw/questions_sample.json").read_text(encoding="utf-8"))
    bday = qs["birthday"]

    item = AudioItem(
        item_id="smoke001",
        audio=audio,
        sample_rate=SAMPLE_RATE,
        duration_s=len(audio) / SAMPLE_RATE,
        question_id="birthday",
        question_text=bday["text"],
        ideal_answers=bday["ideal_answers"],
    )

    print(f"\nitem: {item.duration_s:.1f}s audio", flush=True)
    pipe = Pipeline()

    t0 = time.perf_counter()
    result = pipe.score_item(item)
    elapsed = time.perf_counter() - t0

    print(f"\nscored in {elapsed:.1f}s ({item.duration_s / elapsed:.2f}x realtime)\n", flush=True)
    print(f"transcript: {result.transcript[:150]}...\n")

    print(f"{'category':12s}{'score':>8s}{'conf':>8s}   explanation")
    print("-" * 100)
    for name, cs in result.scores.items():
        print(f"{name:12s}{cs.score:8.1f}{cs.confidence:8.2f}   {result.explanations.get(name, '')[:62]}")

    print(f"\n{'flag':20s}{'score':>8s}{'fired':>8s}   evidence")
    print("-" * 100)
    for f in result.flags:
        print(f"{f.name:20s}{f.score:8.1f}{str(f.fired):>8s}   {f.evidence[:58]}")

    print(f"\nquality: scorable={bool(result.quality['scorable'])} "
          f"conf={result.quality['quality_confidence']:.2f} "
          f"snr={result.quality['snr_db']:.1f}dB "
          f"words={int(result.quality['word_count'])}")
    for w in result.quality_warnings:
        print(f"  warning: {w}")

    n_feats = sum(len(v) for v in result.features.values())
    print(f"\nfeature blocks: " + ", ".join(f"{k}={len(v)}" for k, v in result.features.items())
          + f"  (total {n_feats})")

    xml_path = write_xml([result], "reports/smoke_results.xml",
                         {"source": "smoke_pipeline", "elapsed_s": f"{elapsed:.1f}"})
    json_path = write_json([result], "reports/smoke_results.json",
                           {"source": "smoke_pipeline"})
    print(f"\nwrote {xml_path} ({xml_path.stat().st_size:,} bytes)")
    print(f"wrote {json_path} ({json_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
