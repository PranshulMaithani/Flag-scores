"""Download the public corpora the flag evaluation needs.

All evaluation-only: nothing here influences a shipped parameter.

* **FLEURS** (CC-BY-4.0, ungated) -- non-English speech for the foreign-language
  splicing experiment. Languages chosen to match the client's candidate
  population: Hindi, Tamil, Tagalog, Swahili, Yoruba.
* **Svarah** (Indian-accented English) -- the *negatives* that matter. Language-ID
  models misclassify accented English as the speaker's L1, and the client's
  population is entirely L2 speakers, so a foreign-language threshold tuned only
  against US English would false-positive on exactly the people it must not.

  The canonical `ai4bharat/Svarah` is gated. This pulls an ungated mirror, which
  is fine for measurement but is flagged in the licence audit: provenance should
  be confirmed and the official terms accepted before any of this is cited
  externally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import RAW_DIR
from voxscore.data.hf_audio import decode_clip, undecoded

# (dataset, config, split, n_clips, output subdir)
TARGETS = [
    ("google/fleurs", "en_us", "test", 60, "fleurs_en"),
    ("google/fleurs", "hi_in", "test", 60, "fleurs_hi"),
    ("google/fleurs", "ta_in", "test", 40, "fleurs_ta"),
    ("google/fleurs", "fil_ph", "test", 40, "fleurs_tl"),
    ("google/fleurs", "sw_ke", "test", 40, "fleurs_sw"),
    ("google/fleurs", "yo_ng", "test", 40, "fleurs_yo"),
    ("Bhargav0044/svarah1", None, "train", 80, "svarah_en_in"),
]


def fetch(dataset: str, config: str | None, split: str, n: int, out_name: str) -> int:
    from datasets import load_dataset

    out_dir = RAW_DIR / "eval" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.npy"))
    if len(existing) >= n:
        print(f"  {out_name}: already have {len(existing)} clips, skipping", flush=True)
        return len(existing)

    try:
        ds = load_dataset(dataset, config, split=f"{split}[:{n * 2}]")
    except Exception as exc:
        print(f"  {out_name}: FAILED {type(exc).__name__}: {str(exc)[:110]}", flush=True)
        return 0

    ds = undecoded(ds)
    audio_col = "audio" if "audio" in ds.column_names else "path"
    ref_col = next(
        (c for c in ("transcription", "raw_transcription", "text", "sentence")
         if c in ds.column_names),
        None,
    )

    saved = 0
    refs = {}
    for i in range(min(len(ds), n * 2)):
        if saved >= n:
            break
        try:
            audio, _ = decode_clip(ds[i][audio_col])
        except Exception:
            continue
        if len(audio) < 16000:
            continue
        np.save(out_dir / f"{saved:04d}.npy", audio.astype(np.float32))
        if ref_col:
            refs[f"{saved:04d}"] = str(ds[i][ref_col])
        saved += 1

    if refs:
        import json
        (out_dir / "refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"  {out_name}: saved {saved} clips", flush=True)
    return saved


def main() -> int:
    print("fetching evaluation corpora (evaluation only, never used for fitting)\n")
    total = 0
    for dataset, config, split, n, name in TARGETS:
        print(f"{dataset} [{config or '-'}] -> {name}", flush=True)
        total += fetch(dataset, config, split, n, name)
    print(f"\ntotal clips: {total}")
    print(f"under {RAW_DIR / 'eval'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
