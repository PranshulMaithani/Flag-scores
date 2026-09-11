"""Fetch accent-controlled English for the fairness audit.

**Speech Accent Archive** is the right instrument for this. Every speaker reads
*the same paragraph*, so the text is held constant and the speaker's first
language is the only thing varying. Any score difference between accent groups is
therefore attributable to accent and not to what was said — which is exactly the
confound that makes accent-fairness claims hard to support otherwise.

Downloads only the `original` split (1.1 GB of 24 GB; the rest is neural-codec
variants of the same audio, which we have no use for) and pulls the parquet files
directly rather than through `load_dataset`, which fetches every split.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/fetch_accent_corpora.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import RAW_DIR, SAMPLE_RATE
from voxscore.data.hf_audio import decode_clip

# Two halves of the same archive: the "_english" repo holds native-English
# speakers only (the control group), "_other" holds every other first language.
# Shards in "_other" are ordered alphabetically by L1, so Hindi, Tamil, Tagalog
# and Yoruba are spread across the later ones and all nine are needed.
REPO_NATIVE = "CodecSR/speech_accent_archive_english"
REPO_OTHER = "CodecSR/speech_accent_archive_other"

# Grouped by the population the client actually assesses. "english" is the
# native-speaker control; everything else is an L2 group that must not be
# penalised for its accent.
GROUPS = {
    "native_en": ["english"],
    "indian": ["hindi", "tamil", "telugu", "malayalam", "kannada", "bengali",
               "gujarati", "marathi", "punjabi", "urdu", "nepali", "sinhala"],
    "filipino": ["tagalog", "cebuano", "ilocano", "bikol", "hiligaynon", "filipino"],
    "african": ["yoruba", "igbo", "hausa", "swahili", "amharic", "twi", "wolof",
                "shona", "zulu", "xhosa", "somali", "tigrinya", "ewe", "ganda",
                "krio", "luganda", "kikuyu", "oromo", "akan"],
}
MAX_PER_GROUP = 45


def main() -> int:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        env = Path(".env")
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("HF_TOKEN="):
                    tok = line.split("=", 1)[1].strip()
    if not tok:
        print("no HF_TOKEN found (.env or environment)")
        return 1

    want = {a: g for g, accents in GROUPS.items() for a in accents}
    saved: Counter = Counter()
    meta: dict[str, dict] = {}
    seen_accents: Counter = Counter()

    shards = [(REPO_NATIVE, f"data/original-{i:05d}-of-00003.parquet") for i in range(3)]
    shards += [(REPO_OTHER, f"data/original-{i:05d}-of-00009.parquet") for i in range(9)]

    for repo, fn in shards:
        if all(saved[g] >= MAX_PER_GROUP for g in GROUPS):
            print("all groups full; stopping early")
            break
        print(f"downloading {repo.split('/')[-1]}/{fn.split('/')[-1]} ...", flush=True)
        path = hf_hub_download(repo, fn, repo_type="dataset", token=tok)

        tbl = pq.read_table(path, columns=["id", "accent", "audio"])
        accents = tbl.column("accent").to_pylist()
        seen_accents.update(accents)
        ids = tbl.column("id").to_pylist()
        audio_col = tbl.column("audio")

        for i, acc in enumerate(accents):
            grp = want.get(str(acc).lower())
            if grp is None or saved[grp] >= MAX_PER_GROUP:
                continue
            try:
                a, _ = decode_clip(audio_col[i].as_py())
            except Exception:
                continue
            if len(a) < SAMPLE_RATE * 2:
                continue
            out = RAW_DIR / "eval" / f"saa_{grp}"
            out.mkdir(parents=True, exist_ok=True)
            key = f"{saved[grp]:04d}"
            np.save(out / f"{key}.npy", a.astype(np.float32))
            meta.setdefault(grp, {})[key] = {"id": ids[i], "accent": acc}
            saved[grp] += 1

        print("  " + ", ".join(f"{g}={saved[g]}" for g in GROUPS), flush=True)

    for grp, m in meta.items():
        (RAW_DIR / "eval" / f"saa_{grp}" / "speakers.json").write_text(
            json.dumps(m, indent=1), encoding="utf-8")

    print("\nsaved per group:")
    for g in GROUPS:
        print(f"  {g:12s}{saved[g]:4d}")

    print("\nL1s present in the corpus that we did not map (top 20):")
    unmapped = [(a, n) for a, n in seen_accents.most_common() if a.lower() not in want]
    for a, n in unmapped[:20]:
        print(f"  {a:18s}{n:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
