"""STEP 1 (run on YOUR laptop, where the wavs are).

Reads

    Audiostest/audios/<ciid>/<ciid>_<qid>.wav

and writes three things:

    voxscore_bundle.zip   anonymised .npy audio + the voxscore code   -> UPLOAD
    upload.csv            one row per audio, with a BLANK question    -> FILL IN, THEN UPLOAD
                          column for you to complete
    ciid_mapping.csv      anonymous id -> real ciid                   -> KEEP LOCAL

Only questions 25, 26 and 27 are packaged by default, since each candidate folder
holds the full 1-27 set and only the last three are being scored. Change with
--qids.

Usage
-----
    python make_bundle.py
    python make_bundle.py --qids 25,26,27
    python make_bundle.py --qids all

What to do with upload.csv
--------------------------
Open it, fill the `question_text` column with the question each audio was
answering, and upload it next to the zip. Relevance, off_topic and prompt_read
all need that text; grammar, lexical and fluency do not.

A note on "anonymised": this replaces the *identifier*. The audio is still a
recording of someone's voice, which is biometric. Safe to move between your own
machines; not de-identified in the sense a privacy review would mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
HERE = Path(__file__).resolve().parent
DEFAULT_QIDS = "25,26,27"


def find_voxscore() -> Path | None:
    """Locate the voxscore package so it can ride along in the zip."""
    for cand in (HERE / "voxscore", HERE.parent / "voxscore", Path.cwd() / "voxscore"):
        if (cand / "__init__.py").exists():
            return cand
    return None


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a wav to mono float32 at 16 kHz, without torch or ffmpeg."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        try:
            import soxr

            audio = soxr.resample(audio, sr, SAMPLE_RATE).astype(np.float32)
        except Exception:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return np.ascontiguousarray(audio), sr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="Audiostest/audios",
                    help="directory containing one folder per ciid")
    ap.add_argument("--out", default=".", help="where to write the zip and csvs")
    ap.add_argument("--qids", default=DEFAULT_QIDS,
                    help=f"comma-separated question ids to include, or 'all' "
                         f"(default {DEFAULT_QIDS})")
    ap.add_argument("--prefix", default="S", help="anonymous id prefix")
    ap.add_argument("--seed", type=int, default=20260911,
                    help="shuffle seed; anonymous ids do not follow ciid order")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.exists():
        print(f"ERROR: {in_dir} does not exist")
        return 1

    keep_all = args.qids.strip().lower() == "all"
    keep = set() if keep_all else {q.strip() for q in args.qids.split(",") if q.strip()}

    all_wavs = sorted(in_dir.glob("*/*.wav")) or sorted(in_dir.glob("*.wav"))
    if not all_wavs:
        print(f"ERROR: no .wav files under {in_dir}")
        print("       expected <input>/<ciid>/<ciid>_<qid>.wav")
        return 1

    def qid_of(w: Path) -> str:
        return w.stem.rsplit("_", 1)[1] if "_" in w.stem else "unknown"

    wavs = all_wavs if keep_all else [w for w in all_wavs if qid_of(w) in keep]
    skipped = len(all_wavs) - len(wavs)
    if not wavs:
        found = sorted({qid_of(w) for w in all_wavs})
        print(f"ERROR: no files matched --qids {args.qids}")
        print(f"       question ids present: {found}")
        return 1

    ciids = sorted({(w.parent.name if w.parent != in_dir else w.stem.split("_")[0])
                    for w in wavs})
    # Shuffle before numbering, so S0001 is not simply the alphabetically first
    # candidate. The mapping file is the only way back.
    shuffled = list(ciids)
    random.Random(args.seed).shuffle(shuffled)
    anon = {c: f"{args.prefix}{i + 1:04d}" for i, c in enumerate(shuffled)}

    staging = out_dir / "_bundle_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "npy").mkdir(parents=True)

    print(f"{len(all_wavs)} wav files found, {len(wavs)} match qids "
          f"{'all' if keep_all else sorted(keep)}"
          + (f" ({skipped} skipped)" if skipped else ""))
    print(f"{len(ciids)} candidates\n")

    rows, failures = [], []
    for w in wavs:
        ciid = w.parent.name if w.parent != in_dir else w.stem.split("_")[0]
        qid = qid_of(w)
        item_id = f"{anon[ciid]}_{qid}"

        try:
            audio, orig_sr = load_wav(w)
        except Exception as exc:
            failures.append((str(w), f"{type(exc).__name__}: {exc}"))
            continue

        np.save(staging / "npy" / f"{item_id}.npy", audio, allow_pickle=False)
        (staging / "npy" / f"{item_id}.json").write_text(json.dumps({
            "item_id": item_id, "question_id": qid,
            "orig_sr": SAMPLE_RATE, "source_sr": orig_sr,
            "duration_s": round(len(audio) / SAMPLE_RATE, 3),
        }), encoding="utf-8")
        rows.append({"anon_id": anon[ciid], "ciid": ciid, "question_id": qid,
                     "item_id": item_id, "source_file": str(w),
                     "duration_s": round(len(audio) / SAMPLE_RATE, 3)})

    # --- code ------------------------------------------------------------
    pkg = find_voxscore()
    if pkg is None:
        print("ERROR: could not find the voxscore package next to this script.")
        print("       Put make_bundle.py beside the voxscore/ directory.")
        return 1
    shutil.copytree(pkg, staging / "voxscore",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # --- zip -------------------------------------------------------------
    zip_path = out_dir / "voxscore_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(staging))
    shutil.rmtree(staging)

    # --- upload.csv: goes WITH the zip, question column left for you -----
    # Deliberately one row per audio rather than one per question id. If every
    # candidate answered the same question 25 you can fill one row and copy it
    # down; if the item shown varied by candidate, the per-audio row is the only
    # thing that can express that.
    upload_path = out_dir / "upload.csv"
    with upload_path.open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=["item_id", "anon_id", "question_id",
                                             "duration_s", "question_text"])
        wri.writeheader()
        for r in rows:
            wri.writerow({"item_id": r["item_id"], "anon_id": r["anon_id"],
                          "question_id": r["question_id"],
                          "duration_s": r["duration_s"], "question_text": ""})

    # --- mapping: deliberately NOT uploaded ------------------------------
    map_path = out_dir / "ciid_mapping.csv"
    with map_path.open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=["anon_id", "ciid", "question_id",
                                             "item_id", "source_file", "duration_s"])
        wri.writeheader()
        wri.writerows(rows)

    qids_done = sorted({r["question_id"] for r in rows})
    print(f"{'items packaged':24s}{len(rows)}")
    print(f"{'candidates':24s}{len(ciids)}")
    print(f"{'question ids':24s}{qids_done}")
    if failures:
        print(f"{'FAILED to read':24s}{len(failures)}")
        for f, why in failures[:5]:
            print(f"    {f}: {why}")

    print()
    print(f"  1. FILL IN      {upload_path}")
    print( "                  add the question text for each row "
           "(same question id = same text, so fill one and copy down)")
    print(f"  2. UPLOAD BOTH  {zip_path}   ({zip_path.stat().st_size / 1e9:.2f} GB)")
    print(f"                  {upload_path}")
    print(f"  3. KEEP LOCAL   {map_path}   (the only way back to real ciids)")
    print()
    print("Then on the scoring machine, with run_scoring.py and requirements.txt:")
    print("    pip install -r requirements.txt")
    print("    python run_scoring.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
