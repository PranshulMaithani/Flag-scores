"""STEP 1 (run on YOUR laptop, where the wavs are).

Reads

    Audiostest/audios/<ciid>/<ciid>_<qid>.wav

and writes three things:

    voxscore_bundle.zip   anonymised .npy audio                       -> UPLOAD
    upload.csv            one row per audio, with a BLANK question    -> FILL IN, THEN UPLOAD
                          column for you to complete
    ciid_mapping.csv      anonymous id -> real ciid                   -> KEEP LOCAL

DEPENDENCIES: none required.

    This runs on the standard library alone -- `wave` reads the audio, `audioop`
    resamples it, and the .npy files are written by hand, since the format is
    just a short header followed by raw samples. numpy and soundfile are *used
    if present* and simply make it faster; neither is needed. That matters on a
    locked-down machine where installing packages is not an option.

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
import struct
import sys
import wave
import zipfile
from array import array
from pathlib import Path

SAMPLE_RATE = 16_000
HERE = Path(__file__).resolve().parent
DEFAULT_QIDS = "25,26,27"

# Optional accelerators. Everything works without them.
try:
    import numpy as _np
except Exception:
    _np = None

try:
    import audioop as _audioop           # removed in Python 3.13
except Exception:
    _audioop = None


def find_voxscore() -> Path | None:
    """Include the package in the zip if it happens to be here. Optional."""
    for cand in (HERE / "voxscore", HERE.parent / "voxscore", Path.cwd() / "voxscore"):
        if (cand / "__init__.py").exists():
            return cand
    return None


# --------------------------------------------------------------------------- #
# WAV reading, standard library only
# --------------------------------------------------------------------------- #

def read_wav(path: Path) -> tuple[object, int, int]:
    """Read a PCM wav to mono float32 in [-1, 1].

    Returns ``(samples, original_rate, out_rate)``. ``samples`` is a numpy array
    when numpy is available and an ``array('f')`` otherwise; both write out
    identically.

    Only PCM is supported, which is what `wave` handles. Compressed wavs (rare
    for assessment capture) are reported and skipped rather than silently
    mangled.
    """
    with wave.open(str(path), "rb") as w:
        n_ch, width, rate, n_frames = (w.getnchannels(), w.getsampwidth(),
                                       w.getframerate(), w.getnframes())
        raw = w.readframes(n_frames)

    if width not in (1, 2, 4):
        raise ValueError(f"unsupported sample width {width * 8}-bit")

    # Downmix and resample on the integer samples, while audioop can still help.
    if n_ch > 1 and _audioop is not None:
        raw = _audioop.tomono(raw, width, 0.5, 0.5) if n_ch == 2 else raw
        if n_ch > 2:
            n_ch_eff = n_ch
        else:
            n_ch_eff = 1
    else:
        n_ch_eff = n_ch

    out_rate = rate
    if rate != SAMPLE_RATE and _audioop is not None and n_ch_eff == 1:
        raw, _ = _audioop.ratecv(raw, width, 1, rate, SAMPLE_RATE, None)
        out_rate = SAMPLE_RATE

    if _np is not None:
        dtype = {1: _np.uint8, 2: _np.int16, 4: _np.int32}[width]
        a = _np.frombuffer(raw, dtype=dtype).astype(_np.float32)
        if width == 1:
            a = (a - 128.0) / 128.0
        elif width == 2:
            a = a / 32768.0
        else:
            a = a / 2147483648.0
        if n_ch_eff > 1:
            a = a.reshape(-1, n_ch_eff).mean(axis=1)
        peak = float(abs(a).max()) if a.size else 0.0
        if peak > 1.0:
            a = a / peak
        return _np.ascontiguousarray(a.astype(_np.float32)), rate, out_rate

    # --- pure standard library path ---
    code = {1: "b", 2: "h", 4: "i"}[width]
    ints = array(code)
    ints.frombytes(raw)
    if sys.byteorder == "big":
        ints.byteswap()
    scale = {1: 128.0, 2: 32768.0, 4: 2147483648.0}[width]
    if n_ch_eff > 1:
        n = len(ints) // n_ch_eff
        mono = array("f", (sum(ints[i * n_ch_eff + c] for c in range(n_ch_eff))
                           / n_ch_eff / scale for i in range(n)))
    else:
        offset = 128.0 if width == 1 else 0.0
        mono = array("f", ((s - offset) / scale for s in ints))
    peak = max((abs(s) for s in mono), default=0.0)
    if peak > 1.0:
        mono = array("f", (s / peak for s in mono))
    return mono, rate, out_rate


def write_npy(path: Path, samples) -> None:
    """Write a 1-D float32 .npy without numpy.

    The format is a 6-byte magic, a version, a little-endian header length and a
    short dict describing dtype and shape, padded so the data starts on a 64-byte
    boundary. Writing it by hand removes the last reason this script would need a
    third-party package.
    """
    if _np is not None:
        _np.save(str(path), _np.asarray(samples, dtype=_np.float32),
                 allow_pickle=False)
        return

    n = len(samples)
    header = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({n},), }}"
    prefix = 6 + 2 + 2                      # magic + version + header length
    pad = 64 - ((prefix + len(header) + 1) % 64)
    header = header + " " * pad + "\n"
    with path.open("wb") as fh:
        fh.write(b"\x93NUMPY\x01\x00")
        fh.write(struct.pack("<H", len(header)))
        fh.write(header.encode("latin-1"))
        buf = samples if isinstance(samples, array) else array("f", samples)
        if sys.byteorder == "big":
            buf = array("f", buf)
            buf.byteswap()
        buf.tofile(fh)


# --------------------------------------------------------------------------- #

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

    print(f"numpy {'available' if _np is not None else 'NOT available (using the '
          'standard-library path, which is slower but works)'}")
    if _audioop is None:
        print("audioop not available (Python 3.13+). Audio will be uploaded at its")
        print("original sample rate and resampled on the scoring machine instead;")
        print("the only cost is a larger zip.")
    print()

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
        print(f"ERROR: no files matched --qids {args.qids}")
        print(f"       question ids present: {sorted({qid_of(w) for w in all_wavs})}")
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
    for i, w in enumerate(wavs, 1):
        ciid = w.parent.name if w.parent != in_dir else w.stem.split("_")[0]
        qid = qid_of(w)
        item_id = f"{anon[ciid]}_{qid}"

        try:
            samples, orig_sr, out_sr = read_wav(w)
        except Exception as exc:
            failures.append((str(w), f"{type(exc).__name__}: {exc}"))
            continue

        write_npy(staging / "npy" / f"{item_id}.npy", samples)
        dur = round(len(samples) / out_sr, 3)
        (staging / "npy" / f"{item_id}.json").write_text(json.dumps({
            "item_id": item_id, "question_id": qid,
            "orig_sr": out_sr, "source_sr": orig_sr, "duration_s": dur,
        }), encoding="utf-8")
        rows.append({"anon_id": anon[ciid], "ciid": ciid, "question_id": qid,
                     "item_id": item_id, "source_file": str(w), "duration_s": dur})

        if i % 100 == 0 or i == len(wavs):
            print(f"  {i}/{len(wavs)} converted", flush=True)

    # --- code, if it happens to be here ----------------------------------
    # Optional on purpose. This script is meant to be a single file you drop into
    # the folder holding the audio; when the package is absent, run_scoring.py
    # fetches it from the public repo.
    pkg = find_voxscore()
    if pkg is not None:
        shutil.copytree(pkg, staging / "voxscore",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    zip_path = out_dir / "voxscore_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(staging))
    shutil.rmtree(staging)

    # --- upload.csv: goes WITH the zip, question column left for you -----
    # One row per audio rather than one per question id. If every candidate
    # answered the same question 25 you fill one row and copy down; if the item
    # varied by candidate, the per-audio row is the only thing that can say so.
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

    print()
    print(f"{'items packaged':24s}{len(rows)}")
    print(f"{'candidates':24s}{len(ciids)}")
    print(f"{'question ids':24s}{sorted({r['question_id'] for r in rows})}")
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
