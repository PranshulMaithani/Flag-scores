"""STEP 1 (run on YOUR laptop, where the wavs are).

Turns

    Audiostest/audios/<ciid>/<ciid>_<qid>.wav

into two things:

    voxscore_bundle.zip   anonymised .npy audio + the voxscore code. UPLOAD THIS.
    ciid_mapping.csv      anonymous id -> real ciid. KEEP THIS. DO NOT UPLOAD.

Usage
-----
    python make_bundle.py
    python make_bundle.py --input Audiostest/audios --questions questions.csv

A note on what "anonymised" means here: this replaces the *identifier*. The audio
is still a recording of someone's voice, which is biometric data. The bundle is
safe to move between your own machines; it is not safe to treat as de-identified
in the sense a privacy review would mean.
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


def load_questions(path: Path | None) -> dict:
    """Accept either JSON ({qid: {text, ideal_answers}}) or a two-column CSV."""
    if path is None or not path.exists():
        return {}
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return {}

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return {str(k): v for k, v in data.items() if not str(k).startswith("_")}

    out: dict = {}
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return {}
    start = 1 if rows[0] and not rows[0][0].strip().isdigit() else 0
    for row in rows[start:]:
        if len(row) >= 2 and row[0].strip():
            out[row[0].strip()] = {"text": row[1].strip()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="Audiostest/audios",
                    help="directory containing one folder per ciid")
    ap.add_argument("--out", default=".", help="where to write the zip and mapping")
    ap.add_argument("--questions", default=None,
                    help="questions.json or a CSV of question_id,question_text")
    ap.add_argument("--prefix", default="S", help="anonymous id prefix")
    ap.add_argument("--seed", type=int, default=20260911,
                    help="shuffle seed; anonymous ids do not follow ciid order")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_dir.exists():
        print(f"ERROR: {in_dir} does not exist")
        return 1

    wavs = sorted(in_dir.glob("*/*.wav")) or sorted(in_dir.glob("*.wav"))
    if not wavs:
        print(f"ERROR: no .wav files under {in_dir}")
        print("       expected <input>/<ciid>/<ciid>_<qid>.wav")
        return 1

    ciids = sorted({(w.parent.name if w.parent != in_dir else w.stem.split("_")[0])
                    for w in wavs})
    # Shuffle before numbering, so S0001 is not simply the alphabetically first
    # candidate. The mapping file is the only way back.
    shuffled = list(ciids)
    random.Random(args.seed).shuffle(shuffled)
    anon = {c: f"{args.prefix}{i + 1:04d}" for i, c in enumerate(shuffled)}

    questions = load_questions(Path(args.questions) if args.questions else None)

    staging = out_dir / "_bundle_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "npy").mkdir(parents=True)

    print(f"{len(wavs)} wav files across {len(ciids)} candidates\n")

    rows, qids_seen, failures = [], set(), []
    for w in wavs:
        ciid = w.parent.name if w.parent != in_dir else w.stem.split("_")[0]
        stem = w.stem
        # Everything after the last underscore is treated as the question id.
        qid = stem.rsplit("_", 1)[1] if "_" in stem else "unknown"
        qids_seen.add(qid)
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

    # --- questions -------------------------------------------------------
    missing_q = sorted(q for q in qids_seen if q not in questions)
    if not questions:
        print("!" * 72)
        print("WARNING: no questions file supplied.")
        print("Relevance, off_topic and prompt_read all need the QUESTION TEXT.")
        print("Without it those three outputs are meaningless; grammar, lexical")
        print("and fluency are unaffected.")
        print("Pass --questions questions.csv  (columns: question_id,question_text)")
        print("!" * 72 + "\n")
    elif missing_q:
        print(f"WARNING: {len(missing_q)} question ids have no text: "
              f"{missing_q[:10]}{'...' if len(missing_q) > 10 else ''}\n")

    (staging / "questions.json").write_text(
        json.dumps(questions or {}, indent=1, ensure_ascii=False), encoding="utf-8")

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

    # --- mapping, deliberately OUTSIDE the zip ---------------------------
    map_path = out_dir / "ciid_mapping.csv"
    with map_path.open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=["anon_id", "ciid", "question_id",
                                             "item_id", "source_file", "duration_s"])
        wri.writeheader()
        wri.writerows(rows)

    size_gb = zip_path.stat().st_size / 1e9
    print(f"{'items packaged':24s}{len(rows)}")
    print(f"{'candidates':24s}{len(ciids)}")
    print(f"{'distinct questions':24s}{len(qids_seen)}")
    if failures:
        print(f"{'FAILED to read':24s}{len(failures)}")
        for f, why in failures[:5]:
            print(f"    {f}: {why}")
    print()
    print(f"  UPLOAD THIS      {zip_path}   ({size_gb:.2f} GB)")
    print(f"  KEEP THIS LOCAL  {map_path}   (maps anonymous ids back to ciids)")
    print()
    print("Next: copy voxscore_bundle.zip, run_scoring.py and requirements.txt to")
    print("the scoring machine, then run   python run_scoring.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
