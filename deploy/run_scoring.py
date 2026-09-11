"""STEP 2 (run on the SCORING MACHINE).

Put these three files in one directory and run:

    voxscore_bundle.zip      from make_bundle.py
    run_scoring.py           this file
    requirements.txt

    pip install -r requirements.txt
    python run_scoring.py

Does everything end to end, and every stage is resumable -- rerun it after a
failure and it picks up where it stopped rather than starting over:

    1. extract the bundle            (skipped if already extracted)
    2. download the models           (skipped if already cached)
    3. score every response          (skipped items are reused from cache)
    4. write voxscore_results.xlsx

Output sheets: Scores, Flags, Diagnostics, Features, About.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "voxscore_work"
NPY = WORK / "npy"
MODELS = WORK / "models"
CACHE = WORK / "scored"
MODEL_REPO = "Pransfrance/voxscore-models"


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 1. extract
# --------------------------------------------------------------------------- #

def extract(zip_path: Path) -> None:
    marker = WORK / ".extracted"
    if marker.exists() and NPY.exists() and any(NPY.glob("*.npy")):
        log(f"1/4  bundle already extracted ({len(list(NPY.glob('*.npy')))} items) - skipping")
        return
    log(f"1/4  extracting {zip_path.name} ...")
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(WORK)
    marker.write_text(zip_path.name, encoding="utf-8")
    log(f"     {len(list(NPY.glob('*.npy')))} audio files")


# --------------------------------------------------------------------------- #
# 2. models
# --------------------------------------------------------------------------- #

def fetch_models(token: str | None) -> None:
    marker = MODELS / ".complete"
    if marker.exists():
        log("2/4  models already downloaded - skipping")
        return
    log(f"2/4  downloading models from {MODEL_REPO} (about 6.4 GB, once) ...")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL_REPO, repo_type="model", token=token,
                          local_dir=str(MODELS))
    except Exception as exc:
        log(f"\n     FAILED: {type(exc).__name__}: {exc}")
        log("     If the repo is private, supply a token:")
        log("       set HF_TOKEN=hf_...        (Windows)")
        log("       export HF_TOKEN=hf_...     (Linux/Mac)")
        log("     or make the repo public in its Hugging Face settings.")
        raise SystemExit(1)
    marker.write_text("ok", encoding="utf-8")
    log("     done")


def ensure_spacy_model(name: str = "en_core_web_sm") -> None:
    """Fetch the spaCy English model if it is not installed.

    Kept out of requirements.txt because pinning it there means either a fragile
    direct wheel URL or a second manual command. One fewer step to get wrong on a
    machine you only get to set up once.
    """
    import spacy

    try:
        spacy.load(name)
        return
    except OSError:
        pass
    log(f"     fetching spaCy model {name} ...")
    import subprocess

    subprocess.run([sys.executable, "-m", "spacy", "download", name], check=True)


def point_config_at_local_models() -> None:
    """Rewrite the model registry to use the downloaded copies."""
    from voxscore.config import MODELS as REG

    for key, spec in list(REG.items()):
        local = MODELS / spec.hf_id.replace("/", "__")
        if local.exists():
            object.__setattr__(spec, "hf_id", str(local))


# --------------------------------------------------------------------------- #
# 3. score
# --------------------------------------------------------------------------- #

def score_all(limit: int | None, device: str | None, no_grammar: bool):
    from voxscore.config import PipelineConfig, device_report
    from voxscore.pipeline import Pipeline
    from voxscore.utils.audio_io import load_npy_item, load_questions

    CACHE.mkdir(parents=True, exist_ok=True)
    qfile = WORK / "questions.json"
    questions = load_questions(qfile) if qfile.exists() else {}
    if not questions:
        log("\n     WARNING: no question text available. Relevance, off_topic and")
        log("     prompt_read will be meaningless. Grammar, lexical and fluency are fine.\n")

    files = sorted(NPY.glob("*.npy"))[:limit]
    todo = [f for f in files if not (CACHE / f"{f.stem}.json").exists()]
    log(f"3/4  {len(files)} items, {len(files) - len(todo)} already scored, "
        f"{len(todo)} to do")
    if todo:
        log(f"     {device_report()}")

    pipe = None
    t0 = time.perf_counter()
    for i, f in enumerate(todo, 1):
        if pipe is None:
            pipe = Pipeline(PipelineConfig(device=device), load_grammar=not no_grammar)
        try:
            item = load_npy_item(f, questions=questions)
            res = pipe.score_item(item)
            (CACHE / f"{f.stem}.json").write_text(
                json.dumps(res.to_dict(), ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            (CACHE / f"{f.stem}.json").write_text(json.dumps({
                "item_id": f.stem, "error": f"{type(exc).__name__}: {exc}",
            }), encoding="utf-8")
            log(f"     [{i}/{len(todo)}] {f.stem} FAILED: {type(exc).__name__}: {exc}")
            continue
        if i % 10 == 0 or i == len(todo):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            eta = (len(todo) - i) / max(rate, 1e-9)
            log(f"     [{i}/{len(todo)}]  {rate * 60:.1f} items/min  "
                f"eta {eta / 60:.0f} min")

    return [json.loads((CACHE / f"{f.stem}.json").read_text(encoding="utf-8"))
            for f in files if (CACHE / f"{f.stem}.json").exists()]


# --------------------------------------------------------------------------- #
# 4. excel
# --------------------------------------------------------------------------- #

CATS = ("grammar", "lexical", "fluency", "relevance")


def write_excel(results: list[dict], path: Path) -> None:
    import pandas as pd

    scores, flags, diag, feats = [], [], [], []
    for r in results:
        iid = r.get("item_id", "?")
        anon = iid.rsplit("_", 1)[0] if "_" in iid else iid
        qid = r.get("question_id") or (iid.rsplit("_", 1)[1] if "_" in iid else "")
        if r.get("error"):
            scores.append({"item_id": iid, "anon_id": anon, "question_id": qid,
                           "error": r["error"]})
            continue

        q = r.get("quality", {})
        srow = {"item_id": iid, "anon_id": anon, "question_id": qid,
                "scorable": int(q.get("scorable", 0)),
                "duration_s": r.get("duration_s")}
        for c in CATS:
            sc = r.get("scores", {}).get(c, {})
            srow[c] = sc.get("score")
            srow[f"{c}_confidence"] = sc.get("confidence")
        srow["transcript"] = (r.get("transcript") or "")[:2000]
        scores.append(srow)

        frow = {"item_id": iid, "anon_id": anon, "question_id": qid}
        for fl in r.get("flags", []):
            frow[fl["name"]] = fl["score"]
            frow[f"{fl['name']}_fired"] = int(fl["fired"])
            frow[f"{fl['name']}_why"] = fl.get("evidence", "")[:220]
        flags.append(frow)

        diag.append({"item_id": iid, "anon_id": anon, "question_id": qid,
                     **{k: v for k, v in q.items()},
                     "warnings": "; ".join(r.get("quality_warnings", [])
                                           + r.get("audio_warnings", []))[:500]})

        flat = {"item_id": iid, "anon_id": anon, "question_id": qid}
        for block, fs in (r.get("features") or {}).items():
            for k, v in fs.items():
                flat[f"{block}__{k}"] = v
        feats.append(flat)

    about = pd.DataFrame([
        ("generated", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("items", len(results)),
        ("scorable", sum(1 for s in scores if s.get("scorable") == 1)),
        ("errors", sum(1 for s in scores if s.get("error"))),
        ("score scale", "0-100, higher is better"),
        ("flag scale", "0-100, higher = more suspicious; *_fired uses the default threshold"),
        ("thresholds", "prompt_read 55 | repetition 55 | off_topic 35 | foreign_language 30"),
        ("IMPORTANT", "exclude rows with scorable=0 from any correlation; those are "
                      "silence, too-short, or ASR hallucinations and carry score 0"),
        ("IMPORTANT", "confidence 0 means the measurement was not made, not that the "
                      "candidate scored badly - exclude those too, per category"),
        ("foreign_language", "route to human review, not automatic failure"),
    ], columns=["field", "value"])

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame(scores).to_excel(xl, sheet_name="Scores", index=False)
        pd.DataFrame(flags).to_excel(xl, sheet_name="Flags", index=False)
        pd.DataFrame(diag).to_excel(xl, sheet_name="Diagnostics", index=False)
        pd.DataFrame(feats).to_excel(xl, sheet_name="Features", index=False)
        about.to_excel(xl, sheet_name="About", index=False)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=None, help="bundle path (default: find one here)")
    ap.add_argument("--out", default="voxscore_results.xlsx")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N")
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--no-grammar", action="store_true",
                    help="skip grammar correction; roughly 2x faster")
    ap.add_argument("--token", default=None, help="HF token if the model repo is private")
    args = ap.parse_args()

    zips = [Path(args.zip)] if args.zip else sorted(HERE.glob("*.zip"))
    if not zips or not zips[0].exists():
        log("ERROR: no bundle zip found in this directory.")
        return 1

    extract(zips[0])

    sys.path.insert(0, str(WORK))          # the bundle carries the voxscore package
    os.environ.setdefault("HF_HOME", str(MODELS / "_hf"))

    fetch_models(args.token or os.environ.get("HF_TOKEN"))
    ensure_spacy_model()
    point_config_at_local_models()

    results = score_all(args.limit, args.device, args.no_grammar)
    if not results:
        log("nothing scored")
        return 1

    out = HERE / args.out
    log(f"4/4  writing {out.name} ...")
    write_excel(results, out)

    ok = sum(1 for r in results if not r.get("error")
             and r.get("quality", {}).get("scorable"))
    log("")
    log(f"  done: {out}")
    log(f"  {len(results)} items, {ok} scorable")
    log("")
    log("  Send back the Excel file. Join it to your labels on anon_id + question_id")
    log("  using ciid_mapping.csv, which stayed on your laptop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
