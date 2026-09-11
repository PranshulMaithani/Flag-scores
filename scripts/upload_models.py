"""Mirror the pinned model weights to a single Hugging Face repo.

Why mirror at all, when every one of these is already on the Hub:

* **Pinning.** One repo at known revisions, so a run today and a run in six
  months use identical weights. Upstream repos get updated and occasionally
  removed.
* **One download.** Five repos become one, which matters on a restricted machine.
* **Offline.** If the scoring machine cannot reach the Hub, the client can pull
  this once from somewhere that can and move a single directory.

Every model here is Apache-2.0 or MIT, all of which permit redistribution with
attribution, and the generated README carries the source repo, exact revision and
licence for each. `.bin` duplicates are skipped where safetensors exist, which
roughly halves the upload.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/upload_models.py [--private] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import MODELS, MODELS_CACHE, PROJECT_ROOT

DEFAULT_REPO = "voxscore-models"

# Formats we do not need. safetensors is preferred and present for all five.
SKIP_SUFFIXES = {".bin", ".h5", ".msgpack", ".ot"}
SKIP_NAMES = {".gitattributes"}


def read_token() -> str | None:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    import os
    return os.environ.get("HF_TOKEN")


def find_snapshot(hf_id: str) -> tuple[Path, str] | None:
    """Locate the cached snapshot for ``hf_id``, following refs/main.

    Must read refs/main rather than take the first directory under snapshots/:
    the cache keeps one directory per revision it has ever touched, including
    partial ones fetched for a PR ref or an interrupted download. Sorting by
    name is sorting by commit hash, which is arbitrary.

    This shipped a broken bundle. Unbabel/gec-t5_small had two snapshots -- a
    weights-only 779c... and the real c958... -- and '7' sorts before 'c', so
    the uploaded model folder contained model.safetensors and nothing else. No
    config.json and no tokenizer meant AutoTokenizer could not identify a class
    to build, and the resulting "Couldn't instantiate the backend tokenizer"
    names sentencepiece in its text, which sent the diagnosis somewhere else
    entirely for two rounds.
    """
    d = MODELS_CACHE / "hub" / ("models--" + hf_id.replace("/", "--"))
    if not (d / "snapshots").exists():
        return None

    ref = d / "refs" / "main"
    if ref.exists():
        rev = ref.read_text(encoding="utf-8").strip()
        if (d / "snapshots" / rev).is_dir():
            return d / "snapshots" / rev, rev

    # No main ref: fall back to whichever snapshot is actually complete, which
    # for every model here means it carries a config.
    snaps = [p for p in sorted((d / "snapshots").glob("*")) if p.is_dir()]
    for p in snaps:
        if (p / "config.json").exists():
            return p, p.name
    return (snaps[0], snaps[0].name) if snaps else None


def stage(dest: Path) -> tuple[dict, float]:
    """Copy the needed files into a flat, per-model layout. Returns manifest + GB."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    manifest, total = {}, 0
    for key, spec in MODELS.items():
        found = find_snapshot(spec.hf_id)
        if not found:
            print(f"  {key:10s} {spec.hf_id:46s} NOT CACHED - skipped")
            continue
        snap, revision = found
        out = dest / spec.hf_id.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)

        # Skip .bin ONLY where a safetensors equivalent exists. bge-m3 ships
        # pytorch_model.bin and no safetensors at all, so a blanket skip dropped
        # the only weights file and staged a 0.02 GB "model".
        has_safetensors = any(f.suffix == ".safetensors" for f in snap.rglob("*"))

        copied, size = [], 0
        # rglob, not iterdir: sentence-transformers models carry required
        # subdirectories (bge-m3 needs 1_Pooling/) that a flat copy silently
        # drops, producing a bundle that downloads fine and then fails to load.
        for f in sorted(snap.rglob("*")):
            if not f.is_file() or f.name in SKIP_NAMES:
                continue
            rel = f.relative_to(snap)
            # onnx/ and openvino/ are alternative runtimes we do not use.
            if rel.parts and rel.parts[0] in {"onnx", "openvino", "coreml"}:
                continue
            if f.suffix in SKIP_SUFFIXES and has_safetensors:
                continue
            real = f.resolve()
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(real, target)
            copied.append(str(rel).replace("\\", "/"))
            size += real.stat().st_size

        manifest[key] = {
            "hf_id": spec.hf_id, "revision": revision, "licence": spec.licence,
            "purpose": spec.purpose, "files": sorted(copied),
            "size_gb": round(size / 1e9, 3),
            "local_dir": spec.hf_id.replace("/", "__"),
        }
        total += size
        print(f"  {key:10s} {spec.hf_id:46s} {size/1e9:5.2f} GB  rev={revision[:12]}")
    return manifest, total / 1e9


def write_readme(dest: Path, manifest: dict, repo_id: str) -> None:
    rows = "\n".join(
        f"| `{m['hf_id']}` | {m['licence']} | `{m['revision'][:12]}` | "
        f"{m['size_gb']:.2f} GB | {m['purpose']} |"
        for m in manifest.values()
    )
    (dest / "README.md").write_text(f"""---
license: apache-2.0
tags:
  - speech
  - assessment
  - mirror
---

# voxscore model bundle

Pinned mirror of the models used by **voxscore**, a self-hosted scoring pipeline
for spoken open-ended assessment responses. Nothing here is original work — these
are redistributions at fixed revisions so that a scoring run is reproducible and
can be performed on a machine with limited network access.

## Contents

| Source | Licence | Revision | Size | Used for |
|---|---|---|---|---|
{rows}

Every model is Apache-2.0 or MIT, both of which permit redistribution with
attribution. Full credit belongs to the original authors; please cite and follow
the licence of the upstream repository rather than this mirror.

`.bin` / `.h5` duplicates are omitted where safetensors are available.

## Use

```python
from huggingface_hub import snapshot_download
path = snapshot_download("{repo_id}", repo_type="model")
# then point voxscore at it:
#   VOXSCORE_MODEL_DIR=<path>
```

Or fetch a single model:

```python
snapshot_download("{repo_id}", allow_patterns=["openai__whisper-large-v3/*"])
```

`manifest.json` carries the source repo, revision and licence for each entry in
machine-readable form.

## Why a mirror

Pinning (upstream repos change and are occasionally removed), one download
instead of five, and the ability to move a single directory to a restricted
machine. If you can reach the Hub, prefer the original repositories.
""", encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = read_token()
    if not token:
        print("no HF_TOKEN in .env or environment")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.repo}"

    staging = PROJECT_ROOT / "data" / "interim" / "model_bundle"
    print(f"staging to {staging}\n")
    manifest, gb = stage(staging)
    if not manifest:
        print("nothing staged")
        return 1
    write_readme(staging, manifest, repo_id)
    print(f"\nstaged {len(manifest)} models, {gb:.2f} GB -> {repo_id} "
          f"({'private' if args.private else 'PUBLIC'})")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return 0

    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    print("uploading (this takes a while) ...", flush=True)
    api.upload_folder(
        folder_path=str(staging), repo_id=repo_id, repo_type="model",
        commit_message="Pinned mirror of voxscore model dependencies",
        # Mirror the staging directory exactly. Without this, a file uploaded by
        # an earlier, buggier run survives forever: the first upload staged the
        # wrong gec revision, and its stray model.safetensors would have kept
        # being preferred over the correct pytorch_model.bin beside it -- right
        # config, wrong weights, no error.
        delete_patterns="*",
    )
    print(f"\ndone: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
