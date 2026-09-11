"""Audio loading from HuggingFace datasets without torchcodec.

``datasets`` 5.x and ``torchaudio`` 2.9 both dropped their built-in decoders and
now require ``torchcodec``, which wants system FFmpeg libraries and has no
usable ROCm/Windows path. Rather than take that dependency -- which would also
have to be satisfied on the client's locked-down machine -- we disable the
decoding feature and decode the raw bytes with ``soundfile``.

This works for wav/flac/ogg, which covers every corpus in the eval plan
(LibriSpeech, FLEURS, Svarah, Common Voice). MP3-only datasets would need
another path; none of ours are.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from voxscore.config import SAMPLE_RATE

log = logging.getLogger(__name__)


@dataclass
class HFClip:
    audio: np.ndarray     # float32 mono @ SAMPLE_RATE
    sample_rate: int
    reference: str        # gold transcript, or "" when unavailable
    clip_id: str
    extra: dict


def undecoded(ds):
    """Return ``ds`` with its audio column left as raw bytes.

    Must be applied before any indexing, or datasets will try to decode.
    """
    from datasets import Audio

    col = "audio" if "audio" in ds.column_names else None
    if col is None:
        for c in ("path", "file"):
            if c in ds.column_names:
                return ds
        raise ValueError(f"no audio column in {ds.column_names}")
    return ds.cast_column(col, Audio(decode=False))


def decode_clip(raw: dict | str | bytes, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Decode one dataset audio value to mono float32 at ``target_sr``."""
    data: bytes | None = None
    path: str | None = None

    if isinstance(raw, dict):
        data = raw.get("bytes")
        path = raw.get("path")
    elif isinstance(raw, bytes):
        data = raw
    else:
        path = str(raw)

    if data:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    elif path and Path(path).exists():
        arr, sr = sf.read(path, dtype="float32", always_2d=True)
    else:
        raise ValueError(f"cannot decode audio: bytes={bool(data)} path={path!r}")

    mono = arr.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        t = torch.from_numpy(mono).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        mono = t.squeeze(0).numpy().astype(np.float32)
    return np.ascontiguousarray(mono), target_sr


_REF_COLUMNS = ("text", "transcription", "sentence", "normalized_text", "raw_transcription")


def load_clips(
    dataset_id: str,
    config: str | None = None,
    split: str = "validation",
    limit: int | None = None,
    **kw,
) -> list[HFClip]:
    """Load and decode up to ``limit`` clips from a HF dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=split, **kw)
    ds = undecoded(ds)

    ref_col = next((c for c in _REF_COLUMNS if c in ds.column_names), None)
    audio_col = "audio" if "audio" in ds.column_names else "path"

    n = len(ds) if limit is None else min(limit, len(ds))
    clips: list[HFClip] = []
    for i in range(n):
        row = ds[i]
        try:
            audio, sr = decode_clip(row[audio_col])
        except Exception as exc:
            log.warning("clip %d of %s failed to decode: %s", i, dataset_id, exc)
            continue
        clips.append(
            HFClip(
                audio=audio,
                sample_rate=sr,
                reference=str(row.get(ref_col, "")) if ref_col else "",
                clip_id=str(row.get("id", row.get("path", i))),
                extra={k: v for k, v in row.items() if k not in (audio_col,)},
            )
        )
    return clips
