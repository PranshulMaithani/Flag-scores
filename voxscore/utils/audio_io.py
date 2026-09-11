"""Audio loading, the .npy contract, and the wav -> npy converter.

The client cannot handle .wav on their machine, so audio reaches us as .npy.
That makes this module the actual integration surface, and it is written to be
forgiving about what it receives: int16 or float32, mono or stereo, any sample
rate, with or without a sidecar. Anything ambiguous is reported loudly rather
than silently coerced, because a silent sample-rate error would corrupt every
downstream duration-based feature (all of fluency) without ever raising.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

from voxscore.config import SAMPLE_RATE, MAX_DURATION_S


# --------------------------------------------------------------------------- #
# Item container
# --------------------------------------------------------------------------- #

@dataclass
class AudioItem:
    """One candidate response, conditioned and ready for the pipeline."""

    item_id: str
    audio: np.ndarray          # float32, mono, SAMPLE_RATE, range [-1, 1]
    sample_rate: int
    duration_s: float
    question_id: str | None = None
    question_text: str | None = None
    ideal_answers: list[str] | None = None
    source_path: Path | None = None
    warnings_: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings_ is None:
            self.warnings_ = []

    def meta(self) -> dict:
        """Serialisable metadata, excluding the waveform."""
        d = asdict(self)
        d.pop("audio")
        d["source_path"] = str(self.source_path) if self.source_path else None
        return d


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #

_INT_DTYPES = {
    np.dtype(np.int16): 32768.0,
    np.dtype(np.int32): 2147483648.0,
    np.dtype(np.int8): 128.0,
}


def coerce_waveform(
    arr: np.ndarray,
    orig_sr: int | None,
    item_id: str = "?",
) -> tuple[np.ndarray, list[str]]:
    """Bring an arbitrary array into the internal contract.

    Returns the conditioned waveform and a list of human-readable warnings.
    Warnings are returned rather than logged so they can be attached to the
    item and surfaced in the output XML -- the client needs to be able to tell
    a bad recording from a bad answer.
    """
    notes: list[str] = []
    a = np.asarray(arr)

    if a.ndim == 2:
        # Accept both (channels, samples) and (samples, channels).
        if a.shape[0] <= 8 < a.shape[1]:
            a = a.mean(axis=0)
        elif a.shape[1] <= 8 < a.shape[0]:
            a = a.mean(axis=1)
        else:
            a = a.reshape(-1)
        notes.append("downmixed multi-channel audio to mono")
    elif a.ndim > 2:
        raise ValueError(f"{item_id}: cannot interpret array of shape {a.shape}")

    # Integer PCM -> float. Scale by the dtype's full range, not by observed max,
    # so a quiet recording stays quiet instead of being silently normalised.
    if a.dtype in _INT_DTYPES:
        scale = _INT_DTYPES[a.dtype]
        notes.append(f"converted {a.dtype} PCM to float32 (/{scale:.0f})")
        a = a.astype(np.float32) / scale
    else:
        a = a.astype(np.float32, copy=False)

    if not np.all(np.isfinite(a)):
        n_bad = int((~np.isfinite(a)).sum())
        notes.append(f"replaced {n_bad} non-finite samples with 0")
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1.0:
        # Almost always float data that was never divided down.
        notes.append(f"peak {peak:.2f} exceeded 1.0; rescaled")
        a = a / peak
    elif 0 < peak < 1e-3:
        notes.append(f"near-silent input (peak {peak:.2e})")

    if orig_sr is None:
        notes.append(
            f"sample rate unknown; ASSUMED {SAMPLE_RATE} Hz. "
            "If wrong, every duration-based feature is wrong."
        )
        orig_sr = SAMPLE_RATE

    if orig_sr != SAMPLE_RATE:
        t = torch.from_numpy(a).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_sr, SAMPLE_RATE)
        a = t.squeeze(0).numpy().astype(np.float32)
        notes.append(f"resampled {orig_sr} -> {SAMPLE_RATE} Hz")

    dur = len(a) / SAMPLE_RATE
    if dur > MAX_DURATION_S * 1.5:
        notes.append(
            f"duration {dur:.1f}s far exceeds the {MAX_DURATION_S:.0f}s cap; "
            "likely a sample-rate mismatch rather than a long answer"
        )

    return np.ascontiguousarray(a), notes


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_npy_item(
    npy_path: str | Path,
    sidecar: dict | None = None,
    questions: dict | None = None,
) -> AudioItem:
    """Load one ``.npy`` response, plus its sidecar JSON if present.

    Sidecar schema (all optional except ``item_id``)::

        {"item_id": "...", "question_id": "...", "orig_sr": 16000, "duration_s": 12.3}
    """
    npy_path = Path(npy_path)
    arr = np.load(npy_path, allow_pickle=False)

    if sidecar is None:
        side_path = npy_path.with_suffix(".json")
        sidecar = json.loads(side_path.read_text(encoding="utf-8")) if side_path.exists() else {}

    item_id = sidecar.get("item_id") or npy_path.stem
    orig_sr = sidecar.get("orig_sr") or sidecar.get("sample_rate")

    # A declared duration lets us recover the true sample rate when it is missing
    # or wrong -- the single most damaging silent failure in this pipeline.
    declared_dur = sidecar.get("duration_s")
    if orig_sr is None and declared_dur:
        inferred = len(arr.reshape(-1)) / float(declared_dur)
        for cand in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
            if abs(inferred - cand) / cand < 0.05:
                orig_sr = cand
                break

    audio, notes = coerce_waveform(arr, orig_sr, item_id)

    if declared_dur:
        actual = len(audio) / SAMPLE_RATE
        if abs(actual - float(declared_dur)) > max(0.5, 0.1 * float(declared_dur)):
            notes.append(
                f"declared duration {float(declared_dur):.1f}s but decoded "
                f"{actual:.1f}s -- sample rate is probably wrong"
            )

    qid = sidecar.get("question_id")
    qtext, ideals = None, None
    if questions and qid and qid in questions:
        q = questions[qid]
        qtext = q.get("text")
        ideals = q.get("ideal_answers")

    return AudioItem(
        item_id=item_id,
        audio=audio,
        sample_rate=SAMPLE_RATE,
        duration_s=len(audio) / SAMPLE_RATE,
        question_id=qid,
        question_text=qtext,
        ideal_answers=ideals,
        source_path=npy_path,
        warnings_=notes,
    )


def load_questions(path: str | Path) -> dict:
    """Load ``questions.json``.

    Schema::

        {"q001": {"text": "...", "ideal_answers": ["...", "...", "..."]}}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for qid, q in data.items():
        if "text" not in q:
            raise ValueError(f"question {qid!r} has no 'text'")
        n = len(q.get("ideal_answers") or [])
        if n < 2:
            warnings.warn(
                f"question {qid!r} has {n} ideal answers. The relevance rubric needs "
                "at least 2 to separate required content from personal specifics; "
                "it will fall back to question-only features.",
                stacklevel=2,
            )
    return data


# --------------------------------------------------------------------------- #
# Converter shipped to the client
# --------------------------------------------------------------------------- #

def wav_to_npy(
    wav_path: str | Path,
    out_dir: str | Path,
    item_id: str | None = None,
    question_id: str | None = None,
) -> tuple[Path, Path]:
    """Convert one wav to the .npy contract plus sidecar. Returns both paths.

    This is the client-side entry point: it runs where the wavs live, and only
    the .npy files need to move.
    """
    wav_path, out_dir = Path(wav_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    item_id = item_id or wav_path.stem

    # soundfile rather than torchaudio.load: torchaudio 2.9 removed its own
    # decoding backends and now requires torchcodec, which is a heavy extra
    # dependency for a converter that only ever sees wav files.
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)

    if sr != SAMPLE_RATE:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        audio = t.squeeze(0).numpy().astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    npy_path = out_dir / f"{item_id}.npy"
    json_path = out_dir / f"{item_id}.json"
    np.save(npy_path, audio, allow_pickle=False)
    json_path.write_text(
        json.dumps(
            {
                "item_id": item_id,
                "question_id": question_id,
                "orig_sr": SAMPLE_RATE,
                "duration_s": round(len(audio) / SAMPLE_RATE, 3),
                "source_sr": sr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return npy_path, json_path
