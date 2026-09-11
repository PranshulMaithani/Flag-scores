"""CTC forced alignment: word-level start/end times for the transcript.

Why this exists instead of just using Whisper's timestamps: fluency scoring is
almost entirely a function of *pause boundaries*. Whisper emits segment-level
timestamps quantised to 20 ms but drifting by hundreds of milliseconds in
practice, and a 300 ms error is the difference between "no pause" and "a
hesitation" under the 250 ms threshold. Feeding that noise into the
highest-weight fluency features would cap the achievable correlation before we
started.

We instead align the Whisper transcript against the audio with a wav2vec2 CTC
acoustic model using ``torchaudio.functional.forced_align``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from voxscore.config import SAMPLE_RATE, get_device

log = logging.getLogger(__name__)

ALIGN_MODEL_ID = "facebook/wav2vec2-base-960h"


@dataclass
class Word:
    text: str
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Pause:
    start: float
    end: float
    before_word_idx: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def ctc_forced_align(
    emission: torch.Tensor,
    tokens: list[int],
    blank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Viterbi forced alignment over the CTC lattice.

    Args:
        emission: ``(T, C)`` log-probabilities.
        tokens: target token ids, without blanks.
        blank: id of the CTC blank symbol.

    Returns:
        ``(path, scores)`` where ``path[t]`` is the index into the *blank-extended*
        target sequence assigned to frame ``t``, and ``scores[t]`` is the emission
        probability of the symbol chosen at that frame.

    Implemented here rather than calling ``torchaudio.functional.forced_align``
    for two reasons found by testing: that op has **no CUDA/HIP kernel** (CPU
    only) and it is **deprecated for removal** as torchaudio moves to maintenance.
    Every fluency feature depends on these timings, so the algorithm is worth
    owning outright. Correctness is pinned against the torchaudio reference in
    ``tests/test_align.py`` for as long as that reference exists.

    Standard CTC alignment: the target is interleaved with blanks
    (``b t0 b t1 b ... b``), giving ``S = 2L+1`` states. From state ``s`` a frame
    may stay at ``s``, advance to ``s+1``, or skip to ``s+2`` -- the skip being
    legal only when it lands on a non-blank that differs from the symbol two
    states back, which is what prevents two identical adjacent labels from
    collapsing into one.
    """
    if emission.dim() != 2:
        raise ValueError(f"expected (T, C) emission, got {tuple(emission.shape)}")
    if not tokens:
        raise ValueError("no target tokens to align")

    T = emission.shape[0]
    device = emission.device

    ext = torch.full((2 * len(tokens) + 1,), blank, dtype=torch.long, device=device)
    ext[1::2] = torch.tensor(tokens, dtype=torch.long, device=device)
    S = ext.numel()

    if T < S:
        raise ValueError(f"only {T} frames for {S} alignment states; audio too short")

    # A skip (s-2 -> s) is legal only at a non-blank whose symbol differs from the
    # one two states earlier.
    can_skip = torch.zeros(S, dtype=torch.bool, device=device)
    if S > 2:
        nonblank = ext != blank
        differs = ext[2:] != ext[:-2]
        can_skip[2:] = nonblank[2:] & differs

    NEG = torch.finfo(emission.dtype).min
    emit = emission[:, ext]  # (T, S) score of each state at each frame

    alpha = torch.full((S,), NEG, dtype=emission.dtype, device=device)
    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]

    backptr = torch.zeros((T, S), dtype=torch.int8, device=device)

    for t in range(1, T):
        stay = alpha
        move = torch.cat([torch.full((1,), NEG, dtype=alpha.dtype, device=device), alpha[:-1]])
        skip = torch.cat([torch.full((2,), NEG, dtype=alpha.dtype, device=device), alpha[:-2]])
        skip = torch.where(can_skip, skip, torch.full_like(skip, NEG))

        cand = torch.stack([stay, move, skip], dim=0)  # (3, S)
        best, arg = cand.max(dim=0)
        alpha = best + emit[t]
        backptr[t] = arg.to(torch.int8)

    # Terminate in the final blank or the final real token.
    path = torch.zeros(T, dtype=torch.long, device=device)
    s = S - 1 if S == 1 or alpha[S - 1] >= alpha[S - 2] else S - 2
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        s = s - int(backptr[t, s])
        path[t - 1] = s

    scores = emission[torch.arange(T, device=device), ext[path]].exp()
    return path, scores


@lru_cache(maxsize=1)
def _load_aligner(device_str: str):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(ALIGN_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL_ID).to(torch.device(device_str)).eval()
    return processor, model


def normalise_for_alignment(text: str) -> tuple[list[str], list[str]]:
    """Return (display_words, alignable_words).

    The CTC vocabulary is uppercase A-Z plus apostrophe and a word delimiter, so
    digits and symbols must be removed. We keep the display form alongside so
    reported words match what was actually said.
    """
    display, alignable = [], []
    for raw in text.split():
        cleaned = re.sub(r"[^A-Za-z']", "", raw).upper().strip("'")
        if cleaned:
            display.append(raw.strip(".,!?;:\"()"))
            alignable.append(cleaned)
    return display, alignable


def align_words(
    audio: np.ndarray,
    transcript: str,
    device: torch.device | None = None,
) -> list[Word]:
    """Align ``transcript`` to ``audio``, returning word-level timings.

    Returns an empty list if alignment is impossible (empty transcript, no
    alignable tokens, or a CTC failure). Callers must treat an empty result as
    "fluency features unavailable" rather than as "no pauses".
    """
    device = device or get_device()
    display, alignable = normalise_for_alignment(transcript)
    if not alignable:
        return []

    processor, model = _load_aligner(str(device))
    model = model.to(device)

    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        # fp32 on purpose: alignment is cheap relative to ASR, and bf16 rounding
        # in the CTC lattice can shift boundaries by a frame.
        logits = model(wav).logits
    emission = torch.log_softmax(logits.float(), dim=-1)

    vocab = processor.tokenizer.get_vocab()
    delim = vocab.get("|", 4)
    blank = vocab.get("<pad>", 0)

    tokens: list[int] = []
    spans: list[tuple[int, int]] = []  # token index range per word
    for w in alignable:
        start = len(tokens)
        ids = [vocab[c] for c in w if c in vocab]
        if not ids:
            spans.append((start, start))
            continue
        tokens.extend(ids)
        spans.append((start, len(tokens)))
        tokens.append(delim)

    if tokens and tokens[-1] == delim:
        tokens.pop()
    if not tokens:
        return []

    try:
        path, frame_scores = ctc_forced_align(emission[0], tokens, blank=blank)
    except Exception as exc:  # pragma: no cover
        log.warning("forced alignment failed (%s); no word timings", exc)
        return []

    scores = frame_scores.cpu().numpy()

    # Frame -> time. wav2vec2 downsamples by 320 (20 ms/frame at 16 kHz); derive
    # it from the tensor shapes rather than hardcoding, so swapping the acoustic
    # model cannot silently corrupt every timing.
    sec_per_frame = (len(audio) / SAMPLE_RATE) / emission.shape[1]

    # In the blank-extended sequence, odd states are real tokens and state s maps
    # to token (s-1)//2. This is exact, unlike inferring token boundaries from
    # runs of repeated ids, which cannot distinguish a genuinely doubled letter
    # from one token held across several frames.
    tok_frames: dict[int, list[int]] = {}
    for f, s in enumerate(path.cpu().numpy()):
        if s % 2 == 1:
            tok_frames.setdefault((int(s) - 1) // 2, []).append(f)

    words: list[Word] = []
    for i, (lo, hi) in enumerate(spans):
        frames = [f for p in range(lo, hi) for f in tok_frames.get(p, [])]
        if not frames:
            continue
        s, e = min(frames), max(frames) + 1
        sc = [scores[f] for f in frames]
        words.append(
            Word(
                text=display[i] if i < len(display) else alignable[i],
                start=round(s * sec_per_frame, 3),
                end=round(e * sec_per_frame, 3),
                score=float(np.mean(sc)) if sc else 0.0,
            )
        )

    # Alignment can emit non-monotonic boundaries at word joins; enforce order so
    # downstream pause arithmetic can never produce a negative duration.
    for a, b in zip(words, words[1:]):
        if b.start < a.end:
            mid = (a.end + b.start) / 2
            a.end = b.start = round(mid, 3)

    return words


def extract_pauses(
    words: list[Word],
    total_duration: float,
    min_pause_s: float = 0.25,
    trim_edges: bool = True,
) -> list[Pause]:
    """Silences between aligned words that exceed ``min_pause_s``.

    Leading and trailing silence is excluded by default: it reflects when the
    recorder started and stopped, not the candidate's fluency. Counting it would
    penalise whoever had a slow interface.
    """
    if not words:
        return []
    pauses: list[Pause] = []
    for i, (a, b) in enumerate(zip(words, words[1:]), start=1):
        gap = b.start - a.end
        if gap >= min_pause_s:
            pauses.append(Pause(start=a.end, end=b.start, before_word_idx=i))

    if not trim_edges:
        if words[0].start >= min_pause_s:
            pauses.insert(0, Pause(0.0, words[0].start, 0))
        tail = total_duration - words[-1].end
        if tail >= min_pause_s:
            pauses.append(Pause(words[-1].end, total_duration, len(words)))
    return pauses


def phonation_time(words: list[Word]) -> float:
    """Total time spent actually producing speech."""
    return float(sum(w.duration for w in words))
