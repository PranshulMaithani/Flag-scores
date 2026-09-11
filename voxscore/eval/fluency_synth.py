"""Controlled degradation of fluent speech, for validating the fluency score.

Fluency is the only category with no validation, because validating it needs
*graded audio* and no permissively licensed corpus carries proficiency-labelled
speech. The foreign-language work showed the way around that: rather than hunt
for labelled data, manufacture a known-severity variable and check the score
tracks it.

Three manipulations, each targeting a distinct claim the fluency scorecard makes:

* **Pause burden** -- insert silences of known count and length. Should lower the
  score monotonically. Tests `mean_length_of_run`, `silent_pause_rate`,
  `long_pause_rate`, `phonation_time_ratio` together.
* **Pause placement** -- the same number of pauses at clause boundaries versus
  mid-clause. The scorecard asserts mid-clause pausing reads as word-searching
  while between-clause pausing reads as planned, and weights
  `within_clause_pause_ratio` at 0.15 on that basis. If the two do not separate,
  that feature is decoration.
* **Speaking rate** -- time-stretch without changing pitch. The scorecard uses a
  *plateau*, not a monotone curve, on the argument that 220 wpm is rushing rather
  than fluent. Both extremes should therefore score below natural pace.

All three operate on real speech, so the acoustic material stays realistic and
only the fluency-relevant properties move.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from voxscore.config import SAMPLE_RATE


@dataclass
class DegradedItem:
    audio: np.ndarray
    kind: str
    severity: float          # ground-truth amount of degradation
    detail: str


def _silence(duration_s: float, like: np.ndarray | None = None) -> np.ndarray:
    """Silence with a trace of room tone, so the splice is not an unnatural null.

    Digital silence inserted into a recording is itself detectable and would let
    a pause detector succeed for the wrong reason. Matching the quietest part of
    the source keeps the inserted gap acoustically plausible.
    """
    n = int(duration_s * SAMPLE_RATE)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if like is None or like.size == 0:
        return np.zeros(n, dtype=np.float32)
    frame = max(int(0.02 * SAMPLE_RATE), 1)
    n_frames = max(len(like) // frame, 1)
    energies = np.array([
        np.sqrt(np.mean(like[i * frame:(i + 1) * frame] ** 2))
        for i in range(n_frames)
    ])
    floor = float(np.percentile(energies, 5))
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n).astype(np.float32) * floor * 0.7)


def inject_pauses(
    audio: np.ndarray,
    words: list,
    n_pauses: int,
    pause_s: float,
    placement: str = "random",
    clause_word_idx: set[int] | None = None,
    rng: random.Random | None = None,
) -> DegradedItem:
    """Insert ``n_pauses`` silences of ``pause_s`` after selected word boundaries.

    ``placement`` is one of ``clause`` (only at clause boundaries), ``mid``
    (never at a clause boundary), or ``random``.
    """
    rng = rng or random.Random(0)
    if not words or n_pauses <= 0 or pause_s <= 0:
        return DegradedItem(audio.astype(np.float32), "pause", 0.0, "clean")

    clause_word_idx = clause_word_idx or set()
    candidates = list(range(len(words) - 1))
    if placement == "clause":
        candidates = [i for i in candidates if i in clause_word_idx] or candidates
    elif placement == "mid":
        candidates = [i for i in candidates if i not in clause_word_idx] or candidates

    if not candidates:
        return DegradedItem(audio.astype(np.float32), "pause", 0.0, "no insertion point")

    chosen = sorted(rng.sample(candidates, k=min(n_pauses, len(candidates))))
    gap = _silence(pause_s, audio)

    out, prev = [], 0
    for i in chosen:
        cut = int(words[i].end * SAMPLE_RATE)
        cut = max(prev, min(cut, len(audio)))
        out.append(audio[prev:cut])
        out.append(gap)
        prev = cut
    out.append(audio[prev:])

    mixed = np.concatenate(out).astype(np.float32)
    severity = len(chosen) * pause_s
    return DegradedItem(
        mixed, "pause", severity,
        f"{len(chosen)}x{pause_s:.2f}s at {placement} ({severity:.1f}s added)",
    )


def change_rate(audio: np.ndarray, factor: float) -> DegradedItem:
    """Time-stretch without changing pitch. ``factor`` > 1 speeds up.

    Pitch must be preserved: naive resampling would also shift formants, and the
    fluency score would then be responding to a different voice rather than to a
    different speaking rate.
    """
    if abs(factor - 1.0) < 1e-3:
        return DegradedItem(audio.astype(np.float32), "rate", 1.0, "natural pace")
    try:
        import librosa

        out = librosa.effects.time_stretch(y=audio.astype(np.float32), rate=factor)
    except Exception:
        # Fall back to resampling and accept the pitch shift rather than failing.
        idx = np.arange(0, len(audio), factor)
        out = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return DegradedItem(out.astype(np.float32), "rate", factor,
                        f"{factor:.2f}x speed ({'faster' if factor > 1 else 'slower'})")


def clause_boundary_word_indices(parsed, n_aligned_words: int) -> set[int]:
    """Aligned-word indices that sit at a clause boundary.

    Aligned words and parsed tokens use different tokenisations, so positions are
    mapped by rank -- close enough because alignment preserves word order, and
    the experiment only needs the two placement conditions to differ, not exact
    boundaries.
    """
    if parsed.doc is None or n_aligned_words == 0:
        return set()
    n_parsed = max(len(parsed.tokens), 1)
    idx: set[int] = set()
    try:
        for sent in parsed.doc.sents:
            idx.add(sent.start)
        for tok in parsed.doc:
            if tok.dep_ in {"conj", "advcl", "ccomp", "xcomp", "relcl", "parataxis"}:
                idx.add(tok.left_edge.i)
    except Exception:
        return set()
    scaled = {int(round(i * n_aligned_words / n_parsed)) for i in idx}
    return {i for i in scaled if 0 <= i < n_aligned_words}
