"""Synthesise labelled edge cases for flag evaluation.

The client has no flag labels and is building their own edge-case sets. We can
manufacture ours, and for two of the four flags the construction gives a
*continuous* ground-truth variable rather than a binary label -- the proportion
of non-English audio, or the fraction of the answer that is duplicated. That is
strictly more useful than a positive/negative set, because it yields a
score-versus-severity curve, and it is the curve that tells the client where to
put a threshold.

Everything here is evaluation-only and never touches a shipped parameter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from voxscore.config import SAMPLE_RATE


@dataclass
class SynthItem:
    """One synthesised item with its ground truth."""

    audio: np.ndarray
    label: float              # continuous severity, 0.0 = clean
    kind: str
    detail: str = ""
    transcript: str | None = None


def _crossfade(a: np.ndarray, b: np.ndarray, ms: float = 30.0) -> np.ndarray:
    """Join two clips with a short fade so the splice is not an audible click.

    A hard cut leaves a broadband transient that language ID and the ASR both
    react to, which would let the detector succeed for the wrong reason -- we
    would be measuring splice detection, not language detection.
    """
    n = int(ms * SAMPLE_RATE / 1000)
    if len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    head, tail = a[:-n], b[n:]
    mid = a[-n:] * (1 - ramp) + b[:n] * ramp
    return np.concatenate([head, mid, tail]).astype(np.float32)


def _match_level(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale ``src`` to the RMS of ``ref``.

    Without this the spliced segment often sits at a different loudness, and a
    detector could key on the level change instead of the language.
    """
    s_rms = float(np.sqrt(np.mean(src ** 2))) or 1e-9
    r_rms = float(np.sqrt(np.mean(ref ** 2))) or 1e-9
    return np.clip(src * (r_rms / s_rms), -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# foreign_language
# --------------------------------------------------------------------------- #

def make_foreign_mix(
    english: np.ndarray,
    foreign: np.ndarray,
    proportion: float,
    lang: str,
    rng: random.Random | None = None,
) -> SynthItem:
    """Replace ``proportion`` of an English response with contiguous foreign speech.

    Contiguous rather than scattered, because that is what real code-switching to
    a whole other language looks like: a candidate who cannot continue in English
    switches and stays switched. Scattered single words are a different
    phenomenon, handled by the code-switching cases below.
    """
    rng = rng or random.Random(0)
    proportion = float(np.clip(proportion, 0.0, 1.0))
    if proportion <= 0:
        return SynthItem(english.astype(np.float32), 0.0, "foreign_language", f"0% ({lang})")
    if proportion >= 1.0:
        return SynthItem(_match_level(foreign, english), 1.0, "foreign_language", f"100% ({lang})")

    n_total = len(english)
    n_foreign = int(n_total * proportion)
    seg = _match_level(foreign, english)
    if len(seg) < n_foreign:
        seg = np.tile(seg, int(np.ceil(n_foreign / len(seg))))
    seg = seg[:n_foreign]

    # Insert away from the very start: candidates typically begin in English and
    # switch when they run out of words, and a detector that only ever sees
    # foreign audio at t=0 would be tuned for the wrong case.
    lo = int(n_total * 0.15)
    hi = max(lo + 1, n_total - n_foreign)
    cut = rng.randint(lo, hi) if hi > lo else lo

    mixed = _crossfade(english[:cut], seg)
    mixed = _crossfade(mixed, english[cut:])
    return SynthItem(mixed.astype(np.float32), proportion, "foreign_language",
                     f"{proportion:.0%} {lang} inserted at {cut / SAMPLE_RATE:.1f}s")


def make_code_switch(
    english: np.ndarray,
    foreign: np.ndarray,
    n_switches: int,
    switch_ms: float = 700.0,
    lang: str = "?",
    rng: random.Random | None = None,
) -> SynthItem:
    """Scatter several short foreign fragments through English speech.

    This is the hard case and the one that decides the flag's fairness. A
    candidate dropping two Hindi words into an otherwise English answer is doing
    something very different from one answering entirely in Hindi, and the client
    said they want to set that boundary themselves. Producing both shapes lets
    them see where the score separates them.
    """
    rng = rng or random.Random(0)
    n = int(switch_ms * SAMPLE_RATE / 1000)
    src = _match_level(foreign, english)
    out = english.astype(np.float32).copy()

    positions = sorted(rng.sample(
        range(int(len(out) * 0.1), max(int(len(out) * 0.9) - n, int(len(out) * 0.1) + 1)),
        k=min(n_switches, 6),
    )) if len(out) > 3 * n else []

    for i, p in enumerate(positions):
        start = (i * n) % max(len(src) - n, 1)
        frag = src[start: start + n]
        if len(frag) < n:
            continue
        out[p: p + n] = frag

    total = len(positions) * n / max(len(out), 1)
    return SynthItem(out, total, "code_switch",
                     f"{len(positions)}x{switch_ms:.0f}ms {lang} ({total:.0%} of audio)")


# --------------------------------------------------------------------------- #
# repetition
# --------------------------------------------------------------------------- #

def make_repetition(
    audio: np.ndarray,
    repeat_fraction: float,
    n_repeats: int = 1,
    rng: random.Random | None = None,
) -> SynthItem:
    """Duplicate a contiguous span, the way a candidate pads to fill time.

    The span is re-inserted immediately after itself, which is what restating the
    same point sounds like. Because the duplicate is bit-identical audio this
    also exercises ``audio_self_similarity``, which is the only channel that can
    see a literal replay.
    """
    rng = rng or random.Random(0)
    repeat_fraction = float(np.clip(repeat_fraction, 0.0, 0.8))
    if repeat_fraction <= 0:
        return SynthItem(audio.astype(np.float32), 0.0, "repetition", "clean")

    n = len(audio)
    seg_len = int(n * repeat_fraction)
    if seg_len < SAMPLE_RATE:
        return SynthItem(audio.astype(np.float32), 0.0, "repetition", "too short to repeat")

    start = rng.randint(0, max(n - seg_len - 1, 0))
    seg = audio[start: start + seg_len]

    out = audio[: start + seg_len]
    for _ in range(n_repeats):
        out = _crossfade(out, seg)
    out = _crossfade(out, audio[start + seg_len:])

    label = (seg_len * n_repeats) / max(len(out), 1)
    return SynthItem(out.astype(np.float32), label, "repetition",
                     f"{repeat_fraction:.0%} span repeated {n_repeats}x")


# --------------------------------------------------------------------------- #
# Text-level cases
# --------------------------------------------------------------------------- #

def make_prompt_echo(question: str, target_words: int = 90) -> str:
    """A transcript that restates the prompt to fill time, adding nothing.

    Built by paraphrase-and-repeat rather than literal repetition, because
    literal repetition is trivially caught by n-gram overlap. This is the harder
    positive, and the one `window_sim_vs_ref` exists for.
    """
    q = question.rstrip(" .?!")
    frames = [
        f"So the question is asking me to {q.lower()}",
        f"Um, {q.lower()}, that is what I need to talk about",
        f"Well, about {q.lower()}, yeah",
        f"So basically {q.lower()}, right",
        f"That is the thing, {q.lower()}",
    ]
    out: list[str] = []
    i = 0
    while len(" ".join(out).split()) < target_words:
        out.append(frames[i % len(frames)] + ".")
        i += 1
    return " ".join(out)


def make_text_repetition(text: str, n_repeats: int = 2) -> str:
    """Transcript-level padding: repeat the opening sentences verbatim."""
    sents = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sents) < 2:
        return text
    half = max(len(sents) // 2, 1)
    block = ". ".join(sents[:half]) + "."
    return " ".join([block] * (n_repeats + 1) + [". ".join(sents[half:]) + "."])


def pair_offtopic(responses: dict[str, str], questions: dict[str, str],
                  rng: random.Random | None = None) -> list[tuple[str, str, str]]:
    """Pair each response with a deliberately mismatched question.

    Returns ``(question_id, question_text, response_text)``. These are genuine,
    fluent, well-formed answers -- to a different question -- which is exactly
    the case a similarity-based relevance score gets wrong.
    """
    rng = rng or random.Random(0)
    qids = list(questions)
    out = []
    for rid, text in responses.items():
        wrong = [q for q in qids if q != rid]
        if not wrong:
            continue
        qid = rng.choice(wrong)
        out.append((qid, questions[qid], text))
    return out
