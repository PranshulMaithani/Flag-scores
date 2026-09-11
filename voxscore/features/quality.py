"""Recording-quality diagnostics, and the scorability gate.

These are not a nicety. Whisper **hallucinates on non-speech**: measured on this
project, white noise transcribes as "Thank you." and pure silence as "you".
Without a gate, a candidate who says nothing still produces a scorable transcript
and receives grammar and lexical scores. Any vendor scoring such items is
generating numbers from nothing, and the client cannot currently tell.

The gate exists so the client can separate *"bad answer"* from *"bad recording"*,
which is a differentiator as well as a correctness requirement.
"""

from __future__ import annotations

import numpy as np

from voxscore.config import SAMPLE_RATE
from voxscore.utils.textproc import ratio

QUALITY_FEATURES = (
    "duration_s", "snr_db", "clipping_rate", "speech_presence_ratio",
    "rms_db", "silence_ratio", "asr_avg_logprob", "word_count",
    "words_per_second", "scorable", "quality_confidence",
)

# Below this many words, nothing downstream is trustworthy: diversity metrics are
# noise, grammar has too few clauses, relevance has nothing to cover.
MIN_SCORABLE_WORDS = 15
MIN_SCORABLE_DURATION = 3.0


def quality_features(
    audio: np.ndarray,
    transcript: str,
    asr_avg_logprob: float,
    sr: int = SAMPLE_RATE,
) -> dict[str, float]:
    """Recording diagnostics plus a scorability decision."""
    out = {k: 0.0 for k in QUALITY_FEATURES}
    audio = np.asarray(audio, dtype=np.float32)

    dur = len(audio) / sr
    out["duration_s"] = float(dur)
    if audio.size == 0:
        return out

    abs_a = np.abs(audio)
    out["clipping_rate"] = float((abs_a >= 0.999).mean())

    rms = float(np.sqrt(np.mean(audio ** 2)))
    out["rms_db"] = float(20 * np.log10(max(rms, 1e-9)))

    # Frame energies split into speech / noise by an adaptive percentile split,
    # rather than an absolute threshold that would be wrong for any recording
    # level other than the one it was tuned on.
    frame = int(0.02 * sr)
    n_frames = len(audio) // frame
    if n_frames >= 5:
        frames = audio[: n_frames * frame].reshape(n_frames, frame)
        energies = np.sqrt((frames ** 2).mean(axis=1))
        e_db = 20 * np.log10(np.maximum(energies, 1e-9))

        noise_floor = float(np.percentile(e_db, 10))
        speech_level = float(np.percentile(e_db, 90))
        out["snr_db"] = speech_level - noise_floor

        gate = noise_floor + 0.35 * (speech_level - noise_floor)
        active = e_db > gate
        out["speech_presence_ratio"] = float(active.mean())
        out["silence_ratio"] = float(1.0 - active.mean())

    words = (transcript or "").split()
    out["word_count"] = float(len(words))
    out["words_per_second"] = ratio(len(words), dur)
    out["asr_avg_logprob"] = float(asr_avg_logprob)

    scorable, confidence = _scorability(out, transcript)
    out["scorable"] = 1.0 if scorable else 0.0
    out["quality_confidence"] = confidence
    return out


# Whisper's characteristic outputs on non-speech. Matching these exactly, and
# only when the transcript is *nothing but* one of them, avoids penalising a
# candidate whose genuine answer happens to open with "thank you".
_HALLUCINATION_TRANSCRIPTS = {
    "you", "thank you", "thank you.", "thanks for watching", "bye",
    "thank you for watching", "subscribe", ".", "...", "。",
}


def _scorability(q: dict[str, float], transcript: str) -> tuple[bool, float]:
    """Decide whether this item can be meaningfully scored, and how confidently."""
    text = (transcript or "").strip().lower()

    if text in _HALLUCINATION_TRANSCRIPTS:
        return False, 0.0
    if q["word_count"] < MIN_SCORABLE_WORDS:
        return False, 0.0
    if q["duration_s"] < MIN_SCORABLE_DURATION:
        return False, 0.0

    # Confidence degrades smoothly rather than cliff-edging, so the scoring layer
    # can down-weight marginal items instead of discarding them.
    parts = [
        min(q["word_count"] / 60.0, 1.0),
        min(max(q["snr_db"], 0.0) / 25.0, 1.0),
        min(max(q["asr_avg_logprob"] + 1.2, 0.0) / 1.0, 1.0),
        1.0 - min(q["clipping_rate"] * 20, 1.0),
        min(q["speech_presence_ratio"] / 0.4, 1.0),
    ]
    return True, float(np.mean(parts))


def quality_warnings(q: dict[str, float], transcript: str) -> list[str]:
    """Human-readable problems with the recording, for the output XML."""
    w: list[str] = []
    if not q.get("scorable", 0.0):
        if (transcript or "").strip().lower() in _HALLUCINATION_TRANSCRIPTS:
            w.append("transcript matches a known ASR hallucination on non-speech; "
                     "the recording probably contains no answer")
        elif q.get("word_count", 0) < MIN_SCORABLE_WORDS:
            w.append(f"only {int(q.get('word_count', 0))} words transcribed "
                     f"(minimum {MIN_SCORABLE_WORDS} to score)")
        elif q.get("duration_s", 0) < MIN_SCORABLE_DURATION:
            w.append(f"recording is only {q.get('duration_s', 0):.1f}s")
    if q.get("snr_db", 99) < 10:
        w.append(f"low signal-to-noise ratio ({q.get('snr_db', 0):.1f} dB)")
    if q.get("clipping_rate", 0) > 0.01:
        w.append(f"{q['clipping_rate'] * 100:.1f}% of samples clipped")
    if q.get("speech_presence_ratio", 1) < 0.25:
        w.append("mostly silence")
    if q.get("asr_avg_logprob", 0) < -0.9:
        w.append("ASR confidence very low; transcript may be unreliable")
    return w
