"""Fluency features from word-level alignment.

Fluency is the category where the literature is clearest and the incumbent's 0.60
is least impressive: speech rate and pause statistics alone reach roughly that.
The features that go beyond it are the ones describing *how* speech is broken up
rather than how fast it is -- mean length of run, and whether pauses fall at
clause boundaries or inside them.

Two deliberate exclusions, both fairness-driven:

* ``mean_word_align_score`` (acoustic clarity) is computed but **kept out of the
  score**. It correlates with accent, and accent must not be penalised.
* Leading and trailing silence is excluded everywhere. It reflects when the
  recorder started and stopped, not the candidate.
"""

from __future__ import annotations

import numpy as np

from voxscore.asr.align import Pause, Word, extract_pauses, phonation_time
from voxscore.utils.textproc import ParsedText, count_fillers, find_immediate_repeats, ratio

FLUENCY_FEATURES = (
    "speech_rate_wpm", "articulation_rate_wps", "phonation_time_ratio",
    "mean_length_of_run", "silent_pause_rate", "mean_silent_pause_dur",
    "long_pause_rate", "pause_dur_cv", "within_clause_pause_ratio",
    "filled_pause_rate", "disfluency_repeat_rate", "artic_rate_stability",
    "speech_span_s", "n_words",
)


def fluency_features(
    words: list[Word],
    parsed: ParsedText,
    total_duration: float,
    short_pause_s: float = 0.25,
    long_pause_s: float = 1.0,
) -> dict[str, float]:
    """Compute the fluency block. Returns zeros if alignment produced no words.

    An empty result means *unavailable*, not *perfectly fluent*; the quality
    diagnostics tell them apart and the scoring layer must not treat a failed
    alignment as a fast, pause-free delivery.
    """
    out = {k: 0.0 for k in FLUENCY_FEATURES}
    if not words:
        return out

    n_words = len(words)
    out["n_words"] = float(n_words)

    # Speech span excludes recorder lead-in and lead-out. Measuring rate against
    # total duration would penalise whoever had a slow interface, and in smoke
    # testing depressed phonation ratio from ~0.73 to 0.60 on clean read speech.
    span_start, span_end = words[0].start, words[-1].end
    span = max(span_end - span_start, 1e-6)
    out["speech_span_s"] = float(span)

    pauses = extract_pauses(words, total_duration, short_pause_s, trim_edges=True)
    phon = phonation_time(words)

    out["speech_rate_wpm"] = ratio(n_words * 60.0, span)
    out["articulation_rate_wps"] = ratio(n_words, phon)
    out["phonation_time_ratio"] = ratio(phon, span)

    # Mean length of run: words spoken between hesitations. The single most
    # informative fluency feature in the L2 literature, and invisible to rate.
    out["mean_length_of_run"] = ratio(n_words, len(pauses) + 1)

    out["silent_pause_rate"] = ratio(len(pauses) * 60.0, span)
    durs = np.array([p.duration for p in pauses], dtype=np.float32) if pauses else np.zeros(0, np.float32)
    out["mean_silent_pause_dur"] = float(durs.mean()) if durs.size else 0.0
    out["long_pause_rate"] = ratio(float((durs >= long_pause_s).sum()) * 60.0, span)
    out["pause_dur_cv"] = float(durs.std() / durs.mean()) if durs.size and durs.mean() > 0 else 0.0

    out["within_clause_pause_ratio"] = _within_clause_ratio(pauses, words, parsed)

    n_tokens = max(parsed.n_tokens, 1)
    out["filled_pause_rate"] = ratio(count_fillers(parsed.text) * 100.0, n_tokens)
    repeats = find_immediate_repeats(parsed.tokens)
    out["disfluency_repeat_rate"] = ratio(len(repeats) * 100.0, n_tokens)

    out["artic_rate_stability"] = _rate_stability(words, span_start, span_end)
    return out


def _within_clause_ratio(
    pauses: list[Pause],
    words: list[Word],
    parsed: ParsedText,
) -> float:
    """Fraction of pauses that fall *inside* a clause rather than at its boundary.

    A speaker who pauses between clauses sounds planned; one who pauses mid-phrase
    sounds like they are searching for words. Same pause count, very different
    impression -- and this is the distinction a rate-based system cannot make.
    """
    if not pauses or parsed.doc is None:
        return 0.0

    # Token index of each clause boundary, from sentence starts and clausal heads.
    boundaries: set[int] = {0}
    try:
        for sent in parsed.doc.sents:
            boundaries.add(sent.start)
        for tok in parsed.doc:
            if tok.dep_ in {"conj", "advcl", "ccomp", "xcomp", "relcl", "parataxis"}:
                boundaries.add(tok.left_edge.i)
    except Exception:
        return 0.0

    # Aligned words and parsed tokens are different tokenisations; map by rank,
    # which is close enough because alignment preserves word order.
    n_parsed = max(len(parsed.tokens), 1)
    within = 0
    for p in pauses:
        idx = p.before_word_idx
        approx = int(round(idx * n_parsed / max(len(words), 1)))
        if not any(abs(approx - b) <= 1 for b in boundaries):
            within += 1
    return ratio(within, len(pauses))


def _rate_stability(words: list[Word], start: float, end: float, win_s: float = 5.0) -> float:
    """Std of windowed speaking rate. Lower is steadier."""
    span = end - start
    if span < win_s * 1.5:
        return 0.0
    edges = np.arange(start, end, win_s / 2)
    rates = []
    for e in edges:
        n = sum(1 for w in words if e <= w.start < e + win_s)
        rates.append(n / win_s)
    return float(np.std(rates)) if len(rates) > 1 else 0.0
