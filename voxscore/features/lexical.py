"""Lexical features -- what the client calls the "vocab score".

Two things dominate the design:

**Length.** Responses cap at 60 s, so 100-150 words. Most diversity metrics are
unstable below ~100 tokens, and raw type-token ratio is a pure length artefact
rather than a vocabulary measure -- it is excluded outright. MTLD is the primary
metric precisely because it is length-robust. A ``lexical_confidence`` value is
emitted so the scoring layer can discount short responses instead of pretending
a 30-word answer supports a reliable vocabulary judgement.

**Licensing.** Sophistication is measured with ``wordfreq`` frequency bands
rather than a published CEFR or academic word list. The obvious choices there
(English Vocabulary Profile, CEFR-J, the Academic Word List) carry research or
unclear licences, and constraint C5 requires everything shipped to be
commercially clean. Frequency banding is licence-safe and measures much the same
construct: rarer vocabulary is more sophisticated vocabulary.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from voxscore.utils.textproc import ParsedText, ratio

LEXICAL_FEATURES = (
    "mtld", "mattr_50", "hdd", "mean_log_freq", "rare_word_rate",
    "pct_beyond_zipf5", "pct_beyond_zipf4", "pct_beyond_zipf3",
    "lexical_density", "content_types", "noun_sophistication",
    "verb_sophistication", "adj_sophistication", "lexical_confidence",
    "n_tokens",
)

# Below this many tokens, diversity metrics are noise rather than measurement.
MIN_RELIABLE_TOKENS = 100


@functools.lru_cache(maxsize=200_000)
def _zipf(word: str) -> float:
    """Zipf frequency: ~7 for 'the', ~1-2 for rare vocabulary, 0 if unknown."""
    try:
        from wordfreq import zipf_frequency

        return float(zipf_frequency(word, "en"))
    except Exception:
        return 0.0


def lexical_features(parsed: ParsedText) -> dict[str, float]:
    """Compute the lexical block."""
    out = {k: 0.0 for k in LEXICAL_FEATURES}

    words = [t.lower() for t in parsed.tokens if t.isalpha()]
    if not words:
        return out

    out["n_tokens"] = float(len(words))
    out["lexical_confidence"] = float(min(len(words) / MIN_RELIABLE_TOKENS, 1.0))

    out["mtld"] = _mtld(words)
    out["mattr_50"] = _mattr(words, window=50)
    out["hdd"] = _hdd(words)

    content = [l for l in parsed.content_lemmas if l.isalpha()]
    out["lexical_density"] = ratio(len(content), len(words))
    out["content_types"] = float(len(set(content)))

    if content:
        zipfs = np.array([_zipf(w) for w in content], dtype=np.float32)
        known = zipfs[zipfs > 0]
        if known.size:
            # Lower mean frequency = rarer vocabulary = more sophisticated.
            out["mean_log_freq"] = float(known.mean())
            out["pct_beyond_zipf5"] = float((known < 5.0).mean())
            out["pct_beyond_zipf4"] = float((known < 4.0).mean())
            out["pct_beyond_zipf3"] = float((known < 3.0).mean())
            out["rare_word_rate"] = float((known < 3.5).sum() / len(words) * 100)

    out["noun_sophistication"] = _pos_sophistication(parsed, {"NOUN", "PROPN"})
    out["verb_sophistication"] = _pos_sophistication(parsed, {"VERB"})
    out["adj_sophistication"] = _pos_sophistication(parsed, {"ADJ"})
    return out


def _pos_sophistication(parsed: ParsedText, tags: set[str]) -> float:
    """Mean rarity of the lemmas carrying a given part of speech.

    Split by POS because a candidate may have rich nouns but a thin verb
    repertoire, which is a genuinely different profile and one a single
    aggregate number hides.
    """
    lemmas = [
        l for l, p in zip(parsed.lemmas, parsed.pos)
        if p in tags and l.isalpha()
    ]
    if not lemmas:
        return 0.0
    z = [_zipf(l) for l in lemmas]
    z = [v for v in z if v > 0]
    return float(np.mean(z)) if z else 0.0


# --------------------------------------------------------------------------- #
# Diversity metrics
# --------------------------------------------------------------------------- #

def _mtld(words: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity, bidirectional (McCarthy & Jarvis).

    Counts how many words it takes for the running type-token ratio to fall to
    ``threshold``, then averages the factor length. Unlike TTR this does not
    shrink mechanically with length, which is what makes it usable on 60-second
    responses.
    """
    if len(words) < 10:
        return 0.0

    def one_pass(seq: list[str]) -> float:
        factors, types, tokens = 0.0, set(), 0
        for w in seq:
            tokens += 1
            types.add(w)
            if tokens and len(types) / tokens <= threshold:
                factors += 1
                types, tokens = set(), 0
        if tokens:
            ttr = len(types) / tokens
            # Partial factor for the trailing segment.
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        return len(seq) / factors if factors > 0 else float(len(seq))

    return float((one_pass(words) + one_pass(words[::-1])) / 2)


def _mattr(words: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio. Length-controlled by construction."""
    if len(words) < window:
        return ratio(len(set(words)), len(words))
    ratios = [
        len(set(words[i: i + window])) / window
        for i in range(len(words) - window + 1)
    ]
    return float(np.mean(ratios))


def _hdd(words: list[str], sample: int = 42) -> float:
    """HD-D: expected type count in a random sample, via the hypergeometric.

    More stable than MTLD on very short texts, so the two together cover the
    length range we actually see.
    """
    n = len(words)
    if n < sample:
        return ratio(len(set(words)), n)

    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    total = 0.0
    for c in counts.values():
        # P(at least one token of this type appears in the sample)
        if n - c < sample:
            total += 1.0
        else:
            log_p = (
                math.lgamma(n - c + 1) - math.lgamma(n - c - sample + 1)
                - math.lgamma(n + 1) + math.lgamma(n - sample + 1)
            )
            total += 1.0 - math.exp(log_p)
    return float(total / sample)
