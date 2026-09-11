"""Turn features into 0-100 category scores.

**These weights are not fitted.** With zero client training data, any fitted head
would be unvalidatable and would overfit whatever public corpus we fitted it on.
So the default scorer is a transparent scorecard: every feature is mapped to
[0,1] through an explicit piecewise-linear reference curve, and the category
score is a documented weighted mean.

That has three properties worth more than a small accuracy gain:

* it cannot overfit, so it is a genuine floor rather than an optimistic estimate;
* every score decomposes into named contributions, satisfying the appeals
  requirement directly;
* the client can re-tune any single curve without retraining anything.

Reference ranges come from the L2 speaking-assessment literature and from the
client's own ideal answers where those pin a value. Each is annotated with its
basis. ``scripts/fit_calibration.py`` replaces this with a fitted model when
labelled data exists; the two are reported side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CATEGORIES = ("grammar", "lexical", "fluency", "relevance")


def _curve(x: float, lo: float, hi: float) -> float:
    """Piecewise-linear map onto [0,1]. ``lo`` scores 0, ``hi`` scores 1.

    **``lo > hi`` is how you express "lower is better".** ``_curve(err, 18, 1)``
    already gives 1.0 at one error and 0.0 at eighteen; there is no invert flag,
    deliberately.

    There used to be one, and all six "lower is better" features passed it *and*
    ordered lo > hi -- double-negating every one. Grammar rewarded more errors,
    fluency rewarded more long pauses and more filled pauses, lexical rewarded
    commoner vocabulary, and relevance rewarded topic drift. Nothing raised, and
    every score stayed in a plausible 0-100 range. Removing the flag makes the
    mistake unexpressible rather than merely fixed.
    """
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _band(x: float, lo: float, best_lo: float, best_hi: float, hi: float) -> float:
    """Plateau curve: full marks inside [best_lo, best_hi], tapering outside.

    Used where more is not monotonically better. Speaking rate is the clear case:
    a candidate racing at 220 wpm is not more fluent than one at 150, and a
    monotone curve would reward rushing.
    """
    if x < best_lo:
        return _curve(x, lo, best_lo)
    if x > best_hi:
        return _curve(x, hi, best_hi)
    return 1.0


@dataclass
class CategoryScore:
    name: str
    score: float                                   # 0-100
    confidence: float                              # 0-1
    contributions: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 3),
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Per-category scorecards
# --------------------------------------------------------------------------- #
# Each entry: feature -> (weight, scorer(value) -> [0,1], rationale)

def _fluency_card(f: dict[str, float]) -> dict[str, tuple[float, float, str]]:
    return {
        "mean_length_of_run": (
            0.25, _curve(f.get("mean_length_of_run", 0), 2.0, 12.0),
            "words between hesitations; the strongest single fluency correlate in "
            "the L2 literature and invisible to speaking rate alone",
        ),
        "speech_rate_wpm": (
            0.20, _band(f.get("speech_rate_wpm", 0), 50, 110, 180, 240),
            "plateau, not monotone: 220 wpm is rushing, not fluency",
        ),
        "phonation_time_ratio": (
            0.15, _curve(f.get("phonation_time_ratio", 0), 0.30, 0.75),
            "share of the speech span actually producing speech",
        ),
        "long_pause_rate": (
            0.15, _curve(f.get("long_pause_rate", 0), 10.0, 0.0),
            "pauses over 1 s per minute; hesitation rather than phrasing",
        ),
        "within_clause_pause_ratio": (
            0.15, _curve(f.get("within_clause_pause_ratio", 0), 0.80, 0.15),
            "pausing mid-clause reads as word-searching; pausing between clauses "
            "reads as planned. Same pause count, different impression",
        ),
        "filled_pause_rate": (
            0.10, _curve(f.get("filled_pause_rate", 0), 12.0, 1.0),
            "um/uh per 100 words; weighted low because the client's own ideal "
            "answers contain filled pauses by design",
        ),
    }


def _grammar_card(f: dict[str, float]) -> dict[str, tuple[float, float, str]]:
    return {
        "errors_per_100_words": (
            0.40, _curve(f.get("errors_per_100_words", 0), 18.0, 1.0),
            "overall accuracy from GEC edits",
        ),
        "error_free_sentence_ratio": (
            0.25, _curve(f.get("error_free_sentence_ratio", 0), 0.20, 0.95),
            "share of sentences the corrector left untouched",
        ),
        "clauses_per_sentence": (
            0.20, _curve(f.get("clauses_per_sentence", 0), 1.0, 2.6),
            "complexity, so accuracy through risk-avoidance is not rewarded: "
            "short simple clauses are easy to get right",
        ),
        "mean_dependency_distance": (
            0.15, _curve(f.get("mean_dependency_distance", 0), 1.8, 4.0),
            "structural range; longer dependencies indicate more embedded syntax",
        ),
    }


def _lexical_card(f: dict[str, float]) -> dict[str, tuple[float, float, str]]:
    return {
        "mtld": (
            0.30, _curve(f.get("mtld", 0), 18.0, 80.0),
            "length-robust diversity; chosen over TTR, which is a length artefact",
        ),
        "mean_log_freq": (
            0.25, _curve(f.get("mean_log_freq", 7.0), 5.4, 3.9),
            "mean Zipf frequency of content lemmas; rarer vocabulary scores higher",
        ),
        "pct_beyond_zipf4": (
            0.20, _curve(f.get("pct_beyond_zipf4", 0), 0.05, 0.40),
            "share of content words outside the common band",
        ),
        "lexical_density": (
            0.15, _curve(f.get("lexical_density", 0), 0.32, 0.60),
            "content words per token; low density means filler-heavy speech",
        ),
        "hdd": (
            0.10, _curve(f.get("hdd", 0), 0.60, 0.90),
            "hypergeometric diversity; stabilises MTLD on short responses",
        ),
    }


def _relevance_anchor_card(f: dict[str, float]) -> dict[str, tuple[float, float, str]]:
    """Is this response about the question at all?"""
    share = f.get("shareability", 0.0)
    # Content matching is trusted in proportion to how much the ideal answers
    # agree with each other (ADR-007). Blended continuously, never routed.
    w_content = float(np.clip((share - 0.05) / 0.10, 0.0, 1.0)) * 0.25

    card = {
        "element_coverage": (
            0.40 - w_content * 0.5, _curve(f.get("element_coverage", 0), 0.35, 0.80),
            "coverage of the elements the question itself demands; derived from "
            "the prompt alone, so it works with no ideal answers at all",
        ),
        "sim_q": (
            0.25, _curve(f.get("sim_q", 0), 0.35, 0.72),
            "overall topical match to the question",
        ),
        # Weight collapses when the response was too short to measure drift, so
        # an unmeasurable response abstains instead of scoring a free pass.
        "pct_windows_offtopic": (
            (0.35 - w_content * 0.5) * f.get("drift_available", 1.0),
            _curve(f.get("pct_windows_offtopic", 0), 0.7, 0.0),
            "share of the response drifting off topic, thresholded per question "
            "against the ideal answers rather than absolutely",
        ),
    }
    if w_content > 0:
        card["move_coverage"] = (
            w_content, _curve(f.get("move_coverage", 0), 0.45, 0.85),
            f"coverage of content shared by >=2 ideal answers, weighted at "
            f"{w_content:.2f} because measured shareability is {share:.3f}",
        )
    return card


def _relevance_engagement_card(f: dict[str, float]) -> dict[str, tuple[float, float, str]]:
    """Given that it is on topic, is it a substantive answer?"""
    return {
        "specificity": (
            0.35, _curve(f.get("specificity", 0), 0.3, 5.0),
            "concrete detail present. In probing this was the cleanest separator "
            "of genuine answers from vague or echoed ones (2.9-5.5 vs 0.0)",
        ),
        "content_novelty_vs_q": (
            0.25, _curve(f.get("content_novelty_vs_q", 0), 0.55, 0.92),
            "content beyond the question's own words; the defence against an "
            "answer that scores well by restating the prompt",
        ),
        # Weight collapses to zero when no ideal answers exist for this question,
        # so its absence is redistributed across the other features rather than
        # silently subtracting a fixed amount from every response to that
        # question. _weighted() renormalises by the surviving weights.
        "profile_match": (
            0.25 if f.get("profile_match", 0.0) > 0 else 0.0,
            _curve(f.get("profile_match", 0), 0.05, 0.30),
            "structural fit to the ideal answers as a distribution -- did the "
            "candidate perform the speech act the prompt demanded",
        ),
        "distinct_content_rate": (
            0.15, _curve(f.get("distinct_content_rate", 0), 0.55, 0.95),
            "non-repetition of content; padding scores low",
        ),
    }


def _relevance_card(f: dict[str, float]) -> dict[str, tuple[float, float, str]]:
    """Combined view, used only for explanation display."""
    return {**_relevance_anchor_card(f), **_relevance_engagement_card(f)}


_CARDS = {
    "fluency": _fluency_card,
    "grammar": _grammar_card,
    "lexical": _lexical_card,
    "relevance": _relevance_card,
}


def _weighted(card: dict[str, tuple[float, float, str]]) -> tuple[float, dict[str, float]]:
    total_w = sum(max(w, 0.0) for w, _, _ in card.values())
    if total_w <= 0:
        return 0.0, {}
    contribs = {k: (max(w, 0.0) / total_w) * v for k, (w, v, _) in card.items()}
    return sum(contribs.values()), contribs


def score_category(name: str, feats: dict[str, float], confidence: float = 1.0) -> CategoryScore:
    """Score one category from its feature block."""
    if name == "relevance":
        return score_relevance(feats, confidence)

    card = _CARDS[name](feats)
    value, contributions = _weighted(card)
    if not contributions:
        return CategoryScore(name, 0.0, 0.0)
    return CategoryScore(name, 100.0 * value, confidence, contributions)


# Floor of the engagement multiplier. A perfectly on-topic but contentless answer
# still scores something, because it did address the question.
ENGAGEMENT_FLOOR = 0.35


def score_relevance(feats: dict[str, float], confidence: float = 1.0) -> CategoryScore:
    """Relevance as **anchor x engagement**, not a flat weighted mean.

    Found by the end-to-end smoke test: a response of concatenated LibriSpeech --
    about Victorian art criticism, paired with a question about the candidate's
    birthday -- scored `off_topic` 98.2 while relevance came out at **73.3**.

    The cause is that `specificity` and `content_novelty_vs_q` are *topic-blind*.
    They were added to catch vague and echoed answers, and they do, but a wholly
    off-topic passage is dense with specific, novel content -- it is simply about
    the wrong thing. Averaging them with the topic features let off-topic content
    buy back relevance points.

    Multiplying fixes it in the right direction for every case:

    * off-topic but specific  -> anchor near 0, so the product is near 0
    * on-topic but vague/echo -> engagement near the floor, so heavily discounted
    * on-topic and substantive -> both high

    An additive model cannot express "this is disqualifying", and relevance has
    exactly one disqualifying condition: not being about the question.
    """
    anchor_v, anchor_c = _weighted(_relevance_anchor_card(feats))
    engage_v, engage_c = _weighted(_relevance_engagement_card(feats))
    if not anchor_c:
        return CategoryScore("relevance", 0.0, 0.0)

    multiplier = ENGAGEMENT_FLOOR + (1.0 - ENGAGEMENT_FLOOR) * engage_v
    score = 100.0 * anchor_v * multiplier

    # Report contributions on the same scale they actually influenced the score.
    contributions = {k: v * multiplier for k, v in anchor_c.items()}
    contributions.update({k: v * anchor_v * (1.0 - ENGAGEMENT_FLOOR) for k, v in engage_c.items()})

    cs = CategoryScore("relevance", score, confidence, contributions)
    cs.notes.append(
        f"anchor={anchor_v:.3f} x engagement_multiplier={multiplier:.3f}"
    )
    return cs


def score_all(
    grammar_f: dict[str, float],
    lexical_f: dict[str, float],
    fluency_f: dict[str, float],
    relevance_f: dict[str, float],
    quality_f: dict[str, float],
) -> dict[str, CategoryScore]:
    """Score every category, gated by recording quality."""
    scorable = bool(quality_f.get("scorable", 1.0))
    qconf = float(quality_f.get("quality_confidence", 1.0))

    out = {
        "grammar": score_category("grammar", grammar_f,
                                  qconf * grammar_f.get("grammar_confidence", 1.0)),
        "lexical": score_category("lexical", lexical_f,
                                  qconf * lexical_f.get("lexical_confidence", 1.0)),
        "fluency": score_category("fluency", fluency_f, qconf),
        "relevance": score_category("relevance", relevance_f, qconf),
    }

    if not scorable:
        # Whisper hallucinates fluent text on silence and noise. Emitting a
        # score here would be inventing a measurement from no evidence.
        for cs in out.values():
            cs.score = 0.0
            cs.confidence = 0.0
            cs.notes.append("not scorable: see quality diagnostics")
    return out


def explain(cs: CategoryScore, feats: dict[str, float], top_k: int = 3) -> str:
    """One-sentence explanation naming the largest contributors.

    This is what a candidate contesting a score is shown.
    """
    card = _CARDS[cs.name](feats)
    ranked = sorted(cs.contributions.items(), key=lambda kv: -kv[1])
    parts = []
    for k, _ in ranked[:top_k]:
        _, v, why = card[k]
        verdict = "strong" if v > 0.66 else ("adequate" if v > 0.33 else "weak")
        parts.append(f"{k.replace('_', ' ')} {verdict} ({v:.2f})")
    return f"{cs.name} {cs.score:.0f}/100 - " + "; ".join(parts)
