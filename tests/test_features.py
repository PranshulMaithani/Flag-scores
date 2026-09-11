"""Tests for the fluency, lexical and quality feature blocks.

Emphasis on the two failure modes that would silently corrupt scores rather than
crash: a missing measurement being reported as a good one, and a length artefact
masquerading as a vocabulary measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxscore.asr.align import Word
from voxscore.features.fluency import fluency_features
from voxscore.features.lexical import _hdd, _mattr, _mtld, lexical_features
from voxscore.features.quality import (
    MIN_SCORABLE_WORDS,
    quality_features,
    quality_warnings,
)
from voxscore.utils.textproc import parse

SR = 16000


def _words(spans, texts=None):
    texts = texts or [f"w{i}" for i in range(len(spans))]
    return [Word(t, s, e, 0.9) for t, (s, e) in zip(texts, spans)]


def _steady(n=40, word_s=0.3, gap_s=0.05, start=0.5):
    spans, t = [], start
    for _ in range(n):
        spans.append((t, t + word_s))
        t += word_s + gap_s
    return spans


class TestFluency:
    def test_no_words_returns_zeros_not_perfection(self):
        """A failed alignment must not read as flawless delivery.

        This is the dangerous direction: zero pauses and zero fillers look like
        an excellent speaker unless the caller knows the measurement failed.
        """
        f = fluency_features([], parse(""), 30.0)
        assert f["speech_rate_wpm"] == 0.0
        assert f["mean_length_of_run"] == 0.0
        assert f["n_words"] == 0.0

    def test_speech_span_excludes_recorder_silence(self):
        """Rate is measured over the speech span, not the file length.

        Leading/trailing silence is when the recorder started and stopped. On
        clean read speech, counting it depressed phonation ratio from ~0.73 to
        0.60 - a candidate penalised for a slow interface.
        """
        spans = _steady(n=30, start=5.0)  # 5 s of dead air at the front
        w = _words(spans)
        f = fluency_features(w, parse(" ".join(f"w{i}" for i in range(30))), 60.0)
        # 30 words over ~10.5 s of speech is fast; over 60 s it would look slow.
        assert f["speech_rate_wpm"] > 100
        assert f["speech_span_s"] < 15

    def test_pauses_reduce_mean_length_of_run(self):
        fluent = _words(_steady(n=20))
        halting = _words([
            (0.5, 0.8), (0.9, 1.2), (2.5, 2.8), (2.9, 3.2), (5.0, 5.3),
            (5.4, 5.7), (7.5, 7.8), (7.9, 8.2),
        ])
        txt = parse(" ".join(f"w{i}" for i in range(20)))
        f_fluent = fluency_features(fluent, txt, 12.0)
        f_halt = fluency_features(halting, parse("a b c d e f g h"), 10.0)
        assert f_fluent["mean_length_of_run"] > f_halt["mean_length_of_run"]
        assert f_halt["silent_pause_rate"] > f_fluent["silent_pause_rate"]

    def test_filled_pauses_counted(self):
        text = "um I went to the uh shop and er bought some milk"
        f = fluency_features(_words(_steady(n=11)), parse(text), 10.0)
        assert f["filled_pause_rate"] > 0

    def test_long_pause_rate_separates_hesitation_from_phrasing(self):
        short_gaps = _words([(0.0, 0.4), (0.7, 1.1), (1.4, 1.8)])
        long_gaps = _words([(0.0, 0.4), (3.0, 3.4), (6.0, 6.4)])
        txt = parse("a b c")
        assert (fluency_features(long_gaps, txt, 7.0)["long_pause_rate"]
                > fluency_features(short_gaps, txt, 7.0)["long_pause_rate"])


class TestLexical:
    def test_empty_is_safe(self):
        assert lexical_features(parse(""))["mtld"] == 0.0

    def test_mtld_is_length_robust_unlike_ttr(self):
        """The reason MTLD is primary and raw TTR is excluded.

        TTR falls mechanically as text lengthens, so it would rank a short
        response above a long one of identical vocabulary richness. Client
        responses vary from ~40 to ~150 words, so that artefact would be a large
        share of the lexical score.
        """
        vocab = [f"word{i}" for i in range(60)]
        short = " ".join(vocab[:30])
        long = " ".join(vocab * 3)

        ttr_short = len(set(short.split())) / len(short.split())
        ttr_long = len(set(long.split())) / len(long.split())
        assert ttr_short > ttr_long * 1.5, "TTR should be strongly length-biased"

        m_short, m_long = _mtld(short.split()), _mtld(long.split())
        assert m_long > m_short * 0.5, "MTLD should not collapse with length"

    def test_diversity_metrics_rank_repetition_below_variety(self):
        varied = [f"w{i}" for i in range(60)]
        repeated = ["cat", "dog"] * 30
        assert _mtld(varied) > _mtld(repeated)
        assert _mattr(varied) > _mattr(repeated)
        assert _hdd(varied) > _hdd(repeated)

    def test_rare_vocabulary_scores_as_more_sophisticated(self):
        common = parse("I went to the shop and got some food for my house today")
        rare = parse("I ventured to the emporium and procured provisions for my "
                     "residence subsequently")
        # Lower mean Zipf frequency means rarer vocabulary.
        assert lexical_features(rare)["mean_log_freq"] < \
               lexical_features(common)["mean_log_freq"]

    def test_confidence_degrades_on_short_responses(self):
        # Purely alphabetic: lexical_features drops tokens containing digits, so
        # "w0 w1 w2" would be filtered out entirely and read as an empty response.
        alpha = [chr(97 + i // 26) + chr(97 + i % 26) for i in range(150)]
        short = parse(" ".join(["word"] * 20))
        long = parse(" ".join(alpha))
        assert lexical_features(short)["lexical_confidence"] < 1.0
        assert lexical_features(long)["lexical_confidence"] == 1.0


class TestQuality:
    def _speech(self, dur=10.0, level=0.2):
        t = np.arange(int(dur * SR)) / SR
        env = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)  # crude on/off
        return (level * env * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def test_silence_is_not_scorable(self):
        q = quality_features(np.zeros(SR * 10, np.float32), "you", -0.9)
        assert q["scorable"] == 0.0

    def test_whisper_hallucination_is_caught(self):
        """Whisper emits 'Thank you.' on noise and 'you' on silence.

        Measured on this project. Without this gate a candidate who said nothing
        receives grammar and lexical scores computed from an invented transcript.
        """
        rng = np.random.default_rng(0)
        noise = (0.1 * rng.standard_normal(SR * 10)).astype(np.float32)
        for text in ("Thank you.", "you", "Thanks for watching"):
            q = quality_features(noise, text, -0.87)
            assert q["scorable"] == 0.0, f"{text!r} was accepted as an answer"
            assert any("hallucination" in w for w in quality_warnings(q, text))

    def test_genuine_short_answer_below_word_floor_is_rejected(self):
        q = quality_features(self._speech(), "I went home", -0.2)
        assert q["word_count"] < MIN_SCORABLE_WORDS
        assert q["scorable"] == 0.0

    def test_normal_response_is_scorable(self):
        text = " ".join(["word"] * 80)
        q = quality_features(self._speech(dur=30), text, -0.2)
        assert q["scorable"] == 1.0
        assert q["quality_confidence"] > 0.3

    def test_clipping_is_detected(self):
        a = self._speech()
        a[: SR // 2] = 1.0
        assert quality_features(a, " ".join(["w"] * 50), -0.2)["clipping_rate"] > 0.01

    def test_snr_separates_clean_from_noisy(self):
        clean = self._speech()
        rng = np.random.default_rng(1)
        noisy = clean + 0.08 * rng.standard_normal(len(clean)).astype(np.float32)
        text = " ".join(["w"] * 50)
        assert (quality_features(clean, text, -0.2)["snr_db"]
                > quality_features(noisy, text, -0.5)["snr_db"])

    def test_warnings_are_human_readable(self):
        q = quality_features(np.zeros(SR * 2, np.float32), "", 0.0)
        w = quality_warnings(q, "")
        assert w and all(isinstance(x, str) and len(x) > 10 for x in w)
