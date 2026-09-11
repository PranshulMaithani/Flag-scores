"""Flag detector tests.

The governing requirement is asymmetric: a flag that fires on a genuine candidate
is far more costly than one that misses a cheat, because it voids a real person's
assessment. So the negative cases below matter more than the positive ones, and
several exist specifically to pin behaviour the client asked for -- notably that
disfluency must never be read as gaming.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxscore.flags.detectors import (
    _audio_self_similarity,
    _lcs_length,
    _longest_repeated_span,
    _max_run,
    _squash,
    foreign_language_flag,
    off_topic_flag,
    prompt_read_flag,
    repetition_flag,
)
from voxscore.utils.textproc import parse


class _Win:
    """Minimal stand-in for LangWindow."""

    def __init__(self, p_en: float, lang: str = "hi", start: float = 0.0):
        self.p_english = p_en
        self.top_lang = lang if p_en < 0.5 else "en"
        self.top_prob = max(p_en, 1 - p_en)
        self.start = start
        self.end = start + 5.0

    @property
    def p_non_english(self):
        return 1.0 - self.p_english


GENUINE = (
    "For my last birthday I kept things simple and had dinner at home with my "
    "parents. My mother cooked biryani, which is my favourite, and we talked for "
    "a long time afterwards. Nothing dramatic happened but I felt genuinely "
    "relaxed by the end of the evening."
)
QUESTION = "Share how you celebrated your most recent birthday."


class TestHelpers:
    def test_squash_clamps_both_ends(self):
        assert _squash(-1, 0, 1) == 0.0
        assert _squash(5, 0, 1) == 1.0
        assert _squash(0.5, 0, 1) == pytest.approx(0.5)

    def test_squash_degenerate_range_is_safe(self):
        assert _squash(5, 1, 1) == 0.0

    def test_lcs(self):
        assert _lcs_length(list("abcde"), list("ace")) == 3
        assert _lcs_length([], ["a"]) == 0

    def test_longest_repeated_span(self):
        toks = "a b c d a b c d e f".split()
        assert _longest_repeated_span(toks, min_len=3) >= 3
        assert _longest_repeated_span("one two three four five".split()) == 0

    def test_max_run(self):
        assert _max_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3
        assert _max_run(np.array([0, 0], bool)) == 0


class TestPromptRead:
    def test_genuine_answer_does_not_fire(self):
        p = parse(GENUINE)
        feats = {"content_novelty_vs_q": 0.95, "window_sim_vs_ref": 0.86,
                 "specificity": 4.0, "distinct_content_rate": 1.0}
        assert prompt_read_flag(p, QUESTION, feats).score < 25

    def test_echo_fires(self):
        echo = ("So the question is asking me to share how I celebrated my most "
                "recent birthday. How I celebrated my most recent birthday, that is "
                "what I need to talk about, my most recent birthday celebration.")
        feats = {"content_novelty_vs_q": 0.55, "window_sim_vs_ref": 1.29,
                 "specificity": 0.0, "distinct_content_rate": 0.5}
        assert prompt_read_flag(parse(echo), QUESTION, feats).score > 55

    def test_keys_on_novelty_not_question_similarity(self):
        """High question similarity alone must not fire the flag.

        A genuine answer is topically similar to the question too. In probing, a
        pure echo scored the highest of all cases on raw question similarity --
        so any detector built on that signal ranks echo and good answers together.
        """
        p = parse(GENUINE)
        high_sim_genuine = {"content_novelty_vs_q": 0.95, "window_sim_vs_ref": 1.0,
                            "specificity": 4.0, "distinct_content_rate": 1.0}
        low_novelty = {"content_novelty_vs_q": 0.45, "window_sim_vs_ref": 1.0,
                       "specificity": 0.0, "distinct_content_rate": 0.5}
        assert prompt_read_flag(p, QUESTION, low_novelty).score > \
               prompt_read_flag(p, QUESTION, high_sim_genuine).score + 20


class TestRepetition:
    def test_genuine_answer_does_not_fire(self):
        assert repetition_flag(parse(GENUINE)).score < 25

    def test_padding_fires(self):
        padded = (GENUINE + " " + GENUINE.split(".")[0] + ". "
                  + GENUINE.split(".")[0] + ".")
        assert repetition_flag(parse(padded)).score > \
               repetition_flag(parse(GENUINE)).score + 15

    def test_disfluency_is_not_treated_as_gaming(self):
        """The client's definition is cheating, not stuttering.

        A nervous candidate repeating words must never be accused of gaming, so
        immediate repeats are stripped before anything is measured.
        """
        stuttered = ("For for my last last birthday I I kept things simple and and "
                     "had dinner at at home with my my parents. My mother mother "
                     "cooked biryani, which is my favourite, and we talked for a "
                     "long time afterwards.")
        fluent = ("For my last birthday I kept things simple and had dinner at home "
                  "with my parents. My mother cooked biryani, which is my favourite, "
                  "and we talked for a long time afterwards.")
        s_stut = repetition_flag(parse(stuttered)).score
        s_flu = repetition_flag(parse(fluent)).score
        assert s_stut < 55, f"stuttering scored {s_stut:.1f} and would be flagged"
        assert s_stut - s_flu < 20

    def test_duplicated_audio_is_detected(self):
        rng = np.random.default_rng(0)
        seg = rng.standard_normal(16000 * 3).astype(np.float32) * 0.1
        looped = np.concatenate([seg, seg])
        varied = rng.standard_normal(16000 * 6).astype(np.float32) * 0.1
        assert _audio_self_similarity(looped) > _audio_self_similarity(varied)

    def test_empty_input_is_safe(self):
        assert repetition_flag(parse("")).score >= 0.0


class TestOffTopic:
    def test_on_topic_does_not_fire(self):
        feats = {"sim_q": 0.70, "element_coverage": 0.78,
                 "pct_windows_offtopic": 0.0, "window_sim_vs_ref": 0.88}
        assert not off_topic_flag(feats).fired

    def test_off_topic_fires(self):
        feats = {"sim_q": 0.38, "element_coverage": 0.41,
                 "pct_windows_offtopic": 1.0, "window_sim_vs_ref": 0.60}
        assert off_topic_flag(feats).fired

    def test_default_threshold_is_lower_than_other_flags(self):
        # Set from a measured ROC: at 55 it caught only half of genuinely
        # mismatched answers.
        assert off_topic_flag({}).threshold == 35.0


class TestForeignLanguage:
    def test_english_throughout_does_not_fire(self):
        wins = [_Win(0.99, start=i * 2.5) for i in range(8)]
        r = foreign_language_flag(wins, GENUINE, asr_avg_logprob=-0.15)
        assert r.score < 20
        assert not r.fired

    def test_sustained_foreign_speech_fires(self):
        wins = ([_Win(0.98, start=0), _Win(0.97, start=2.5)]
                + [_Win(0.03, "hi", start=5 + i * 2.5) for i in range(8)])
        # Devanagari transcript corroborates the acoustic channel.
        r = foreign_language_flag(wins, "मैंने अपना जन्मदिन घर पर मनाया था और यह बहुत अच्छा था",
                                  asr_avg_logprob=-1.1)
        assert r.score > 55

    def test_score_is_monotone_in_foreign_proportion(self):
        def score(frac_foreign: float) -> float:
            n = 10
            k = int(n * frac_foreign)
            wins = ([_Win(0.03, "hi", start=i * 2.5) for i in range(k)]
                    + [_Win(0.98, start=(k + i) * 2.5) for i in range(n - k)])
            return foreign_language_flag(wins, GENUINE, -0.3).score

        scores = [score(f) for f in (0.0, 0.2, 0.5, 0.8)]
        assert scores == sorted(scores), f"not monotone: {scores}"

    def test_isolated_confused_window_does_not_fire(self):
        """The fairness case: one misread window must not flag a real candidate.

        LID models misclassify accented English as the speaker's L1, and this
        population is entirely L2 speakers. A single confused window is the most
        common way that manifests.
        """
        wins = [_Win(0.99, start=i * 2.5) for i in range(9)]
        wins.insert(4, _Win(0.35, "hi", start=10.0))
        r = foreign_language_flag(wins, GENUINE, asr_avg_logprob=-0.2)
        assert not r.fired, f"single confused window scored {r.score:.1f}"

    def test_no_windows_is_safe(self):
        assert foreign_language_flag([], "", 0.0).score >= 0.0


class TestEvidence:
    def test_every_flag_explains_itself(self):
        """Evidence strings are what a contesting candidate is shown."""
        p = parse(GENUINE)
        feats = {"content_novelty_vs_q": 0.95, "window_sim_vs_ref": 0.86,
                 "specificity": 4.0, "distinct_content_rate": 1.0, "sim_q": 0.7,
                 "element_coverage": 0.8, "pct_windows_offtopic": 0.0}
        for f in (prompt_read_flag(p, QUESTION, feats), repetition_flag(p),
                  off_topic_flag(feats), foreign_language_flag([], GENUINE, -0.2)):
            assert f.evidence, f"{f.name} produced no evidence"
            d = f.as_dict()
            assert set(d) >= {"name", "score", "threshold", "fired", "evidence", "features"}
            assert 0.0 <= d["score"] <= 100.0
