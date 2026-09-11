"""Tests for question-demand detection.

This is now the primary router for relevance, replacing `shareability`. That
change came from a client constraint: their ideal answers are AI-generated, are
not a gold standard, and the pipeline must not depend on them. The question text
is always present and states the speech act it wants in its own wording.

Because it is load-bearing, the client's real questions are used as the test
cases rather than invented ones.
"""

from __future__ import annotations

import pytest

from voxscore.features.qtype import profile_question

PERSONAL = [
    "Share how you celebrated your most recent birthday.",
    "Talk about a friend you used to be close with but later lost touch.",
    "Describe a game you liked to play as a child.",
    "Talk about a time when you felt nervous before doing something new.",
    "Describe how you spend your evenings on weekends.",
]

OPINION = [
    "In your opinion, does technology improve human thinking or make people more dependent? Explain with reasons.",
    "Is it better to specialise in one skill or learn many different ones? Why do you think so?",
    "Does teamwork always produce better results than working alone? Explain your answer with reasons.",
    "What social pressures do young people face today that previous generations did not?",
]


class TestFamilyDetection:
    @pytest.mark.parametrize("q", PERSONAL)
    def test_personal_prompts_read_as_narrative(self, q):
        p = profile_question(q)
        assert p.narrativity > p.argumentativeness, f"{q!r} -> {p.family}"

    @pytest.mark.parametrize("q", OPINION)
    def test_opinion_prompts_read_as_argumentative(self, q):
        p = profile_question(q)
        assert p.argumentativeness >= p.narrativity, f"{q!r} -> {p.family}"

    def test_auxiliary_initial_question_is_argumentative(self):
        """A yes/no question about the world is a request for a position.

        "Does teamwork always produce better results than working alone?" matched
        none of the opinion phrase patterns despite being unambiguously an
        opinion prompt, which is why the auxiliary-initial rule exists.
        """
        p = profile_question("Does teamwork always produce better results than working alone?")
        assert p.argumentativeness > 0
        assert p.family == "opinion"


class TestDemands:
    def test_explanation_is_detected_when_requested(self):
        assert profile_question(
            "Does teamwork always produce better results? Explain your answer with reasons."
        ).wants_explanation
        assert not profile_question(
            "Describe a game you liked to play as a child."
        ).wants_explanation

    def test_comparison_is_detected(self):
        assert profile_question(
            "Is it better to specialise in one skill or learn many different ones?"
        ).wants_comparison

    def test_focus_terms_exclude_framing_vocabulary(self):
        """Framing verbs appear in every prompt and carry no topical information."""
        p = profile_question("Talk about a friend you used to be close with but later lost touch.")
        assert "friend" in p.focus_terms
        assert "talk" not in p.focus_terms

    def test_focus_terms_capture_the_subject(self):
        p = profile_question(
            "In your opinion, does technology improve human thinking or make people more dependent?"
        )
        assert "technology" in p.focus_terms


class TestRobustness:
    def test_empty_question_does_not_raise(self):
        p = profile_question("")
        assert 0.0 <= p.argumentativeness <= 1.0
        assert 0.0 <= p.narrativity <= 1.0

    def test_unseen_phrasing_degrades_to_neutral_not_to_zero(self):
        """An unrecognised prompt should sit in the middle, not collapse.

        The weighting is continuous, so a neutral profile gives both feature
        blocks partial weight rather than silently disabling one of them.
        """
        p = profile_question("Elaborate upon the municipal drainage infrastructure.")
        assert 0.2 <= p.argumentativeness <= 0.8
        assert 0.2 <= p.narrativity <= 0.8

    def test_profile_is_serialisable(self):
        d = profile_question(PERSONAL[0]).as_dict()
        assert set(d) >= {"argumentativeness", "narrativity", "wants_explanation",
                          "wants_comparison", "focus_terms", "imperative"}
