"""Scoring-layer tests, with monotonicity as the headline.

A scoring bug does not crash. It produces numbers in a plausible range that are
quietly backwards, and it survives every smoke test that only checks "did it
run". One did survive here: all six "lower is better" features were
double-negated, so grammar rewarded errors and fluency rewarded filled pauses.

The direction tests below assert the sign of every feature's effect on every
category, which is the specific thing that failed.
"""

from __future__ import annotations

import pytest

from voxscore.scoring.aggregate import (
    ENGAGEMENT_FLOOR,
    _band,
    _curve,
    score_all,
    score_category,
    score_relevance,
)

# feature, category, better_value, worse_value
DIRECTIONS = [
    # lower is better
    ("errors_per_100_words", "grammar", 0.0, 15.0),
    ("filled_pause_rate", "fluency", 0.0, 12.0),
    ("long_pause_rate", "fluency", 0.0, 10.0),
    ("within_clause_pause_ratio", "fluency", 0.1, 0.9),
    ("mean_log_freq", "lexical", 3.9, 5.6),
    # higher is better
    ("error_free_sentence_ratio", "grammar", 0.95, 0.1),
    ("clauses_per_sentence", "grammar", 2.6, 1.0),
    ("mean_length_of_run", "fluency", 12.0, 2.0),
    ("phonation_time_ratio", "fluency", 0.75, 0.2),
    ("mtld", "lexical", 80.0, 15.0),
    ("hdd", "lexical", 0.9, 0.5),
    ("lexical_density", "lexical", 0.6, 0.3),
    ("pct_beyond_zipf4", "lexical", 0.4, 0.0),
]


class TestScoreDirections:
    @pytest.mark.parametrize("feature,category,better,worse", DIRECTIONS)
    def test_feature_moves_score_the_right_way(self, feature, category, better, worse):
        s_better = score_category(category, {feature: better}).score
        s_worse = score_category(category, {feature: worse}).score
        assert s_better > s_worse, (
            f"{feature} in {category}: value {better} scored {s_better:.1f} but "
            f"{worse} scored {s_worse:.1f} -- the curve is backwards"
        )

    def test_relevance_penalises_off_topic_drift(self):
        base = {"sim_q": 0.7, "element_coverage": 0.8, "specificity": 4.0,
                "content_novelty_vs_q": 0.9, "profile_match": 0.25,
                "distinct_content_rate": 0.95}
        on = score_relevance({**base, "pct_windows_offtopic": 0.0}).score
        off = score_relevance({**base, "pct_windows_offtopic": 1.0}).score
        assert on > off


class TestCurves:
    def test_curve_clamps(self):
        assert _curve(-5, 0, 10) == 0.0
        assert _curve(50, 0, 10) == 1.0

    def test_descending_bounds_mean_lower_is_better(self):
        # This ordering IS the inversion; there is no flag.
        assert _curve(1.0, 18.0, 1.0) == pytest.approx(1.0)
        assert _curve(18.0, 18.0, 1.0) == pytest.approx(0.0)

    def test_band_plateaus_and_penalises_both_sides(self):
        assert _band(150, 50, 110, 180, 240) == 1.0
        assert _band(110, 50, 110, 180, 240) == 1.0
        assert _band(80, 50, 110, 180, 240) < 1.0
        # Rushing must not beat a comfortable rate.
        assert _band(230, 50, 110, 180, 240) < _band(150, 50, 110, 180, 240)

    def test_equal_bounds_do_not_divide_by_zero(self):
        assert _curve(5, 3, 3) == 0.0


class TestRelevanceIsMultiplicative:
    """Off-topic content must not buy relevance with specificity.

    The failing case: concatenated LibriSpeech (dense, specific, novel, and
    entirely unrelated to the question) scored 73.3 under an additive model.
    """

    ONTOPIC = {"sim_q": 0.70, "element_coverage": 0.78, "pct_windows_offtopic": 0.0}
    OFFTOPIC = {"sim_q": 0.38, "element_coverage": 0.41, "pct_windows_offtopic": 1.0}
    RICH = {"specificity": 6.0, "content_novelty_vs_q": 1.0,
            "profile_match": 0.38, "distinct_content_rate": 0.85}
    EMPTY = {"specificity": 0.0, "content_novelty_vs_q": 0.6,
             "profile_match": 0.02, "distinct_content_rate": 0.5}

    def test_specific_but_off_topic_scores_low(self):
        s = score_relevance({**self.OFFTOPIC, **self.RICH}).score
        assert s < 45, f"off-topic but content-rich scored {s:.1f}"

    def test_on_topic_and_substantive_scores_well(self):
        s = score_relevance({**self.ONTOPIC, **self.RICH}).score
        assert s > 60, f"good answer scored only {s:.1f}"

    def test_on_topic_but_empty_is_discounted(self):
        good = score_relevance({**self.ONTOPIC, **self.RICH}).score
        empty = score_relevance({**self.ONTOPIC, **self.EMPTY}).score
        assert empty < good * 0.8

    def test_off_topic_always_below_on_topic_regardless_of_engagement(self):
        best_off = score_relevance({**self.OFFTOPIC, **self.RICH}).score
        worst_on = score_relevance({**self.ONTOPIC, **self.EMPTY}).score
        assert best_off < worst_on, (
            "the best off-topic answer must not outscore the worst on-topic one"
        )

    def test_engagement_floor_is_respected(self):
        zero_engagement = score_relevance({**self.ONTOPIC, "specificity": 0.0,
                                           "content_novelty_vs_q": 0.0,
                                           "profile_match": 0.0,
                                           "distinct_content_rate": 0.0})
        assert zero_engagement.score > 0
        assert "anchor=" in zero_engagement.notes[0]


class TestScorabilityGate:
    def test_unscorable_zeroes_every_category(self):
        feats = {"errors_per_100_words": 1.0, "mtld": 70.0,
                 "mean_length_of_run": 10.0, "sim_q": 0.7}
        scores = score_all(feats, feats, feats, feats,
                           {"scorable": 0.0, "quality_confidence": 0.0})
        assert all(cs.score == 0.0 for cs in scores.values())
        assert all(cs.confidence == 0.0 for cs in scores.values())
        assert all("not scorable" in " ".join(cs.notes) for cs in scores.values())

    def test_scorable_items_keep_their_scores(self):
        feats = {"errors_per_100_words": 1.0, "mtld": 70.0, "hdd": 0.85,
                 "mean_length_of_run": 10.0, "sim_q": 0.7,
                 "element_coverage": 0.8, "specificity": 4.0}
        scores = score_all(feats, feats, feats, feats,
                           {"scorable": 1.0, "quality_confidence": 0.9})
        assert all(cs.score > 0 for cs in scores.values())
