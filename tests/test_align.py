"""Tests for the CTC forced aligner.

The aligner is ours rather than torchaudio's (no CUDA kernel there, and the op is
deprecated for removal). That makes correctness our problem, so these tests pin
it against the torchaudio reference while that reference still exists, and check
the invariants that fluency scoring depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from voxscore.asr.align import (
    Word,
    ctc_forced_align,
    extract_pauses,
    normalise_for_alignment,
    phonation_time,
)


def _emission(T: int, C: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(T, C, generator=g), dim=-1)


class TestCtcForcedAlign:
    def test_shapes_and_monotonicity(self):
        emission = _emission(200, 32)
        tokens = [5, 9, 9, 3, 7]
        path, scores = ctc_forced_align(emission, tokens, blank=0)
        assert path.shape == (200,)
        assert scores.shape == (200,)
        # The alignment path may never move backwards.
        d = path[1:] - path[:-1]
        assert int(d.min()) >= 0
        assert int(d.max()) <= 2

    def test_every_token_receives_at_least_one_frame(self):
        emission = _emission(300, 32, seed=3)
        tokens = [4, 11, 2, 19, 6, 6, 8]
        path, _ = ctc_forced_align(emission, tokens, blank=0)
        covered = {(int(s) - 1) // 2 for s in path if int(s) % 2 == 1}
        assert covered == set(range(len(tokens)))

    def test_repeated_tokens_are_kept_separate(self):
        # "LL" must occupy two distinct states with a blank between; if the skip
        # rule were wrong they would collapse and every word containing a double
        # letter would get the wrong end time.
        emission = _emission(120, 12, seed=7)
        path, _ = ctc_forced_align(emission, [3, 3], blank=0)
        token_states = sorted({int(s) for s in path if int(s) % 2 == 1})
        assert token_states == [1, 3]

    def test_confident_emission_aligns_where_expected(self):
        # Hand-built emission: token 1 dominates the first half, token 2 the second.
        T, C = 100, 4
        e = torch.full((T, C), -10.0)
        e[:, 0] = -5.0                  # blank, mildly likely everywhere
        e[:50, 1] = 0.0
        e[50:, 2] = 0.0
        e = torch.log_softmax(e, dim=-1)
        path, _ = ctc_forced_align(e, [1, 2], blank=0)
        first = [t for t, s in enumerate(path.tolist()) if s == 1]
        second = [t for t, s in enumerate(path.tolist()) if s == 3]
        assert max(first) < min(second)
        assert abs(min(second) - 50) < 10

    def test_rejects_audio_too_short_for_targets(self):
        with pytest.raises(ValueError, match="too short"):
            ctc_forced_align(_emission(5, 32), [1, 2, 3, 4, 5], blank=0)

    def test_rejects_empty_targets(self):
        with pytest.raises(ValueError, match="no target tokens"):
            ctc_forced_align(_emission(50, 32), [], blank=0)

    def test_matches_torchaudio_reference(self):
        """Pin against torchaudio's CPU implementation where available."""
        try:
            import warnings
            import torchaudio.functional as AF
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                emission = _emission(200, 32, seed=11)
                tokens = [5, 9, 3, 7, 12, 4]
                ref_path, _ = AF.forced_align(
                    emission.unsqueeze(0), torch.tensor([tokens], dtype=torch.int32), blank=0
                )
        except Exception:
            pytest.skip("torchaudio forced_align reference unavailable")

        ours, _ = ctc_forced_align(emission, tokens, blank=0)

        # torchaudio returns token ids per frame; we return extended-state
        # indices. Compare the induced token segmentation, which is what matters.
        ref_ids = ref_path[0].tolist()
        ref_seg, prev, pos = [], 0, -1
        for tid in ref_ids:
            if tid == 0:
                prev = 0
                ref_seg.append(None)
                continue
            if tid != prev:
                pos += 1
            ref_seg.append(pos)
            prev = tid

        our_seg = [((int(s) - 1) // 2 if int(s) % 2 == 1 else None) for s in ours]
        agree = sum(a == b for a, b in zip(ref_seg, our_seg))
        assert agree / len(ref_seg) > 0.95, f"only {agree}/{len(ref_seg)} frames agree"


class TestNormalisation:
    def test_strips_punctuation_and_digits_but_keeps_display(self):
        display, alignable = normalise_for_alignment("I have 3 cats, really!")
        assert alignable == ["I", "HAVE", "CATS", "REALLY"]
        assert "cats" in display
        assert len(display) == len(alignable)

    def test_apostrophes_survive(self):
        _, alignable = normalise_for_alignment("don't isn't")
        assert alignable == ["DON'T", "ISN'T"]


class TestPauses:
    def _words(self, spans):
        return [Word(f"w{i}", s, e, 1.0) for i, (s, e) in enumerate(spans)]

    def test_detects_only_gaps_over_threshold(self):
        w = self._words([(0.0, 0.5), (0.6, 1.0), (1.5, 2.0)])
        pauses = extract_pauses(w, 2.0, min_pause_s=0.25)
        assert len(pauses) == 1
        assert pauses[0].duration == pytest.approx(0.5)

    def test_edge_silence_excluded_by_default(self):
        # Leading/trailing silence reflects the recorder, not the speaker.
        w = self._words([(3.0, 3.5), (3.6, 4.0)])
        assert extract_pauses(w, 10.0, 0.25) == []
        assert len(extract_pauses(w, 10.0, 0.25, trim_edges=False)) == 2

    def test_phonation_time_sums_word_durations(self):
        w = self._words([(0.0, 0.5), (1.0, 1.5)])
        assert phonation_time(w) == pytest.approx(1.0)

    def test_no_words_yields_no_pauses(self):
        assert extract_pauses([], 10.0) == []
