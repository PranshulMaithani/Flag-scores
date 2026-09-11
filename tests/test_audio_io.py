"""Tests for the .npy audio contract.

These matter more than they look. A silent sample-rate error makes every
duration-based feature wrong -- the entire fluency category -- while raising
nothing and producing plausible numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from voxscore.config import SAMPLE_RATE
from voxscore.utils.audio_io import coerce_waveform, load_npy_item, wav_to_npy


def _tone(dur_s: float = 2.0, sr: int = SAMPLE_RATE, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestCoerceWaveform:
    def test_float32_mono_passthrough(self):
        a = _tone()
        out, notes = coerce_waveform(a, SAMPLE_RATE)
        assert out.dtype == np.float32
        assert out.shape == a.shape
        assert notes == []

    def test_int16_is_scaled_by_dtype_range_not_observed_max(self):
        # A quiet int16 recording must stay quiet. Normalising by observed max
        # would destroy the level information that SNR/quality features rely on.
        quiet = (_tone() * 0.1 * 32767).astype(np.int16)
        out, notes = coerce_waveform(quiet, SAMPLE_RATE)
        assert out.dtype == np.float32
        assert np.max(np.abs(out)) == pytest.approx(0.05, abs=0.01)
        assert any("PCM" in n for n in notes)

    @pytest.mark.parametrize("shape_fn", [
        lambda a: np.stack([a, a]),        # (channels, samples)
        lambda a: np.stack([a, a], axis=1),  # (samples, channels)
    ])
    def test_stereo_downmix_both_layouts(self, shape_fn):
        a = _tone()
        out, notes = coerce_waveform(shape_fn(a), SAMPLE_RATE)
        assert out.ndim == 1
        assert len(out) == len(a)
        assert any("mono" in n for n in notes)

    def test_resampling_preserves_duration(self):
        a = _tone(dur_s=3.0, sr=44100)
        out, notes = coerce_waveform(a, 44100)
        assert len(out) / SAMPLE_RATE == pytest.approx(3.0, abs=0.01)
        assert any("resampled" in n for n in notes)

    def test_non_finite_samples_are_repaired(self):
        a = _tone()
        a[100:105] = np.nan
        a[200] = np.inf
        out, notes = coerce_waveform(a, SAMPLE_RATE)
        assert np.all(np.isfinite(out))
        assert any("non-finite" in n for n in notes)

    def test_overrange_float_is_rescaled(self):
        out, notes = coerce_waveform(_tone() * 40.0, SAMPLE_RATE)
        assert np.max(np.abs(out)) == pytest.approx(1.0, abs=1e-5)
        assert any("exceeded 1.0" in n for n in notes)

    def test_unknown_sample_rate_is_warned_not_hidden(self):
        _, notes = coerce_waveform(_tone(), None)
        assert any("ASSUMED" in n for n in notes)

    def test_implausible_duration_is_flagged(self):
        # 16 kHz data mislabelled as 8 kHz reads as double length.
        long = _tone(dur_s=200.0)
        _, notes = coerce_waveform(long, SAMPLE_RATE)
        assert any("exceeds" in n for n in notes)


class TestLoadNpyItem:
    def test_sidecar_fields_are_attached(self, tmp_path):
        np.save(tmp_path / "x1.npy", _tone(dur_s=2.0))
        (tmp_path / "x1.json").write_text(json.dumps({
            "item_id": "x1", "question_id": "q1",
            "orig_sr": SAMPLE_RATE, "duration_s": 2.0,
        }))
        questions = {"q1": {"text": "Talk about a friend.", "ideal_answers": ["a", "b", "c"]}}
        item = load_npy_item(tmp_path / "x1.npy", questions=questions)
        assert item.item_id == "x1"
        assert item.question_text == "Talk about a friend."
        assert len(item.ideal_answers) == 3
        assert item.duration_s == pytest.approx(2.0, abs=0.01)

    def test_sample_rate_recovered_from_declared_duration(self):
        # The realistic failure: sidecar omits orig_sr, audio is 44.1 kHz.
        # Duration lets us infer the true rate instead of assuming 16 kHz and
        # reporting a 5.5 s answer as 15 s.
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            np.save(d / "y.npy", _tone(dur_s=5.0, sr=44100))
            (d / "y.json").write_text(json.dumps({"item_id": "y", "duration_s": 5.0}))
            item = load_npy_item(d / "y.npy")
        assert item.duration_s == pytest.approx(5.0, abs=0.05)

    def test_sample_rate_mismatch_is_reported(self, tmp_path):
        np.save(tmp_path / "z.npy", _tone(dur_s=10.0))
        (tmp_path / "z.json").write_text(json.dumps({
            "item_id": "z", "orig_sr": SAMPLE_RATE, "duration_s": 3.0,
        }))
        item = load_npy_item(tmp_path / "z.npy")
        assert any("sample rate is probably wrong" in w for w in item.warnings_)

    def test_missing_sidecar_falls_back_to_filename(self, tmp_path):
        np.save(tmp_path / "bare.npy", _tone())
        item = load_npy_item(tmp_path / "bare.npy")
        assert item.item_id == "bare"


class TestWavToNpy:
    def test_roundtrip_matches_within_tolerance(self, tmp_path):
        import soundfile as sf
        a = _tone(dur_s=2.0)
        sf.write(tmp_path / "in.wav", a, SAMPLE_RATE)
        npy_path, json_path = wav_to_npy(tmp_path / "in.wav", tmp_path / "out", question_id="q9")
        assert npy_path.exists() and json_path.exists()
        meta = json.loads(json_path.read_text())
        assert meta["question_id"] == "q9"
        assert meta["orig_sr"] == SAMPLE_RATE
        back = load_npy_item(npy_path)
        assert back.duration_s == pytest.approx(2.0, abs=0.01)
        assert np.corrcoef(back.audio[:len(a)], a)[0, 1] > 0.99
