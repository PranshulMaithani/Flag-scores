"""Central configuration: paths, device selection, model identifiers.

Device handling is deliberately indirect. Development runs on AMD/ROCm where
``torch.cuda`` is the HIP shim, production runs on NVIDIA/CUDA. Nothing in the
codebase may call a CUDA-specific API; everything goes through :func:`get_device`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MODELS_CACHE = PROJECT_ROOT / "models_cache"

for _d in (
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SYNTHETIC_DIR,
    FIGURES_DIR, TABLES_DIR, MODELS_CACHE,
):
    _d.mkdir(parents=True, exist_ok=True)

# Keep every HF download inside the project so the offline bundle is one directory.
os.environ.setdefault("HF_HOME", str(MODELS_CACHE))


# --------------------------------------------------------------------------- #
# Audio contract
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 16_000
"""All internal audio is 16 kHz mono float32 in [-1, 1]. Enforced at load time."""

MAX_DURATION_S = 60.0
"""Client responses cap at 60 s. Longer input signals a data problem, not a long answer."""


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #

def get_device(prefer: str | None = None) -> torch.device:
    """Return the compute device.

    ``torch.cuda`` is also the ROCm/HIP entry point, so this works unchanged on
    the AMD development box and on NVIDIA production hardware.
    """
    if prefer:
        return torch.device(prefer)
    env = os.environ.get("VOXSCORE_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(device: torch.device | None = None) -> torch.dtype:
    """Preferred inference dtype.

    bf16 measured at ~138 TFLOPS vs ~15.8 for fp32 on the dev GPU (8.7x), and
    unlike fp16 it needs no loss scaling. Falls back to fp32 on CPU, where bf16
    is usually slower rather than faster.
    """
    device = device or get_device()
    if device.type == "cpu":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def device_report() -> str:
    """One-line human-readable description of the compute environment."""
    dev = get_device()
    if dev.type == "cpu":
        return "device=cpu dtype=float32"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    backend = "ROCm/HIP" if torch.version.hip else "CUDA"
    return (
        f"device={dev.type} backend={backend} gpu={name} "
        f"vram={total:.1f}GB dtype={get_dtype(dev)}"
    )


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelSpec:
    """A model we depend on, with the licence recorded.

    Licence is a first-class field because the client must clear every asset with
    legal (constraint C5). ``commercial_ok`` is our reading, not legal advice, and
    the audit table in ``reports/`` is generated straight from this registry.
    """

    hf_id: str
    licence: str
    commercial_ok: bool
    purpose: str
    notes: str = ""
    in_use: bool = True
    """False for models that are designed for but not wired into the pipeline.

    The licence audit reads this registry, so an entry that implies a dependency
    we do not actually ship would be misleading to whoever signs it off."""


MODELS: dict[str, ModelSpec] = {
    "asr": ModelSpec(
        hf_id="openai/whisper-large-v3",
        licence="Apache-2.0",
        commercial_ok=True,
        purpose="Transcription + per-window language identification",
        notes="Multilingual on purpose: the foreign-language flag needs real "
              "recognition of Hindi/Tagalog/etc, not English gibberish.",
    ),
    "aligner": ModelSpec(
        hf_id="facebook/wav2vec2-base-960h",
        licence="Apache-2.0",
        commercial_ok=True,
        purpose="CTC forced alignment for word-level timings",
        notes="Feeds every fluency pause feature. Whisper's own timestamps are "
              "too coarse for this.",
    ),
    "embedder": ModelSpec(
        hf_id="BAAI/bge-m3",
        licence="MIT",
        commercial_ok=True,
        purpose="Sentence embeddings for relevance and repetition",
    ),
    "nli": ModelSpec(
        hf_id="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        licence="MIT",
        commercial_ok=True,
        purpose="Entailment coverage and stance detection",
    ),
    "gec": ModelSpec(
        hf_id="Unbabel/gec-t5_small",
        licence="Apache-2.0",
        commercial_ok=True,
        purpose="Grammatical error correction, diffed by ERRANT into typed errors",
        notes="Chosen on LICENCE, not quality. Every stronger GEC model on the Hub "
              "is non-commercial: vennify/t5-base-grammar-correction is CC-BY-NC-SA "
              "(86k downloads, the popular default), grammarly/coedit-large is "
              "CC-BY-NC, pszemraj/flan-t5-large-grammar-synthesis is dual-licensed "
              "with an NC term, and prithivida/grammar_error_correcter_v1 states no "
              "licence at all (no licence means no grant). This is the only clean "
              "option and it is t5-small. See reports/ for the measured accuracy cost.",
    ),
    "lid": ModelSpec(
        hf_id="speechbrain/lang-id-voxlingua107-ecapa",
        licence="Apache-2.0",
        commercial_ok=True,
        purpose="Second-opinion acoustic language ID",
        in_use=False,
        notes="NOT WIRED IN. The foreign-language flag uses Whisper's own windowed "
              "posteriors plus a text channel, and reached 0% accent false positives "
              "without a second acoustic model, so this was never added. Kept in the "
              "registry because it is the obvious next channel if the flag needs one. "
              "Weights Apache-2.0; VoxLingua107 corpus CC-BY-4.0.",
    ),
    "speaker": ModelSpec(
        hf_id="speechbrain/spkrec-ecapa-voxceleb",
        licence="Apache-2.0",
        commercial_ok=True,
        purpose="Speaker embeddings for the optional multi-speaker flag",
        in_use=False,
        notes="NOT WIRED IN. The multi-speaker flag was scoped as a nice-to-have "
              "(client: 'you can if you want to') and not built.",
    ),
}


@dataclass
class PipelineConfig:
    """Runtime knobs. Defaults are the research settings, not the fast settings."""

    device: str | None = None
    # Resolved in __post_init__, not as a field default. A dataclass default is
    # evaluated once at class-definition time, so an offline deployment that
    # rewrites the registry afterwards had no effect here and Whisper kept
    # downloading from the Hub despite a local copy being present.
    asr_model: str | None = None
    asr_beam_size: int = 5

    # Pause thresholds. 250 ms is the standard boundary between articulatory
    # juncture and a perceptible hesitation in the L2 fluency literature.
    short_pause_s: float = 0.25
    long_pause_s: float = 1.0

    # Language-ID windowing for the foreign-language flag.
    lid_window_s: float = 5.0
    lid_hop_s: float = 2.5

    cache_stages: bool = True
    cache_dir: Path = field(default_factory=lambda: INTERIM_DIR / "stage_cache")

    def __post_init__(self) -> None:
        if self.asr_model is None:
            self.asr_model = MODELS["asr"].hf_id
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
