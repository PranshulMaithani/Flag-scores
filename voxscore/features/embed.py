"""Sentence embedding and NLI, wrapped so the rest of the code never touches a model.

Both are loaded lazily and cached. Everything here is batched: relevance scoring
compares every response sentence against every rubric item, so per-sentence calls
would dominate runtime.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch

from voxscore.config import MODELS, get_device

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_embedder(model_id: str, device_str: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device_str)


class Embedder:
    """Bi-encoder sentence embeddings, L2-normalised so dot product is cosine."""

    def __init__(self, model_id: str | None = None, device: torch.device | None = None):
        self.model_id = model_id or MODELS["embedder"].hf_id
        self.device = device or get_device()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_embedder(self.model_id, str(self.device))
        return self._model

    def encode(self, texts: list[str] | str, batch_size: int = 32) -> np.ndarray:
        """Return ``(n, d)`` normalised embeddings. Empty strings yield zero vectors."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        keep = [i for i, t in enumerate(items) if t and t.strip()]
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        if keep:
            vecs = self.model.encode(
                [items[i] for i in keep],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out[keep] = vecs.astype(np.float32)
        return out[0] if single else out

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between normalised embedding sets."""
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0).astype(np.float32)


def max_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of ``a``, its best cosine against any row of ``b``.

    This is the workhorse of coverage scoring: it asks "is this point addressed
    anywhere in the response?" rather than "do these two texts look alike
    overall", which is what makes extra original content free rather than costly.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros(a.shape[0] if a.ndim > 1 else 0, dtype=np.float32)
    return cosine_matrix(a, b).max(axis=1)


# --------------------------------------------------------------------------- #
# NLI
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _load_nli(model_id: str, device_str: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model = model.to(torch.device(device_str)).eval()
    return tok, model


class NLI:
    """Entailment scoring for content coverage and stance detection."""

    def __init__(self, model_id: str | None = None, device: torch.device | None = None):
        self.model_id = model_id or MODELS["nli"].hf_id
        self.device = device or get_device()
        self._tok = None
        self._model = None
        self._label_idx: dict[str, int] = {}

    def _ensure(self):
        if self._model is None:
            self._tok, self._model = _load_nli(self.model_id, str(self.device))
            # Label order differs between checkpoints; read it rather than assume.
            id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
            for idx, name in id2label.items():
                for key in ("entail", "neutral", "contradict"):
                    if name.startswith(key):
                        self._label_idx[key] = idx
        return self._tok, self._model

    def entailment(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """``(n, 3)`` array of [entailment, neutral, contradiction] probabilities."""
        tok, model = self._ensure()
        if not premises:
            return np.zeros((0, 3), dtype=np.float32)

        probs: list[np.ndarray] = []
        for i in range(0, len(premises), batch_size):
            p = premises[i: i + batch_size]
            h = hypotheses[i: i + batch_size]
            enc = tok(p, h, return_tensors="pt", truncation=True,
                      max_length=256, padding=True).to(self.device)
            with torch.inference_mode():
                logits = model(**enc).logits.float()
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

        raw = np.concatenate(probs, axis=0)
        order = [
            self._label_idx.get("entail", 0),
            self._label_idx.get("neutral", 1),
            self._label_idx.get("contradict", 2),
        ]
        return raw[:, order].astype(np.float32)

    def max_entailment(self, premise_pool: list[str], hypothesis: str) -> float:
        """Best entailment of ``hypothesis`` from any premise in the pool.

        Used for coverage: a rubric point counts as addressed if *any* part of the
        response entails it, which is the right semantics for spontaneous speech
        where a single idea may be spread across sentences.
        """
        if not premise_pool or not hypothesis.strip():
            return 0.0
        probs = self.entailment(premise_pool, [hypothesis] * len(premise_pool))
        return float(probs[:, 0].max()) if len(probs) else 0.0
