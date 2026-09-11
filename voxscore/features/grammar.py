"""Grammar features: typed error rates plus syntactic complexity.

Emitted in **two variants**:

* ``strict`` -- the written norm.
* ``spoken`` -- filled pauses and self-repairs removed before correction.

The spoken variant is the default, and the client's own data is why. Their ideal
answers are scripted to imitate speech: *"Okay uh, it wasn't some huge
celebration"*, *"Um, when I walked in"*, *"Er, nothing dramatic happened"*,
*"wanna"*. A written-English corrector marks down the client's own gold standard.
Both are reported so they can keep whichever correlates better with their raters.

Errors are **typed** via ERRANT rather than counted. "3 article errors and 2
agreement errors per 100 words" is defensible to a candidate contesting a score;
a single number is not. Typed rates are also more robust than the total, since a
weak corrector tends to miss errors rather than invent categories.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from voxscore.config import MODELS, get_device
from voxscore.utils.textproc import ParsedText, ratio, strip_fillers

log = logging.getLogger(__name__)

# ERRANT error types we surface individually. These are the categories that
# dominate L2 English and that raters actually notice.
TRACKED_ERROR_TYPES = {
    "DET": "article_determiner",
    "PREP": "preposition",
    "VERB:SVA": "subject_verb_agreement",
    "VERB:TENSE": "verb_tense",
    "VERB:FORM": "verb_form",
    "NOUN:NUM": "noun_number",
    "WO": "word_order",
    "VERB": "verb_choice",
    "NOUN": "noun_choice",
    "PRON": "pronoun",
    "MORPH": "morphology",
    "SPELL": "spelling",
}

GRAMMAR_FEATURES = (
    tuple(f"err_{v}_per100" for v in TRACKED_ERROR_TYPES.values())
    + (
        "errors_per_100_words", "error_free_sentence_ratio", "n_edits",
        "clauses_per_sentence", "subordination_ratio", "mean_dependency_distance",
        "mean_sentence_len", "verb_ratio", "grammar_confidence", "gec_available",
    )
)


@lru_cache(maxsize=1)
def _load_gec(model_id: str, device_str: str):
    """Load the corrector, preferring the fast tokenizer but not requiring it.

    Building a *fast* T5 tokenizer from a sentencepiece model needs both
    `sentencepiece` and `protobuf`, and when either is missing transformers
    raises "Couldn't instantiate the backend tokenizer". The *slow* tokenizer
    needs only sentencepiece and no conversion step at all, so it is the right
    fallback -- marginally slower per call, and the grammar block is not the
    bottleneck.

    Observed on the client's EC2 instance: sentencepiece present, fast conversion
    still failing, grammar silently abstaining for every item.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        log.warning("fast tokenizer unavailable (%s); using the slow one", exc)
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(torch.device(device_str)).eval()
    return tok, model


@lru_cache(maxsize=1)
def _load_errant():
    import errant

    return errant.load("en")


class GrammarScorer:
    """Corrects sentences, then types the edits."""

    def __init__(self, model_id: str | None = None, device=None):
        self.model_id = model_id or MODELS["gec"].hf_id
        self.device = device or get_device()
        self._gec = None
        self._errant = None
        self.last_call_ok = True
        """False when the corrector could not run.

        Load failures are silent in the worst possible direction: the corrector
        returns the text unchanged, so zero edits are found, error_free_sentence
        _ratio reads 1.0, and grammar scores near MAXIMUM. A missing sentencepiece
        install would hand every candidate a perfect grammar mark. The scoring
        layer reads this and abstains instead."""

    def correct(self, sentences: list[str], batch_size: int = 8) -> list[str]:
        """Run GEC over sentences. Returns the input unchanged on failure."""
        if not sentences:
            return []
        self.last_call_ok = True
        try:
            import torch

            if self._gec is None:
                self._gec = _load_gec(self.model_id, str(self.device))
            tok, model = self._gec

            out: list[str] = []
            for i in range(0, len(sentences), batch_size):
                batch = [f"gec: {s}" for s in sentences[i: i + batch_size]]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(self.device)
                with torch.inference_mode():
                    gen = model.generate(**enc, max_length=128, num_beams=4)
                out.extend(tok.batch_decode(gen, skip_special_tokens=True))
            return out
        except Exception as exc:
            self.last_call_ok = False
            log.error("GEC unavailable (%s). Grammar will ABSTAIN for this item "
                      "rather than report a perfect score.", exc)
            return list(sentences)

    def typed_edits(self, original: list[str], corrected: list[str]) -> list[str]:
        """ERRANT error-type labels for each edit between the two versions."""
        try:
            if self._errant is None:
                self._errant = _load_errant()
            ann = self._errant
        except Exception as exc:
            log.warning("ERRANT unavailable (%s); errors will not be typed", exc)
            return []

        types: list[str] = []
        for o, c in zip(original, corrected):
            if o.strip() == c.strip():
                continue
            try:
                edits = ann.annotate(ann.parse(o), ann.parse(c))
                types.extend(e.type for e in edits)
            except Exception:
                continue
        return types


def grammar_features(
    parsed: ParsedText,
    scorer: GrammarScorer | None = None,
    variant: str = "spoken",
) -> dict[str, float]:
    """Grammar block for one response.

    ``variant='spoken'`` strips filled pauses before correction, so "um" and "er"
    are not counted as grammatical errors. ``variant='strict'`` scores the
    transcript as written English.
    """
    out = {k: 0.0 for k in GRAMMAR_FEATURES}
    sentences = [s for s in parsed.sentences if len(s.split()) >= 3]
    if not sentences:
        return out

    n_words = max(parsed.n_tokens, 1)
    out["grammar_confidence"] = float(min(n_words / 60.0, 1.0))
    out["gec_available"] = 1.0

    if variant == "spoken":
        # Self-repairs are a speech phenomenon, not a grammatical error. Leaving
        # them in makes every hesitant-but-correct speaker look ungrammatical.
        target = [strip_fillers(_strip_immediate_repeats(s)) for s in sentences]
        target = [s for s in target if len(s.split()) >= 3]
    else:
        target = sentences

    if not target:
        return out

    scorer = scorer or GrammarScorer()
    corrected = scorer.correct(target)
    if not getattr(scorer, "last_call_ok", True):
        out["gec_available"] = 0.0
        return out

    changed = sum(1 for o, c in zip(target, corrected) if o.strip() != c.strip())
    out["error_free_sentence_ratio"] = ratio(len(target) - changed, len(target), 1.0)

    types = scorer.typed_edits(target, corrected)
    out["n_edits"] = float(len(types))
    out["errors_per_100_words"] = ratio(len(types) * 100.0, n_words)

    for prefix, name in TRACKED_ERROR_TYPES.items():
        # ERRANT types look like "R:VERB:TENSE" / "M:DET" / "U:PREP"; the leading
        # letter is the operation (Replace/Missing/Unnecessary), which we fold
        # together because raters react to the error, not to its repair.
        count = sum(1 for t in types if t.split(":", 1)[-1] == prefix)
        out[f"err_{name}_per100"] = ratio(count * 100.0, n_words)

    out.update(_complexity(parsed))
    return out


def _strip_immediate_repeats(text: str) -> str:
    """Collapse ``the the`` and ``I I`` before grammar checking."""
    return re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)


def _complexity(parsed: ParsedText) -> dict[str, float]:
    """Syntactic complexity, which separates accuracy from sophistication.

    A candidate producing only short simple clauses can be error-free without
    being a strong speaker. Complexity keeps the grammar score from rewarding
    risk-avoidance.
    """
    out = {
        "clauses_per_sentence": 0.0,
        "subordination_ratio": 0.0,
        "mean_dependency_distance": 0.0,
        "mean_sentence_len": 0.0,
        "verb_ratio": 0.0,
    }
    if parsed.doc is None:
        return out

    try:
        sents = list(parsed.doc.sents)
        if not sents:
            return out

        n_sents = len(sents)
        clause_deps = {"ccomp", "xcomp", "advcl", "relcl", "acl", "csubj"}
        n_clauses = sum(1 for t in parsed.doc if t.dep_ in clause_deps)
        n_verbs = sum(1 for t in parsed.doc if t.pos_ == "VERB")

        out["clauses_per_sentence"] = ratio(n_clauses + n_sents, n_sents)
        out["subordination_ratio"] = ratio(n_clauses, max(n_verbs, 1))
        out["mean_sentence_len"] = ratio(parsed.n_tokens, n_sents)
        out["verb_ratio"] = ratio(n_verbs, max(parsed.n_tokens, 1))

        dists = [abs(t.i - t.head.i) for t in parsed.doc if t.head is not t]
        out["mean_dependency_distance"] = float(np.mean(dists)) if dists else 0.0
    except Exception as exc:
        log.debug("complexity features failed: %s", exc)
    return out
