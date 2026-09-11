"""Text processing shared across feature modules: parsing, masking, disfluency.

Entity masking is the load-bearing piece. On personal-narrative prompts the ideal
answers are saturated with specifics -- a friend's name, a city, a year -- that no
other candidate will ever reproduce. Comparing raw text to those answers measures
whose story it is, not whether the question was answered. Masking strips exactly
that layer while preserving the structure we actually want to compare.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

# Entity types worth masking. PERSON/GPE/LOC/FAC/ORG/DATE/TIME are the ones that
# carry personal specificity; we keep PERCENT/MONEY/QUANTITY because in opinion
# answers they are often part of a genuine argument.
MASKABLE_ENTS = {
    "PERSON": "<PERSON>",
    "GPE": "<PLACE>",
    "LOC": "<PLACE>",
    "FAC": "<PLACE>",
    "ORG": "<ORG>",
    "DATE": "<TIME>",
    "TIME": "<TIME>",
    "EVENT": "<EVENT>",
    "WORK_OF_ART": "<WORK>",
    "NORP": "<GROUP>",
    "LANGUAGE": "<GROUP>",
}

FILLERS = {
    "um", "uh", "erm", "er", "ah", "hmm", "mm", "mhm", "eh", "uhh", "umm",
}

# Discourse markers that look like fillers in spoken English but are not errors.
DISCOURSE_MARKERS = {
    "well", "so", "okay", "ok", "right", "like", "actually", "basically",
    "honestly", "anyway", "yeah", "i mean", "you know", "sort of", "kind of",
}

CAUSAL_CONNECTIVES = {
    "because", "since", "therefore", "thus", "hence", "so", "as", "consequently",
    "due", "reason", "result", "leads", "causes",
}

CONTRAST_CONNECTIVES = {
    "however", "although", "though", "whereas", "but", "yet", "nevertheless",
    "nonetheless", "conversely", "admittedly", "still", "otherwise",
}

OPINION_MARKERS = {
    "think", "believe", "feel", "opinion", "view", "argue", "reckon", "suppose",
    "personally", "convinced", "doubt", "agree", "disagree",
}

HEDGES = {
    "maybe", "perhaps", "possibly", "probably", "might", "could", "seems",
    "somewhat", "fairly", "rather", "generally", "usually", "tend", "often",
}

AFFECT_WORDS = {
    "happy", "sad", "glad", "grateful", "warm", "calm", "nervous", "anxious",
    "excited", "surprised", "emotional", "special", "wonderful", "terrible",
    "enjoyed", "loved", "hated", "missed", "proud", "relieved", "comfortable",
    "lonely", "guilty", "awkward", "content", "peaceful", "frustrated", "angry",
}


@functools.lru_cache(maxsize=2)
def get_nlp(model: str = "en_core_web_sm"):
    """Load and cache a spaCy pipeline."""
    import spacy

    try:
        return spacy.load(model)
    except OSError:
        return spacy.load("en_core_web_sm")


@dataclass
class ParsedText:
    text: str
    masked: str
    sentences: list[str] = field(default_factory=list)
    masked_sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)
    pos: list[str] = field(default_factory=list)
    content_lemmas: list[str] = field(default_factory=list)
    entities: list[tuple[str, str]] = field(default_factory=list)
    doc: object | None = None

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def parse(text: str, nlp=None) -> ParsedText:
    """Parse once, reuse everywhere. spaCy is the bottleneck if called repeatedly."""
    nlp = nlp or get_nlp()
    doc = nlp(text or "")

    masked_parts: list[str] = []
    last = 0
    for ent in doc.ents:
        if ent.label_ in MASKABLE_ENTS:
            masked_parts.append(doc.text[last:ent.start_char])
            masked_parts.append(MASKABLE_ENTS[ent.label_])
            last = ent.end_char
    masked_parts.append(doc.text[last:])
    masked = "".join(masked_parts)

    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    masked_sentences = _mask_sentences(doc)

    tokens = [t.text for t in doc if not t.is_space]
    lemmas = [t.lemma_.lower() for t in doc if not t.is_space]
    pos = [t.pos_ for t in doc if not t.is_space]
    content = [
        t.lemma_.lower()
        for t in doc
        if t.pos_ in CONTENT_POS and not t.is_stop and t.is_alpha
    ]

    return ParsedText(
        text=text,
        masked=masked,
        sentences=sentences,
        masked_sentences=masked_sentences,
        tokens=tokens,
        lemmas=lemmas,
        pos=pos,
        content_lemmas=content,
        entities=[(e.text, e.label_) for e in doc.ents],
        doc=doc,
    )


def _mask_sentences(doc) -> list[str]:
    """Per-sentence masked text, aligned 1:1 with the unmasked sentence list."""
    out: list[str] = []
    for sent in doc.sents:
        if not sent.text.strip():
            continue
        parts, last = [], sent.start_char
        for ent in sent.ents:
            if ent.label_ in MASKABLE_ENTS:
                parts.append(doc.text[last:ent.start_char])
                parts.append(MASKABLE_ENTS[ent.label_])
                last = ent.end_char
        parts.append(doc.text[last:sent.end_char])
        out.append("".join(parts).strip())
    return out


# --------------------------------------------------------------------------- #
# Disfluency
# --------------------------------------------------------------------------- #

_FILLER_RE = re.compile(
    r"\b(" + "|".join(sorted(FILLERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def count_fillers(text: str) -> int:
    return len(_FILLER_RE.findall(text or ""))


def strip_fillers(text: str) -> str:
    """Remove filled pauses, then tidy the whitespace and punctuation they leave."""
    out = _FILLER_RE.sub(" ", text or "")
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"([,.!?;:])\s*\1+", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def find_immediate_repeats(tokens: list[str], max_span: int = 4) -> list[tuple[int, int]]:
    """Immediate repetitions -- ``the the``, ``I went I went``.

    These are *disfluencies*, not the padding-level repetition the client wants
    flagged. They belong to fluency, and are explicitly excluded from the
    `repetition` flag so that a nervous but honest speaker is never accused of
    gaming the assessment.
    """
    low = [t.lower() for t in tokens]
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(low):
        matched = False
        for n in range(max_span, 0, -1):
            if i + 2 * n <= len(low) and low[i: i + n] == low[i + n: i + 2 * n]:
                spans.append((i, i + 2 * n))
                i += 2 * n
                matched = True
                break
        if not matched:
            i += 1
    return spans


def ratio(numer: float, denom: float, default: float = 0.0) -> float:
    """Division that returns ``default`` instead of raising on a zero denominator."""
    return float(numer) / float(denom) if denom else default
