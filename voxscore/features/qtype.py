"""What does this question demand? Derived from the question text alone.

This replaces `shareability` as the primary router. Shareability measures how much
the three ideal answers agree with each other, which is a real signal but needs
ideal answers to exist and to be good. The client has stated that theirs are
AI-generated, not a gold standard, and that the pipeline must not depend on them.

The question text is always present, is authored deliberately, and states the
speech act it wants in its own wording. *"Do you think technology improves human
thinking or makes people more dependent? Explain with reasons"* demands a
position and supporting reasons. *"Talk about a friend you used to be close with
but later lost touch"* demands a personal narrative with concrete detail. An
answer that performs the wrong act is irrelevant however fluent it is, and that
judgement needs nothing but the prompt.

Everything here is lexical and syntactic. No model, no labels, no training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from voxscore.utils.textproc import ParsedText, get_nlp, parse, ratio

# Phrases that solicit a position. Ordered longest-first so the more specific
# pattern wins when several match.
_OPINION_PATTERNS = [
    r"\bin your opinion\b", r"\bdo you think\b", r"\bdo you agree\b",
    r"\bwhat do you think\b", r"\bwhy do you think\b", r"\bis it better\b",
    r"\bwould you say\b", r"\bdo you believe\b", r"\bhow far do you agree\b",
    r"\bshould\b", r"\bin which ways\b", r"\bwhat.*pressures\b",
]

# Phrases that solicit a personal experience.
_NARRATIVE_PATTERNS = [
    r"\btalk about a\b", r"\btalk about the\b", r"\bdescribe a\b",
    r"\bdescribe the\b", r"\bshare how\b", r"\btell me about\b",
    r"\bdescribe how you\b", r"\ba time when\b", r"\bthe most recent time\b",
    r"\byou (?:used to|liked|enjoyed|had|felt|do|spend|help)\b",
]

_EXPLANATION_PATTERNS = [
    r"\bexplain\b", r"\bwhy\b", r"\bgive reasons\b", r"\bwith reasons\b",
    r"\bhow (?:did|do) you\b",
]

_COMPARISON_PATTERNS = [
    r"\bbetter to\b", r"\bor\b.{0,40}\?", r"\bmore than\b", r"\brather than\b",
    r"\bcompared\b", r"\bchanged from\b", r"\balways produce\b",
]


@dataclass
class QuestionProfile:
    """What the prompt asks the candidate to do."""

    argumentativeness: float   # 0-1, how much a stance is demanded
    narrativity: float         # 0-1, how much personal experience is demanded
    wants_explanation: bool    # reasons are explicitly requested
    wants_comparison: bool     # two options are set against each other
    focus_terms: list          # the content the answer must be about
    imperative: str            # the leading verb, e.g. "talk", "describe"

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def family(self) -> str:
        """Human-readable label. Diagnostic only -- scoring uses the continuous values."""
        if self.argumentativeness > self.narrativity + 0.15:
            return "opinion"
        if self.narrativity > self.argumentativeness + 0.15:
            return "personal"
        return "mixed"


def _hits(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))


def profile_question(question_text: str, nlp=None, parsed: ParsedText | None = None) -> QuestionProfile:
    """Read the demands of a question off its own wording."""
    q = (question_text or "").strip()
    p = parsed or parse(q, nlp or get_nlp())
    low = q.lower()

    n_op = _hits(_OPINION_PATTERNS, low)
    n_na = _hits(_NARRATIVE_PATTERNS, low)

    # A question opening with an auxiliary is a yes/no question, and a yes/no
    # question about the world is a request for a position. Caught "Does teamwork
    # always produce better results than working alone?", which none of the
    # phrase patterns matched despite being unambiguously an opinion prompt.
    if re.match(r"^\s*(does|do|did|is|are|was|were|can|could|should|would|will|has|have)\b",
                low):
        n_op += 1

    # Second-person past reference ("you used to", "you liked") is a strong
    # narrative cue that the phrase list alone can miss.
    if p.doc is not None:
        for tok in p.doc:
            if tok.lower_ == "you" and tok.head.pos_ in {"VERB", "AUX"}:
                if tok.head.morph.get("Tense") == ["Past"]:
                    n_na += 1
                    break

    # A bare question mark with no opinion phrase still leans interrogative.
    if q.endswith("?") and n_op == 0 and n_na == 0:
        n_op += 1

    total = max(n_op + n_na, 1)
    argumentativeness = min(n_op / total, 1.0) if (n_op or n_na) else 0.5
    narrativity = min(n_na / total, 1.0) if (n_op or n_na) else 0.5

    imperative = ""
    if p.doc is not None and len(p.doc):
        for tok in p.doc:
            if tok.pos_ == "VERB":
                imperative = tok.lemma_.lower()
                break

    # Focus terms: the content words the answer has to be about. Drop the framing
    # verbs, which appear in every prompt and carry no topical information.
    framing = {"talk", "describe", "share", "explain", "tell", "think", "say",
               "opinion", "view", "reason", "way", "people"}
    focus = [l for l in p.content_lemmas if l not in framing]

    return QuestionProfile(
        argumentativeness=float(argumentativeness),
        narrativity=float(narrativity),
        wants_explanation=bool(_hits(_EXPLANATION_PATTERNS, low)),
        wants_comparison=bool(_hits(_COMPARISON_PATTERNS, low)),
        focus_terms=sorted(set(focus)),
        imperative=imperative,
    )
