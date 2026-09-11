"""Relevance: does the response answer the question?

This is the module the project stands on. The incumbent vendor manages ~0.60
correlation on grammar, lexical and fluency but only ~0.40 on relevance, and the
asymmetry is explicable: the other three are properties of the response alone,
while relevance is the only category that requires reasoning about the prompt.

The obvious implementation -- embed the response, embed the ideal answers, take
cosine -- is close to uncorrelated with relevance on *personal-narrative* prompts,
because it is dominated by whose story it is. The client's three ideal answers for
"Describe what you did on your last birthday" describe a quiet dinner at home, a
surprise gathering, and a solo day out. Content
overlap is near zero; all three are perfect answers.

So we do not compare content. We derive a per-question rubric:

* **Required elements** come from the question itself and are always available.
* **Required moves** are masked sentences appearing in >= 2 of 3 ideal answers.
  Two independent ideal answers agreeing makes something a requirement; one
  mentioning it makes it personal idiosyncrasy.
* **Profile** is the structural fingerprint of a good answer -- how first-person
  it is, how much concrete detail, how much evaluative language -- estimated from
  the ideal answers as a *distribution* rather than as content to be matched.
* **Shareability** (how much the ideal answers agree with each other) decides how
  far to trust content matching at all, blending the narrative and argumentative
  strategies continuously instead of routing on a brittle question-type label.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field

import numpy as np

from voxscore.features.embed import Embedder, NLI, cosine_matrix, max_align
from voxscore.features.qtype import QuestionProfile, profile_question
from voxscore.utils import textproc as tp
from voxscore.utils.textproc import ParsedText, ratio

log = logging.getLogger(__name__)

# A masked sentence counts as shared between two ideal answers above this cosine.
# Tuned to sit above paraphrase-of-different-content and below same-move.
MOVE_SIM_THRESHOLD = 0.55

# Two required moves closer than this are the same move; keep one.
MOVE_DEDUP_THRESHOLD = 0.85

# Soft match threshold for a question content term being addressed.
TERM_MATCH_THRESHOLD = 0.45

# Topic-drift windowing. Windows below ~25 words embed too noisily to compare.
MIN_WINDOW_WORDS = 25
MIN_DRIFT_WORDS = 75

# A window below this cosine to the question counts as off-topic. Calibrated on
# observed values: on-topic windows measure 0.43-0.73, off-topic 0.33-0.43.
DRIFT_OFFTOPIC_THRESHOLD = 0.42

# Stand-in for the ideal answers' own question-similarity, used only by
# `window_sim_vs_ref` when no ideal answers are supplied. Median observed across
# model answers we have measured.
DEFAULT_REF_WINDOW_SIM = 0.62


# --------------------------------------------------------------------------- #
# Structural profile
# --------------------------------------------------------------------------- #

PROFILE_KEYS = (
    "first_person_ratio",
    "past_tense_ratio",
    "entity_density",
    "affect_ratio",
    "causal_density",
    "contrast_density",
    "opinion_density",
    "hedge_density",
    "mean_sentence_len",
)


def structural_profile(p: ParsedText) -> dict[str, float]:
    """Functional fingerprint of a response, independent of its topic.

    These say *what kind of speech act* was performed -- narrating a personal
    memory, or arguing a position. Relevance for an open prompt is largely the
    question of whether the candidate performed the act the prompt demanded, and
    that is invisible to a bag-of-embeddings similarity.
    """
    n = max(p.n_tokens, 1)
    lem = p.lemmas
    low = [t.lower() for t in p.tokens]

    first_person = sum(1 for t in low if t in {"i", "me", "my", "mine", "myself", "we", "our"})
    past = sum(1 for t, tag in zip(p.tokens, p.pos) if tag == "VERB") or 1
    past_verbs = 0
    if p.doc is not None:
        past_verbs = sum(
            1 for t in p.doc
            if t.pos_ in {"VERB", "AUX"} and t.morph.get("Tense") == ["Past"]
        )

    affect = sum(1 for l in lem if l in tp.AFFECT_WORDS)
    causal = sum(1 for l in lem if l in tp.CAUSAL_CONNECTIVES)
    contrast = sum(1 for l in lem if l in tp.CONTRAST_CONNECTIVES)
    opinion = sum(1 for l in lem if l in tp.OPINION_MARKERS)
    hedge = sum(1 for l in lem if l in tp.HEDGES)

    n_sents = max(len(p.sentences), 1)

    return {
        "first_person_ratio": ratio(first_person, n),
        "past_tense_ratio": ratio(past_verbs, past),
        "entity_density": ratio(len(p.entities), n) * 100,
        "affect_ratio": ratio(affect, n) * 100,
        "causal_density": ratio(causal, n) * 100,
        "contrast_density": ratio(contrast, n) * 100,
        "opinion_density": ratio(opinion, n) * 100,
        "hedge_density": ratio(hedge, n) * 100,
        "mean_sentence_len": ratio(p.n_tokens, n_sents),
    }


# --------------------------------------------------------------------------- #
# Rubric
# --------------------------------------------------------------------------- #

@dataclass
class RelevanceRubric:
    """Everything derived once per question and reused for every response."""

    question_id: str
    question_text: str
    question_terms: list[str] = field(default_factory=list)
    question_elements: list[str] = field(default_factory=list)
    required_moves: list[str] = field(default_factory=list)
    masked_ideals: list[str] = field(default_factory=list)
    ideal_sentences: list[str] = field(default_factory=list)
    shareability: float = 0.0
    ref_window_sim: float = 0.0
    qprofile: QuestionProfile | None = None
    profile: dict[str, tuple[float, float]] = field(default_factory=dict)

    emb_question: np.ndarray | None = None
    emb_elements: np.ndarray | None = None
    emb_moves: np.ndarray | None = None
    emb_ideal_sents: np.ndarray | None = None
    emb_terms: np.ndarray | None = None

    @property
    def has_ideals(self) -> bool:
        return len(self.masked_ideals) >= 2

    def summary(self) -> dict:
        """Inspectable form -- the client can audit what the rubric decided."""
        return {
            "question_id": self.question_id,
            "shareability": round(self.shareability, 4),
            "n_required_moves": len(self.required_moves),
            "n_question_elements": len(self.question_elements),
            "question_terms": self.question_terms,
            "question_profile": self.qprofile.as_dict() if self.qprofile else None,
            "required_moves": self.required_moves,
            "profile": {k: [round(m, 4), round(s, 4)] for k, (m, s) in self.profile.items()},
        }


def _question_elements(p: ParsedText) -> list[str]:
    """Decompose the question into the things a response must address.

    "Talk about someone you were once close to but no longer see" yields the
    noun phrase plus each modifying clause, so an answer that names a friend but
    never mentions losing contact is correctly marked incomplete.
    """
    if p.doc is None:
        return []
    elements: list[str] = []

    for chunk in p.doc.noun_chunks:
        if any(t.pos_ in {"NOUN", "PROPN"} and not t.is_stop for t in chunk):
            elements.append(chunk.text.strip())

    # Relative and subordinate clauses carry the real constraints.
    for tok in p.doc:
        if tok.dep_ in {"relcl", "advcl", "acl", "xcomp", "ccomp"}:
            span = p.doc[tok.left_edge.i: tok.right_edge.i + 1]
            txt = span.text.strip()
            if 2 <= len(txt.split()) <= 14:
                elements.append(txt)

    seen, out = set(), []
    for e in elements:
        k = e.lower()
        if k not in seen and len(e.split()) >= 2:
            seen.add(k)
            out.append(e)
    return out[:8]


def build_rubric(
    question_id: str,
    question_text: str,
    ideal_answers: list[str] | None,
    embedder: Embedder,
    nlp=None,
) -> RelevanceRubric:
    """Derive the relevance rubric for one question. Cache the result."""
    nlp = nlp or tp.get_nlp()
    ideal_answers = [a for a in (ideal_answers or []) if a and a.strip()]

    qp = tp.parse(question_text, nlp)
    rub = RelevanceRubric(
        question_id=question_id,
        question_text=question_text,
        question_terms=sorted(set(qp.content_lemmas)),
        question_elements=_question_elements(qp),
        qprofile=profile_question(question_text, nlp, parsed=qp),
    )

    parsed_ideals = [tp.parse(a, nlp) for a in ideal_answers]
    rub.masked_ideals = [p.masked for p in parsed_ideals]

    per_ideal_sents: list[list[str]] = [
        [s for s in p.masked_sentences if len(s.split()) >= 3] for p in parsed_ideals
    ]
    rub.ideal_sentences = [s for group in per_ideal_sents for s in group]

    if len(parsed_ideals) >= 2:
        embs = [embedder.encode(g) if g else np.zeros((0, embedder.dim), np.float32)
                for g in per_ideal_sents]
        rub.shareability = _shareability(parsed_ideals, set(rub.question_terms))
        rub.required_moves = _required_moves(per_ideal_sents, embs)
        rub.profile = _profile_distribution(parsed_ideals)
    elif len(parsed_ideals) == 1:
        # One ideal answer cannot separate requirement from idiosyncrasy, so we
        # deliberately extract no moves rather than treat one person's story as
        # the target. Question-derived features carry the score instead.
        log.info("question %s has 1 ideal answer; moves not derivable", question_id)
        rub.profile = _profile_distribution(parsed_ideals)

    # Per-question reference: how similar a *known-good* answer's windows are to
    # the question. An absolute off-topic threshold cannot work, because baseline
    # similarity varies per question -- measured 0.45-0.82 across probe cases on a
    # single prompt. The ideal answers give us the scale for free.
    rub.emb_question = embedder.encode(question_text)
    if rub.masked_ideals:
        ideal_sims = cosine_matrix(embedder.encode(rub.masked_ideals), rub.emb_question)[:, 0]
        rub.ref_window_sim = float(ideal_sims.mean())
    rub.emb_elements = embedder.encode(rub.question_elements) if rub.question_elements else None
    rub.emb_moves = embedder.encode(rub.required_moves) if rub.required_moves else None
    rub.emb_ideal_sents = embedder.encode(rub.ideal_sentences) if rub.ideal_sentences else None
    rub.emb_terms = embedder.encode(rub.question_terms) if rub.question_terms else None
    return rub


@functools.lru_cache(maxsize=100_000)
def _informativeness(lemma: str) -> float:
    """How much a shared word tells us. Rare words weigh more than common ones.

    Zipf frequency runs ~7 for "the" down to ~1-2 for rare vocabulary. Words at
    or above Zipf 6 ("make", "take", "thing") carry no evidence of shared content
    and are zeroed out.
    """
    try:
        from wordfreq import zipf_frequency

        return max(0.0, 6.0 - zipf_frequency(lemma, "en"))
    except Exception:
        return 1.0


def _shareability(parsed: list[ParsedText], question_terms: set[str]) -> float:
    """Mean pairwise **informativeness-weighted content overlap** of the ideal answers.

    High for opinion prompts, where three model answers converge on the same
    substantive vocabulary (navigation, calculator, arithmetic, dependence). Near
    zero for personal prompts, where the rare words are one person's specifics
    (one person's biryani, another's harbour, a third's allotment) and only common function vocabulary is shared.

    Took three attempts, and both failures are instructive:

    1. **Sentence-embedding agreement** ranked personal *above* opinion
       (0.711 vs 0.676) -- backwards. Bi-encoder similarity conflates form with
       content: *"It felt warm, easy, and very real"* and *"It felt personal,
       unforced, and easy to enjoy"* are near-identical in form while sharing no
       content at all.
    2. **Plain lemma Jaccard with question terms removed** was also backwards
       (0.085 vs 0.047), for two compounding reasons: excluding question terms
       deleted precisely the shared substance of an opinion prompt (*technology*,
       *dependent*, *thinking*), while the remaining overlap on both families was
       dominated by common verbs any two English texts share.

    Weighting by rarity fixes both: shared rare words are evidence of reusable
    content, shared common words are evidence of nothing. Question terms stay in,
    because for an opinion prompt they *are* the content.
    """
    sets: list[set[str]] = []
    for p in parsed:
        sets.append({l for l in p.content_lemmas if not l.startswith("<")})

    scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            union = sets[i] | sets[j]
            wu = sum(_informativeness(l) for l in union)
            if wu > 0:
                wi = sum(_informativeness(l) for l in inter)
                scores.append(wi / wu)
    return float(np.mean(scores)) if scores else 0.0


def _required_moves(groups: list[list[str]], embs: list[np.ndarray]) -> list[str]:
    """Masked sentences corroborated by at least one other ideal answer."""
    candidates: list[tuple[float, str]] = []
    for i, sents in enumerate(groups):
        if not sents or embs[i].size == 0:
            continue
        others = [embs[j] for j in range(len(embs)) if j != i and embs[j].size]
        if not others:
            continue
        pool = np.vstack(others)
        best = max_align(embs[i], pool)
        for s, score in zip(sents, best):
            if score >= MOVE_SIM_THRESHOLD:
                candidates.append((float(score), s))

    candidates.sort(key=lambda x: -x[0])
    kept: list[str] = []
    kept_emb: list[np.ndarray] = []
    all_emb = {s: e for group, emb in zip(groups, embs) for s, e in zip(group, emb)}
    for score, s in candidates:
        e = all_emb.get(s)
        if e is None:
            continue
        if kept_emb and float(cosine_matrix(e, np.vstack(kept_emb)).max()) > MOVE_DEDUP_THRESHOLD:
            continue
        kept.append(s)
        kept_emb.append(e)
    return kept[:10]


def _profile_distribution(parsed: list[ParsedText]) -> dict[str, tuple[float, float]]:
    """Mean and std of each structural feature across the ideal answers."""
    rows = [structural_profile(p) for p in parsed]
    out: dict[str, tuple[float, float]] = {}
    for k in PROFILE_KEYS:
        vals = np.array([r[k] for r in rows], dtype=np.float32)
        # Floor the std: with three samples a coincidentally tight spread would
        # otherwise make the z-distance explode for a perfectly good answer.
        out[k] = (float(vals.mean()), float(max(vals.std(), 0.15 * abs(vals.mean()) + 1e-3)))
    return out


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def relevance_features(
    response: ParsedText,
    rubric: RelevanceRubric,
    embedder: Embedder,
    nli: NLI | None = None,
    n_windows: int = 4,
) -> dict[str, float]:
    """Relevance features for one response against one question's rubric."""
    f: dict[str, float] = {}
    sents = [s for s in response.sentences if s.split()]
    masked_sents = [s for s in response.masked_sentences if s.split()]

    if not response.text.strip():
        return {k: 0.0 for k in _FEATURE_NAMES}

    emb_resp = embedder.encode(response.masked)
    emb_sents = embedder.encode(masked_sents) if masked_sents else np.zeros((0, embedder.dim), np.float32)

    # --- question anchoring (always available, no ideal answers needed) ---
    f["sim_q"] = float(cosine_matrix(emb_resp, rubric.emb_question)[0, 0])

    if rubric.emb_elements is not None and emb_sents.size:
        cov = max_align(rubric.emb_elements, emb_sents)
        f["element_coverage"] = float(cov.mean())
        f["element_coverage_min"] = float(cov.min())
    else:
        f["element_coverage"] = f["sim_q"]
        f["element_coverage_min"] = f["sim_q"]

    if rubric.emb_terms is not None and emb_sents.size:
        term_best = max_align(rubric.emb_terms, emb_sents)
        f["topic_term_coverage"] = float((term_best >= TERM_MATCH_THRESHOLD).mean())
    else:
        f["topic_term_coverage"] = 0.0

    # --- ideal-answer conditioned (gated by shareability) ---
    f["shareability"] = rubric.shareability

    if rubric.emb_ideal_sents is not None and emb_sents.size:
        f["sent_align_mean"] = float(max_align(emb_sents, rubric.emb_ideal_sents).mean())
    else:
        f["sent_align_mean"] = 0.0

    if rubric.emb_moves is not None and emb_sents.size:
        mv = max_align(rubric.emb_moves, emb_sents)
        f["move_coverage"] = float(mv.mean())
        f["move_coverage_min"] = float(mv.min())
    else:
        f["move_coverage"] = 0.0
        f["move_coverage_min"] = 0.0

    # --- structural profile match ---
    prof = structural_profile(response)
    if rubric.profile:
        # Mean of per-feature exp(-|z|), not exp(-mean|z|). The latter is
        # dominated by whichever single feature happens to deviate most and
        # compressed every response into 0.01-0.12, which ranks but does not
        # calibrate. Averaging the per-feature scores keeps the output spread
        # across [0,1] and lets one unusual dimension be outvoted.
        scores = [
            float(np.exp(-abs(prof[k] - m) / s))
            for k, (m, s) in rubric.profile.items()
            if k in prof and s > 0
        ]
        f["profile_match"] = float(np.mean(scores)) if scores else 0.0
    else:
        f["profile_match"] = 0.0
    for k in PROFILE_KEYS:
        f[f"prof_{k}"] = float(prof.get(k, 0.0))

    # --- novelty and substance ---------------------------------------------
    # Measured probing showed similarity features systematically *reward* the
    # two things we most need to catch: prompt echo scored highest of all on
    # sim_q (0.787) and element_coverage (0.886), and a contentless vague answer
    # matched a genuine one on move_coverage (0.761 vs 0.788). Cosine cannot
    # distinguish "addressed the question" from "restated the question" or from
    # "said nothing at length". These two features can.
    q_terms = set(rubric.question_terms)
    content = response.content_lemmas
    novel = [l for l in content if l not in q_terms]
    f["content_novelty_vs_q"] = ratio(len(set(novel)), max(len(set(content)), 1))
    f["content_word_rate"] = ratio(len(content), max(response.n_tokens, 1))
    f["distinct_content_rate"] = ratio(len(set(content)), max(len(content), 1))

    # --- topic drift ---
    drift = _topic_drift(response, rubric, embedder, n_windows)
    f.update(drift)

    # --- specificity: that detail exists, never which detail ---
    # Counts concrete *vocabulary* as well as named entities. Counting only NER
    # and numerals scored 0.000 on a strong answer that was dense with detail
    # ("inseparable", "secondary school", "terrible taste in music") but which
    # deliberately never named the friend -- saying "someone" and "she"
    # throughout. Declining to name a person is a stylistic choice, not an
    # absence of specificity, and penalising it would systematically mark down
    # exactly the discreet, fluent answers we most want to reward.
    n = max(response.n_tokens, 1)
    numerals = sum(1 for t in response.pos if t == "NUM")
    concrete = sum(
        1 for l, pos in zip(response.lemmas, response.pos)
        if pos in {"NOUN", "PROPN", "VERB", "ADJ"} and l.isalpha()
        and _informativeness(l) >= 1.5
    )
    f["specificity"] = float((len(response.entities) + numerals + concrete) / n * 100)
    f["entity_specificity"] = float((len(response.entities) + numerals) / n * 100)

    # --- unsupported content ---
    q_and_ideal = set(rubric.question_terms)
    for s in rubric.ideal_sentences:
        q_and_ideal.update(w.lower().strip(".,!?") for w in s.split())
    unsupported = [l for l in response.content_lemmas if l not in q_and_ideal]
    f["unsupported_content_ratio"] = ratio(len(unsupported), max(len(response.content_lemmas), 1))

    # --- what the question demands, read off the question itself ---
    # This is the primary router. `shareability` (below) needs ideal answers to
    # exist and to be trustworthy; the client has said theirs are AI-generated
    # and that the pipeline must not depend on them. The prompt is always present
    # and states the speech act it wants in its own wording.
    qp = rubric.qprofile or profile_question(rubric.question_text)
    f["q_argumentativeness"] = qp.argumentativeness
    f["q_narrativity"] = qp.narrativity
    f["q_wants_explanation"] = 1.0 if qp.wants_explanation else 0.0
    f["q_wants_comparison"] = 1.0 if qp.wants_comparison else 0.0

    # --- argumentative block ---
    f.update(_argument_features(response, prof))

    # --- NLI coverage ---
    if nli is not None and sents:
        pool = sents[:12]
        if rubric.required_moves:
            ent = [nli.max_entailment(pool, m) for m in rubric.required_moves[:6]]
            f["nli_move_entailment"] = float(np.mean(ent)) if ent else 0.0
        else:
            f["nli_move_entailment"] = 0.0
        f["nli_self_contradiction"] = _self_contradiction(nli, sents)
    else:
        f["nli_move_entailment"] = 0.0
        f["nli_self_contradiction"] = 0.0

    return f


def _topic_drift(
    response: ParsedText,
    rubric: RelevanceRubric,
    embedder: Embedder,
    n_windows: int,
) -> dict[str, float]:
    """Track similarity to the question across the response.

    Catches the answer that starts on topic and wanders -- invisible to a single
    whole-response embedding, which averages the drift away.
    """
    words = response.masked.split()
    # drift_available distinguishes "measured as on-topic" from "too short to
    # measure". Returning 0.0 for both reads as a clean bill of health, which
    # silently favoured short responses: with a 75-word minimum, a 52-word answer
    # scored a free 0% off-topic while an 80-word answer was actually assessed.
    _empty = {"topic_drift_slope": 0.0, "min_window_sim": 0.0, "mean_window_sim": 0.0,
              "window_sim_vs_ref": 0.0, "pct_windows_offtopic": 0.0,
              "drift_available": 0.0}

    # Minimum window size, not a fixed window count. Short windows give unstable
    # embeddings: on 48-87 word fixtures a fixed 4-way split produced ~15-word
    # windows and pct_windows_offtopic came out 0.50 / 0.00 / 0.50 across three
    # responses that were all squarely on topic -- pure noise, and it was the
    # single largest source of error in relevance ranking.
    if len(words) < MIN_DRIFT_WORDS or rubric.emb_question is None:
        return dict(_empty)

    size = max(len(words) // n_windows, MIN_WINDOW_WORDS)
    chunks = [" ".join(words[i: i + size]) for i in range(0, len(words), size)]
    chunks = [c for c in chunks if len(c.split()) >= MIN_WINDOW_WORDS // 2]
    # Fewer than three windows cannot distinguish drift from noise; report
    # neutral rather than a number that looks like a measurement.
    if len(chunks) < 3:
        return dict(_empty)

    sims = cosine_matrix(embedder.encode(chunks), rubric.emb_question)[:, 0]
    x = np.arange(len(sims), dtype=np.float32)
    slope = float(np.polyfit(x, sims, 1)[0]) if len(sims) > 1 else 0.0

    # Fixed threshold, deliberately NOT derived from the ideal answers.
    #
    # Three versions of this have now been measured. Grading each response
    # against its own mean never fired at all (0.000 everywhere, off-topic cases
    # included) because a uniformly off-topic answer sets itself a low bar.
    # Scaling to the ideal answers looked principled and was worse: model answers
    # are *unusually* on-topic by construction, so 0.75x their similarity is a bar
    # real answers fail. Measured cost -- on the graded fixture it flagged the
    # BEST opinion answer as 50% off-topic while every weaker answer scored 0.0,
    # dropping Spearman from 1.000 to 0.900. The penalty landed on the best answer
    # alone because it was the only one long enough for drift to be measured.
    #
    # A fixed bar calibrated on observed window similarities (on-topic 0.43-0.73,
    # off-topic 0.33-0.43) restores 1.000 and removes an ideal-answer dependency
    # from a feature that never needed one.
    thresh = DRIFT_OFFTOPIC_THRESHOLD

    # `window_sim_vs_ref` is a different quantity and does still want a reference:
    # it is the prompt-echo signature (a genuine answer sits BELOW a model
    # answer's question-similarity because it adds content; echo sits above).
    # Falls back to the calibrated baseline when no ideal answers exist, so the
    # signal survives the no-ideal-answers configuration rather than reading 0.
    ref = rubric.ref_window_sim if rubric.ref_window_sim > 0 else DEFAULT_REF_WINDOW_SIM
    return {
        "topic_drift_slope": slope,
        "min_window_sim": float(sims.min()),
        "mean_window_sim": float(sims.mean()),
        "window_sim_vs_ref": float(sims.mean() / ref) if ref > 0 else 0.0,
        "pct_windows_offtopic": float((sims < thresh).mean()),
        "drift_available": 1.0,
    }


def _argument_features(p: ParsedText, prof: dict[str, float]) -> dict[str, float]:
    """Features that matter for opinion prompts.

    An answer to "do you think online news makes people better informed?" is
    irrelevant if it never takes a position, no matter how fluent it is. These
    degrade to neutral on narrative prompts rather than to zero.
    """
    lem = p.lemmas
    n = max(p.n_tokens, 1)

    stance_lex = sum(1 for l in lem if l in tp.OPINION_MARKERS)
    agree = sum(1 for l in lem if l in {"agree", "disagree", "yes", "no", "definitely",
                                        "certainly", "absolutely"})
    modal = sum(1 for t, tag in zip(p.tokens, p.pos) if tag == "AUX" and
                t.lower() in {"should", "must", "would", "can", "will"})

    causal_markers = [l for l in lem if l in tp.CAUSAL_CONNECTIVES]
    contrast_markers = [l for l in lem if l in tp.CONTRAST_CONNECTIVES]
    example_markers = sum(1 for l in lem if l in {"example", "instance", "such", "like", "say"})

    return {
        "stance_clarity": float(min((stance_lex + agree + modal) / n * 100 / 4.0, 1.0)),
        "reason_count": float(min(len(causal_markers), 6)),
        "counterargument_presence": float(min(len(contrast_markers) / 2.0, 1.0)),
        "evidence_presence": float(min(example_markers / 2.0, 1.0)),
    }


def _self_contradiction(nli: NLI, sents: list[str]) -> float:
    """Maximum contradiction between the first and second half of the response.

    A response that argues both sides without acknowledging it is drifting, not
    reasoning. Compared in halves rather than pairwise to keep this O(1) in NLI
    calls rather than O(n^2).
    """
    if len(sents) < 4:
        return 0.0
    mid = len(sents) // 2
    a, b = " ".join(sents[:mid]), " ".join(sents[mid:])
    probs = nli.entailment([a], [b])
    return float(probs[0, 2]) if len(probs) else 0.0


_FEATURE_NAMES = (
    "sim_q", "element_coverage", "element_coverage_min", "topic_term_coverage",
    "shareability", "sent_align_mean", "move_coverage", "move_coverage_min",
    "profile_match", "topic_drift_slope", "min_window_sim", "pct_windows_offtopic",
    "specificity", "unsupported_content_ratio", "stance_clarity", "reason_count",
    "content_novelty_vs_q", "content_word_rate", "distinct_content_rate",
    "mean_window_sim", "window_sim_vs_ref", "entity_specificity", "drift_available",
    "q_argumentativeness", "q_narrativity", "q_wants_explanation", "q_wants_comparison",
    "counterargument_presence", "evidence_presence", "nli_move_entailment",
    "nli_self_contradiction",
) + tuple(f"prof_{k}" for k in PROFILE_KEYS)
