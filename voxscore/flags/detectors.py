"""Cheating / gaming flags.

Per the client these detect **gaming**, not disfluency. A nervous candidate who
stumbles and repeats a word is not cheating, and must never be flagged; a
candidate who restates the prompt for 60 seconds is. That distinction drives
several choices below, most visibly the removal of immediate repetitions before
any repetition scoring happens.

Every flag emits a **continuous 0-100 score** plus a recommended threshold. The
client sets the operating point later from the ROC curves we ship, so nothing
here hard-codes a decision. ``fired`` is a convenience at the default threshold,
not a verdict.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field

import numpy as np

from voxscore.features.embed import Embedder, cosine_matrix
from voxscore.utils import textproc as tp
from voxscore.utils.textproc import ParsedText, ratio

log = logging.getLogger(__name__)


@dataclass
class FlagResult:
    name: str
    score: float                      # 0-100, higher = more suspicious
    threshold: float                  # recommended default operating point
    features: dict[str, float] = field(default_factory=dict)
    evidence: str = ""

    @property
    def fired(self) -> bool:
        return self.score >= self.threshold

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 2),
            "threshold": self.threshold,
            "fired": self.fired,
            "evidence": self.evidence,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }


def _squash(x: float, lo: float, hi: float) -> float:
    """Map a raw feature onto [0,1] suspicion, clamped outside ``[lo, hi]``."""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# prompt_read
# --------------------------------------------------------------------------- #

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length. O(len(a) * len(b)), both are short."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def prompt_read_flag(
    response: ParsedText,
    question_text: str,
    relevance_feats: dict[str, float],
    threshold: float = 55.0,
) -> FlagResult:
    """Candidate restates or paraphrases the prompt to fill time.

    The naive signal -- similarity to the question -- is actively misleading: in
    probing, a pure prompt-echo response scored the *highest of all* cases on
    both `sim_q` (0.787) and `element_coverage` (0.886), above two genuine
    answers. A good answer is topically similar to the question too.

    What separates echo is that it adds **nothing of its own**. Three independent
    channels capture that:

    * *lexical* -- literal overlap with the question (n-grams, LCS)
    * *comparative* -- `window_sim_vs_ref`, similarity to the question measured
      against what the ideal answers achieve. A genuine answer scores **below**
      the reference because it introduces new content; echo scored 1.289, i.e.
      more question-like than a model answer. This catches paraphrased echo that
      literal overlap misses.
    * *substantive* -- new content words and concrete detail, both near zero for echo
    """
    r_tokens = [t.lower() for t in response.tokens if t.isalnum()]
    q_tokens = [t.lower() for t in question_text.split() if t.strip(".,?!").isalnum()]

    lcs = _lcs_length(r_tokens, q_tokens)
    lcs_ratio = ratio(lcs, max(len(q_tokens), 1))

    overlaps = {}
    for n in (3, 4):
        rg, qg = _ngrams(r_tokens, n), _ngrams(q_tokens, n)
        overlaps[f"ngram{n}_overlap_q"] = ratio(len(rg & qg), max(len(rg), 1))

    novelty = relevance_feats.get("content_novelty_vs_q", 1.0)
    vs_ref = relevance_feats.get("window_sim_vs_ref", 0.0)
    specificity = relevance_feats.get("specificity", 0.0)
    distinct = relevance_feats.get("distinct_content_rate", 1.0)

    feats = {
        "lcs_ratio_q": lcs_ratio,
        **overlaps,
        "content_novelty_vs_q": novelty,
        "window_sim_vs_ref": vs_ref,
        "specificity": specificity,
        "distinct_content_rate": distinct,
    }

    # Weights reflect how diagnostic each channel proved in probing, not a fit.
    signals = {
        "echoes question lexically": (_squash(lcs_ratio, 0.45, 0.9), 0.20),
        "repeats question n-grams": (_squash(max(overlaps.values()), 0.05, 0.4), 0.15),
        "more question-like than a model answer": (_squash(vs_ref, 0.95, 1.30), 0.30),
        "adds little new content": (_squash(1.0 - novelty, 0.15, 0.5), 0.25),
        "no concrete detail": (_squash(2.0 - specificity, 1.0, 2.0), 0.10),
    }
    score = 100.0 * sum(v * w for v, w in signals.values())

    top = sorted(signals.items(), key=lambda kv: -kv[1][0] * kv[1][1])
    evidence = "; ".join(f"{k} ({v:.2f})" for k, (v, _) in top[:3] if v > 0.1) or "no echo signal"

    return FlagResult("prompt_read", score, threshold, feats, evidence)


# --------------------------------------------------------------------------- #
# repetition
# --------------------------------------------------------------------------- #

def repetition_flag(
    response: ParsedText,
    embedder: Embedder | None = None,
    audio: np.ndarray | None = None,
    threshold: float = 55.0,
) -> FlagResult:
    """Padding-level repetition: saying the same thing again to fill time.

    Explicitly **not** disfluency. The client's definition is cheating detection,
    so immediate repetitions (``the the``, ``I went I went``) are stripped before
    anything is measured -- they are stutters and belong to the fluency score. A
    candidate who is nervous must not be accused of gaming the assessment.
    """
    tokens = [t.lower() for t in response.tokens if t.isalnum()]

    # Remove stutters so only deliberate restatement remains.
    disfluent = tp.find_immediate_repeats(response.tokens)
    drop: set[int] = set()
    for lo, hi in disfluent:
        mid = lo + (hi - lo) // 2
        drop.update(range(lo, mid))
    cleaned = [t for i, t in enumerate(tokens) if i not in drop]

    feats: dict[str, float] = {
        "n_tokens": float(len(cleaned)),
        "disfluent_repeat_spans": float(len(disfluent)),
    }

    for n in (1, 2, 3, 4):
        grams = [tuple(cleaned[i: i + n]) for i in range(len(cleaned) - n + 1)]
        feats[f"distinct_{n}"] = ratio(len(set(grams)), max(len(grams), 1), 1.0)

    # gzip ratio: cheap, model-free, and surprisingly hard to fool. Repeated text
    # compresses; varied text does not.
    raw = " ".join(cleaned).encode("utf-8")
    feats["gzip_ratio"] = ratio(len(gzip.compress(raw, 6)), max(len(raw), 1), 1.0) if raw else 1.0

    feats["longest_repeated_span"] = float(_longest_repeated_span(cleaned))
    feats["repeated_span_coverage"] = ratio(feats["longest_repeated_span"] * 2, max(len(cleaned), 1))

    sents = [s for s in response.sentences if len(s.split()) >= 4]
    if embedder is not None and len(sents) >= 2:
        emb = embedder.encode(sents)
        sim = cosine_matrix(emb, emb)
        np.fill_diagonal(sim, -1.0)
        feats["sent_self_sim_max"] = float(sim.max())
        feats["sent_self_sim_mean_top"] = float(np.sort(sim, axis=1)[:, -1].mean())
    else:
        feats["sent_self_sim_max"] = 0.0
        feats["sent_self_sim_mean_top"] = 0.0

    if audio is not None and len(audio) > 16000 * 4:
        feats["audio_self_similarity"] = _audio_self_similarity(audio)
    else:
        feats["audio_self_similarity"] = 0.0

    signals = {
        "repeats long word spans": (_squash(feats["repeated_span_coverage"], 0.12, 0.5), 0.25),
        "low n-gram variety": (_squash(1.0 - feats["distinct_3"], 0.1, 0.45), 0.25),
        "transcript compresses unusually well": (_squash(1.0 - feats["gzip_ratio"], 0.55, 0.75), 0.15),
        "near-duplicate sentences": (_squash(feats["sent_self_sim_max"], 0.80, 0.97), 0.25),
        "duplicated audio": (_squash(feats["audio_self_similarity"], 0.80, 0.97), 0.10),
    }
    score = 100.0 * sum(v * w for v, w in signals.values())
    top = sorted(signals.items(), key=lambda kv: -kv[1][0] * kv[1][1])
    evidence = "; ".join(f"{k} ({v:.2f})" for k, (v, _) in top[:3] if v > 0.1) or "no repetition signal"

    return FlagResult("repetition", score, threshold, feats, evidence)


def _longest_repeated_span(tokens: list[str], min_len: int = 3) -> int:
    """Length of the longest token span occurring more than once."""
    n = len(tokens)
    if n < 2 * min_len:
        return 0
    best = 0
    seen: dict[tuple[str, ...], int] = {}
    for length in range(min_len, min(n // 2, 30) + 1):
        seen.clear()
        found = False
        for i in range(n - length + 1):
            g = tuple(tokens[i: i + length])
            if g in seen and i - seen[g] >= length:
                best = length
                found = True
                break
            seen.setdefault(g, i)
        if not found:
            break
    return best


def _audio_self_similarity(audio: np.ndarray, sr: int = 16000) -> float:
    """Peak off-diagonal similarity between 1-second mel frames.

    Catches literally re-played or looped audio, which transcript features cannot
    see: a spliced repeat produces an identical transcript segment *and* an
    identical waveform, and only this notices the second part.
    """
    try:
        import librosa

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40, hop_length=512)
        logmel = librosa.power_to_db(mel)
        win = max(int(sr / 512), 4)
        frames = [
            logmel[:, i: i + win].mean(axis=1)
            for i in range(0, logmel.shape[1] - win, win // 2)
        ]
        if len(frames) < 4:
            return 0.0
        m = np.stack(frames)
        m = m - m.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.maximum(norm, 1e-8)
        sim = m @ m.T
        # Ignore the diagonal band: adjacent frames are similar by construction.
        for k in range(-2, 3):
            np.fill_diagonal(sim[max(0, k):, max(0, -k):], -1.0)
        return float(sim.max())
    except Exception as exc:  # pragma: no cover
        log.debug("audio self-similarity failed: %s", exc)
        return 0.0


# --------------------------------------------------------------------------- #
# off_topic
# --------------------------------------------------------------------------- #

def off_topic_flag(
    relevance_feats: dict[str, float],
    threshold: float = 35.0,
) -> FlagResult:
    """Response does not address the question.

    Reads the relevance feature block but is calibrated as a **classifier**, not
    derived from the relevance score: the decision boundary and the score curve
    optimise different things, and a borderline-weak answer is not the same
    object as an answer to a different question.

    Default threshold is **35**, not the 55 used by the other flags. Measured on
    36 labelled text pairs (AUC 0.994): at 55 the flag caught only 50% of
    genuinely mismatched answers, while 35 caught 94% with a 0% false-positive
    rate. The asymmetry is real -- off-topic answers score lower than the other
    gaming behaviours because they lack the emphatic signature echo and padding
    have -- and it is why per-flag thresholds are set from curves rather than
    shared as one constant.
    """
    sim_q = relevance_feats.get("sim_q", 0.0)
    elem = relevance_feats.get("element_coverage", 0.0)
    pct_off = relevance_feats.get("pct_windows_offtopic", 0.0)
    min_win = relevance_feats.get("min_window_sim", 0.0)
    vs_ref = relevance_feats.get("window_sim_vs_ref", 1.0)

    feats = {
        "sim_q": sim_q,
        "element_coverage": elem,
        "pct_windows_offtopic": pct_off,
        "min_window_sim": min_win,
        "window_sim_vs_ref": vs_ref,
    }

    signals = {
        "low similarity to question": (_squash(0.70 - sim_q, 0.05, 0.30), 0.30),
        "question elements unaddressed": (_squash(0.75 - elem, 0.05, 0.35), 0.30),
        "much of the answer is off-topic": (_squash(pct_off, 0.2, 0.8), 0.25),
        "far below the reference answer": (_squash(0.95 - vs_ref, 0.05, 0.35), 0.15),
    }
    score = 100.0 * sum(v * w for v, w in signals.values())
    top = sorted(signals.items(), key=lambda kv: -kv[1][0] * kv[1][1])
    evidence = "; ".join(f"{k} ({v:.2f})" for k, (v, _) in top[:3] if v > 0.1) or "on topic"

    return FlagResult("off_topic", score, threshold, feats, evidence)


# --------------------------------------------------------------------------- #
# foreign_language
# --------------------------------------------------------------------------- #

def foreign_language_flag(
    lang_windows: list,
    transcript: str,
    asr_avg_logprob: float = 0.0,
    threshold: float = 30.0,
    transcript_auto: str | None = None,
) -> FlagResult:
    """Non-English speech in the response.

    **The highest-risk flag in the project.** Language-ID models routinely
    misclassify heavily accented English as the speaker's L1 -- Indian-accented
    English as Hindi, Filipino-accented as Tagalog. The client's population is
    *entirely* L2 speakers, so a naive implementation false-positives on exactly
    the candidates it must not, and the failure is invisible unless you
    specifically test for it.

    Mitigations here:
    * the acoustic channel uses **posteriors**, not the argmax label, so a
      moderately confused window contributes proportionally instead of flipping
    * a text channel must corroborate for the score to reach the upper range
    * thresholds are meant to be set against **accented-English negatives**
      (Svarah, FLEURS accented sets), not against native English

    ``transcript_auto`` is a second transcription made **without forcing English**.
    It exists because the scoring transcript is deliberately forced to English, so
    on Hindi audio Whisper emits garbled English-looking text and a text language
    detector reads it as English -- the corroborating channel could never fire.
    Measured consequence before this was added: a response that was **100%**
    Hindi scored only 63.8, and 75% Hindi scored 46.8, below the old default
    threshold of 55. When it is not supplied the text channel abstains (its
    weight collapses) rather than silently voting "English".

    **Measured on accented English**, which is what sets the threshold. Both
    groups below speak English throughout: Indian-accented (Svarah) and US
    (FLEURS en_us), n=20 each.

    Before the corroboration multiplier, acoustic evidence alone gave accented
    English mean 8.5 / max 43.1 and a **16.7% false-positive rate at threshold
    25**, against 0.0% for US English. After it, accented English falls to mean
    3.4 / max 21.9:

    | threshold | FPR, US English | FPR, Indian English | catches (mean over hi/ta/sw) |
    |---|---|---|---|
    | 15 | 0.0% | 15.0% | ~10% foreign |
    | 25 | 0.0% | 0.0% | ~20% foreign |
    | **30** | **0.0%** | **0.0%** | ~20-25% foreign |
    | 45 | 0.0% | 0.0% | ~35% foreign |

    Default **30**: the measured floor for a zero false-positive rate is 25, and
    30 adds margin over the worst observed accented score (21.9) without giving
    up much sensitivity. Suppressing accent false-positives does cost detection
    -- a 20%-foreign response scores 29.3 where it scored 51.0 before -- and that
    trade is deliberate. Flagging a real candidate for their accent is far worse
    than missing a partial code-switch.

    n=20 per group. Re-run at higher n before treating the 0% as settled, note
    that sensitivity is language-dependent (Yoruba is much weaker than Swahili,
    so one global threshold is not equally fair across first languages), and
    route this flag to human review rather than to an automatic fail.
    """
    feats: dict[str, float] = {}

    if lang_windows:
        p_non = np.array([w.p_non_english for w in lang_windows], dtype=np.float32)
        feats["max_p_non_english"] = float(p_non.max())
        feats["mean_p_non_english"] = float(p_non.mean())
        feats["pct_windows_non_english"] = float((p_non > 0.5).mean())
        # Sustained non-English matters; isolated confused windows do not.
        feats["max_run_non_english"] = float(_max_run(p_non > 0.5))
        feats["nonen_duration_s"] = float(
            (p_non > 0.5).sum() * (lang_windows[0].end - lang_windows[0].start)
        )
        langs = [w.top_lang for w in lang_windows if w.p_non_english > 0.5]
        feats["n_distinct_foreign_langs"] = float(len(set(langs)))
    else:
        for k in ("max_p_non_english", "mean_p_non_english", "pct_windows_non_english",
                  "max_run_non_english", "nonen_duration_s", "n_distinct_foreign_langs"):
            feats[k] = 0.0

    # Prefer the unforced transcription for language ID; the forced one can only
    # reveal non-Latin script leaking through, not a genuine language judgement.
    text_for_lid = transcript_auto if transcript_auto else transcript
    feats["text_p_non_english"] = _text_lid_non_english(text_for_lid)
    feats["text_lid_available"] = 1.0 if transcript_auto else 0.0
    feats["asr_avg_logprob"] = asr_avg_logprob

    # Acoustic evidence: what the language-ID posteriors say.
    signals = {
        "sustained non-English audio": (
            _squash(feats["max_run_non_english"], 1.0, 4.0), 0.45),
        "much of the audio is non-English": (
            _squash(feats["mean_p_non_english"], 0.15, 0.65), 0.35),
        "ASR struggled to read it as English": (
            _squash(-asr_avg_logprob, 0.6, 1.1), 0.20),
    }
    acoustic = sum(v * w for v, w in signals.values())

    # Corroboration, applied as a MULTIPLIER rather than another additive vote.
    #
    # Measured fairness result that forced this: on Indian-accented English from
    # Svarah -- genuinely English throughout -- the acoustic channels alone gave
    # mean 8.5, p90 26.8, max 43.1, a 16.7% false-positive rate at threshold 25.
    # US English scored 0.0 across the board. Language-ID models mistake accent
    # for language, and this population is entirely L2 speakers, so acoustic
    # evidence on its own is not safe to act on.
    #
    # As an additive term the text channel could only decline to add points. As a
    # multiplier it can actively suppress: if we re-listen to the spans that
    # sounded foreign and they transcribe as English, that is positive evidence
    # the accent was misread, and the flag should fall rather than merely fail to
    # rise.
    degenerate = _is_degenerate(transcript_auto)
    feats["auto_transcript_degenerate"] = 1.0 if degenerate else 0.0

    if transcript_auto and not degenerate:
        corroboration = _corroboration(feats["text_p_non_english"])
    elif degenerate:
        # The unforced pass hallucinated rather than found a language. On an
        # African-accented speaker it emitted a run of repeated Malayalam
        # characters, which the text detector scored as emphatically non-English
        # and the flag then "corroborated" -- producing 79.5 for someone reading
        # the paragraph fluently.
        #
        # Weighted at 0.35, below the no-evidence case, because degeneracy is
        # itself weak evidence *against* foreign speech: genuinely foreign audio
        # decodes cleanly into its own script, while accented English is what
        # makes the decoder lock up.
        corroboration = 0.35
    else:
        # No suspect spans were worth transcribing, so there is nothing to
        # corroborate; discount rather than trust acoustics outright.
        corroboration = 0.70
    feats["corroboration"] = corroboration

    signals["transcript of the suspect spans is not English"] = (
        _squash(feats["text_p_non_english"], 0.2, 0.7), 0.0)  # reported, not summed
    score = 100.0 * acoustic * corroboration
    top = sorted(signals.items(), key=lambda kv: -kv[1][0] * kv[1][1])
    evidence = "; ".join(f"{k} ({v:.2f})" for k, (v, _) in top[:3] if v > 0.1) or "English throughout"

    return FlagResult("foreign_language", score, threshold, feats, evidence)


# Text-LID probabilities bounding the three regimes below.
TEXT_LID_CONFIDENT_EN = 0.15
TEXT_LID_UNCERTAIN_HI = 0.55


def _corroboration(p_non_english: float) -> float:
    """Multiplier on acoustic evidence, given what the text channel says.

    Deliberately **asymmetric and three-regime**, because the text channel is
    unreliable in both directions and its uncertainty must not be mistaken for a
    verdict:

    * **Confidently English** (p < 0.15) -> 0.12. Positive evidence the accent was
      misread, so collapse the score. This is the Tagalog case: acoustic
      p(non-English) hit 0.94 while the suspect spans transcribed as flawless
      English.
    * **Uncertain** (0.15-0.55) -> ~0.70, the same as having no text evidence at
      all. Absence of evidence is not evidence of absence.
    * **Confidently non-English** (p > 0.55) -> 1.0. Corroborates the acoustics.

    The first version interpolated linearly from 0.10, which made *uncertainty*
    suppressive. Measured cost: a response spoken entirely in Tamil fell to 46.7
    and 100%-foreign overall dropped from 95.1 to 73.0, because the text detector
    is often merely unsure about transcribed non-Latin script. Fairness bought by
    refusing to detect anything is not fairness.
    """
    p = float(np.clip(p_non_english, 0.0, 1.0))
    if p <= TEXT_LID_CONFIDENT_EN:
        return 0.12
    if p >= TEXT_LID_UNCERTAIN_HI:
        return 1.0
    if p <= 0.30:
        # 0.15 -> 0.30 : climb out of "confidently English" to neutral
        return 0.12 + (0.70 - 0.12) * (p - TEXT_LID_CONFIDENT_EN) / (0.30 - TEXT_LID_CONFIDENT_EN)
    # 0.30 -> 0.55 : neutral up to full corroboration
    return 0.70 + (1.0 - 0.70) * (p - 0.30) / (TEXT_LID_UNCERTAIN_HI - 0.30)


def _is_degenerate(text: str | None) -> bool:
    """Is this transcript a decoding failure rather than a language?

    Whisper, asked to auto-detect the language of heavily accented English,
    sometimes locks onto a script and emits a long run of one character. That
    string scores as emphatically non-English on any text detector, so without
    this check a hallucination becomes the strongest possible corroboration.

    Observed instance: a single character repeated ~70 times, scoring the flag
    at 79.5 for a speaker reading the standard English paragraph fluently.
    """
    if not text:
        return False
    stripped = "".join(text.split())
    if len(stripped) < 8:
        return True

    # A single character taking more than half the string is a decoding lock-up.
    # This is the check that actually works, and it is length-robust: measured
    # 0.077-0.236 on genuine Tamil/Hindi/Tagalog/Swahili against ~1.0 for the
    # repeated-character hallucination.
    counts: dict[str, int] = {}
    for ch in stripped:
        counts[ch] = counts.get(ch, 0) + 1
    if max(counts.values()) / len(stripped) > 0.5:
        return True

    # Word-level loop: Whisper sometimes repeats a phrase for the rest of the
    # window. Measured 0.67-0.93 distinct-word ratio on genuine transcripts.
    words = text.split()
    if len(words) >= 12 and len(set(words)) / len(words) < 0.25:
        return True

    return False

    # REMOVED: a character-diversity check, `len(set(s))/len(s) < 0.12`.
    #
    # It is a length artefact, not a degeneracy measure. Distinct characters are
    # bounded by the alphabet while the denominator grows with the text, so any
    # long passage scores low by arithmetic. Measured on genuine 340-450
    # character transcripts: Tamil 0.080, Tagalog 0.086, Hindi 0.095, Swahili
    # 0.111. Three of the four were rejected as "hallucinations" and Swahili
    # passed by 0.011 of luck.
    #
    # The consequence was severe and language-dependent: rejecting the text
    # channel dropped corroboration to 0.35, so a response spoken *entirely* in
    # Tagalog scored 28 and fired 0 times out of 8, while Yoruba -- which
    # happened to clear the threshold -- fired 8 out of 8.
    #
    # This is the same mistake the lexical module explicitly avoids by excluding
    # raw type-token ratio, documented there as "a length artefact, not a
    # vocabulary measure". I wrote a character-level TTR as a guard and did not
    # recognise it.


def _max_run(mask: np.ndarray) -> int:
    """Longest run of True values."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _text_lid_non_english(text: str) -> float:
    """Probability the transcript is not English, from a text LID model."""
    if not text or len(text.split()) < 4:
        return 0.0
    try:
        from lingua import Language, LanguageDetectorBuilder

        detector = _lingua_detector()
        conf = detector.compute_language_confidence_values(text)
        for c in conf:
            if c.language == Language.ENGLISH:
                return float(max(0.0, 1.0 - c.value))
        return 1.0
    except Exception as exc:  # pragma: no cover
        log.debug("text LID failed: %s", exc)
        return 0.0


_DETECTOR = None


def _lingua_detector():
    """Lingua detector restricted to languages plausible for this population.

    Restricting the candidate set matters: an unrestricted detector will happily
    propose Scots or Afrikaans for accented English transcripts, inflating the
    non-English probability for exactly the speakers we must not penalise.
    """
    global _DETECTOR
    if _DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder

        langs = [
            Language.ENGLISH, Language.HINDI, Language.BENGALI, Language.TAMIL,
            Language.TELUGU, Language.MARATHI, Language.GUJARATI, Language.PUNJABI,
            Language.URDU, Language.TAGALOG, Language.SWAHILI, Language.YORUBA,
            Language.ARABIC, Language.FRENCH, Language.SPANISH,
        ]
        _DETECTOR = LanguageDetectorBuilder.from_languages(*langs).build()
    return _DETECTOR
