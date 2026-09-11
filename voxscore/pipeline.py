"""End-to-end scoring pipeline.

Stage order matters: ASR and alignment are expensive and everything downstream
depends on them, so they run once per item and are cached. Feature extraction is
pure and cheap, which means the whole feature set can be recomputed -- during the
many iterations this project will need -- without re-running a single model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from voxscore import __version__
from voxscore.asr.align import align_words
from voxscore.asr.whisper_asr import WhisperASR
from voxscore.config import MODELS, PipelineConfig, device_report
from voxscore.features.embed import NLI, Embedder
from voxscore.features.fluency import fluency_features
from voxscore.features.grammar import GrammarScorer, grammar_features
from voxscore.features.lexical import lexical_features
from voxscore.features.quality import quality_features, quality_warnings
from voxscore.features.relevance import RelevanceRubric, build_rubric, relevance_features
from voxscore.flags.detectors import (
    foreign_language_flag,
    off_topic_flag,
    prompt_read_flag,
    repetition_flag,
)
from voxscore.scoring.aggregate import explain, score_all
from voxscore.utils import textproc as tp
from voxscore.utils.audio_io import AudioItem
from voxscore.xml_out import ItemResult

log = logging.getLogger(__name__)

# Acoustic mean p(non-English) above which a second, unforced transcription is
# worth running. Low enough to catch partial code-switching, high enough that
# clean English responses (measured mean p(non-en) < 0.02) never trigger it.
AUTO_TRANSCRIBE_THRESHOLD = 0.12


@dataclass
class Stage1:
    """Cached expensive output: transcription, timings, language posteriors."""

    transcript: str
    words: list
    lang_windows: list
    avg_logprob: float
    asr_seconds: float
    transcript_auto: str | None = None


class Pipeline:
    """Scores responses. Models load lazily and are reused across items."""

    def __init__(self, cfg: PipelineConfig | None = None, load_grammar: bool = True):
        self.cfg = cfg or PipelineConfig()
        self.nlp = tp.get_nlp()
        self.asr = WhisperASR(self.cfg)
        self.embedder = Embedder()
        self.nli = NLI()
        self.grammar = GrammarScorer() if load_grammar else None
        self._rubrics: dict[str, RelevanceRubric] = {}

    # ------------------------------------------------------------------ #

    def rubric_for(self, question_id: str, question_text: str,
                   ideal_answers: list[str] | None) -> RelevanceRubric:
        """Build (once) and cache the relevance rubric for a question."""
        if question_id not in self._rubrics:
            t0 = time.perf_counter()
            self._rubrics[question_id] = build_rubric(
                question_id, question_text, ideal_answers, self.embedder, self.nlp
            )
            log.info("rubric for %s built in %.2fs", question_id, time.perf_counter() - t0)
        return self._rubrics[question_id]

    def run_stage1(self, audio: np.ndarray) -> Stage1:
        """Transcribe, align, and read language posteriors."""
        t0 = time.perf_counter()
        asr = self.asr.transcribe(audio, language="en")
        words = align_words(audio, asr.text) if asr.text.strip() else []
        lang_windows = self.asr.language_windows(audio)

        # Corroborating pass for the foreign-language flag, run only when the
        # acoustic channel is already suspicious. It transcribes the *suspect
        # spans* rather than the whole clip: with auto language detection over
        # the full response, Whisper picks the majority language, so a partly
        # Hindi answer transcribes as English and the text channel argues
        # against the evidence it was meant to corroborate.
        transcript_auto = None
        if lang_windows:
            mean_non_en = float(np.mean([w.p_non_english for w in lang_windows]))
            if mean_non_en > AUTO_TRANSCRIBE_THRESHOLD:
                try:
                    txt = self.asr.transcribe_suspect_spans(audio, lang_windows)
                    transcript_auto = txt or None
                except Exception as exc:
                    log.warning("suspect-span pass failed: %s", exc)

        return Stage1(
            transcript=asr.text,
            words=words,
            lang_windows=lang_windows,
            avg_logprob=asr.avg_logprob,
            asr_seconds=time.perf_counter() - t0,
            transcript_auto=transcript_auto,
        )

    def score_item(self, item: AudioItem) -> ItemResult:
        """Score one response end to end."""
        s1 = self.run_stage1(item.audio)
        parsed = tp.parse(s1.transcript, self.nlp)

        qual = quality_features(item.audio, s1.transcript, s1.avg_logprob)
        qwarn = quality_warnings(qual, s1.transcript)

        rubric = self.rubric_for(
            item.question_id or "unknown",
            item.question_text or "",
            item.ideal_answers,
        )

        # Feature blocks. Relevance runs first because two flags read its output.
        rel_f = relevance_features(parsed, rubric, self.embedder, self.nli)
        flu_f = fluency_features(
            s1.words, parsed, item.duration_s,
            self.cfg.short_pause_s, self.cfg.long_pause_s,
        )
        lex_f = lexical_features(parsed)
        gra_f = (
            grammar_features(parsed, self.grammar, variant="spoken")
            if self.grammar else {}
        )

        flags = [
            prompt_read_flag(parsed, item.question_text or "", rel_f),
            repetition_flag(parsed, self.embedder, item.audio),
            off_topic_flag(rel_f),
            foreign_language_flag(s1.lang_windows, s1.transcript, s1.avg_logprob,
                                  transcript_auto=s1.transcript_auto),
        ]

        scores = score_all(gra_f, lex_f, flu_f, rel_f, qual)
        feats_by_cat = {
            "grammar": gra_f, "lexical": lex_f,
            "fluency": flu_f, "relevance": rel_f,
        }
        explanations = {
            name: explain(cs, feats_by_cat[name])
            for name, cs in scores.items() if feats_by_cat.get(name)
        }

        return ItemResult(
            item_id=item.item_id,
            question_id=item.question_id,
            question_text=item.question_text,
            transcript=s1.transcript,
            duration_s=item.duration_s,
            scores=scores,
            flags=flags,
            features=feats_by_cat,
            quality=qual,
            quality_warnings=qwarn,
            audio_warnings=list(item.warnings_ or []),
            explanations=explanations,
            rubric_summary=rubric.summary(),
            provenance={
                "voxscore_version": __version__,
                "asr_model": MODELS["asr"].hf_id,
                "embedder": MODELS["embedder"].hf_id,
                "nli": MODELS["nli"].hf_id,
                "gec": MODELS["gec"].hf_id if self.grammar else "disabled",
                "device": device_report(),
                "asr_seconds": f"{s1.asr_seconds:.2f}",
            },
        )

    def score_batch(self, items: list[AudioItem]) -> list[ItemResult]:
        """Score many responses, continuing past individual failures.

        One bad file must not abort a batch run -- the client processes in bulk
        and would otherwise lose the whole job to a single corrupt item.
        """
        out: list[ItemResult] = []
        for i, item in enumerate(items, 1):
            try:
                out.append(self.score_item(item))
                log.info("scored %d/%d: %s", i, len(items), item.item_id)
            except Exception as exc:
                log.exception("item %s failed: %s", item.item_id, exc)
                out.append(ItemResult(
                    item_id=item.item_id,
                    question_id=item.question_id,
                    duration_s=item.duration_s,
                    quality_warnings=[f"scoring failed: {type(exc).__name__}: {exc}"],
                ))
        return out
