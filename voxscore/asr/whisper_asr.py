"""Whisper transcription and windowed language identification.

Two jobs, deliberately in one module because they share an expensive encoder:

1. Transcribe the response (feeds grammar, lexical, relevance, every flag).
2. Produce a per-window language posterior trajectory, which is the primary
   channel for the foreign-language flag.

Language ID is implemented by hand -- encode a window, decode a single step from
``<|startoftranscript|>``, and softmax over the language token ids -- rather than
through a helper method. That keeps it stable across transformers versions and,
more importantly, gives us the full posterior instead of just an argmax. The
client asked for a tunable score, and a hard language label cannot provide one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from voxscore.config import SAMPLE_RATE, PipelineConfig, get_device, get_dtype

log = logging.getLogger(__name__)

WHISPER_CHUNK_S = 30.0  # Whisper's fixed receptive field.


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class LangWindow:
    """Language posterior for one audio window."""

    start: float
    end: float
    top_lang: str
    top_prob: float
    p_english: float

    @property
    def p_non_english(self) -> float:
        return 1.0 - self.p_english


@dataclass
class ASRResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    language_prob: float = 0.0
    lang_windows: list[LangWindow] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    model_id: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# --------------------------------------------------------------------------- #

@lru_cache(maxsize=2)
def _load(model_id: str, device_str: str, dtype_str: str):
    """Load and cache processor + model. Cached because loading large-v3 is slow."""
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)

    processor = AutoProcessor.from_pretrained(model_id)
    try:
        # transformers >= 5 renamed torch_dtype -> dtype.
        model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)

    model = model.to(device).eval()
    return processor, model


class WhisperASR:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.device = get_device(self.cfg.device)
        self.dtype = get_dtype(self.device)
        self.model_id = self.cfg.asr_model
        self._processor = None
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._processor, self._model = _load(
                self.model_id, str(self.device), str(self.dtype).split(".")[-1]
            )
        return self._processor, self._model

    # ---------------------------------------------------------------- #
    # Transcription
    # ---------------------------------------------------------------- #

    def transcribe(self, audio: np.ndarray, language: str | None = "en") -> ASRResult:
        """Transcribe a response.

        ``language`` defaults to English on purpose. We want the *English reading*
        of the audio for the scoring path -- forcing English means a Hindi stretch
        produces visibly broken output rather than a clean Hindi transcript that
        the grammar and lexical modules would then score as if it were an answer.
        The foreign-language flag is handled separately by :meth:`language_windows`,
        which sees the true posteriors.

        The short/long branch below is not cosmetic. Whisper's encoder takes a
        fixed 3000-frame (30 s) window. Passing ``padding="longest",
        truncation=False`` on *short* audio yields 585 frames instead of 3000 and
        pushes generation into the sequential long-form algorithm on input it was
        never meant to see -- which hangs indefinitely when combined with beam
        search. Measured cost of getting this wrong: a >10 minute stall on 6 s of
        audio. Client responses straddle the 30 s boundary, so both paths are live.
        """
        processor, model = self._ensure()
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / SAMPLE_RATE

        gen_kwargs: dict = {}
        if language:
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"

        if duration <= WHISPER_CHUNK_S:
            # Short form: let the processor pad to the full 30 s window.
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            feats = inputs.input_features.to(self.device, self.dtype)
            gen_kwargs["num_beams"] = self.cfg.asr_beam_size
        else:
            # Long form: transformers requires an attention mask and timestamps
            # to drive sequential decoding. Beam search is not supported here.
            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
                padding="longest",
                truncation=False,
            )
            feats = inputs.input_features.to(self.device, self.dtype)
            gen_kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
            gen_kwargs["return_timestamps"] = True
            gen_kwargs["num_beams"] = 1

        with torch.inference_mode():
            out = model.generate(feats, **gen_kwargs)

        seq = out.sequences if hasattr(out, "sequences") else out
        text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()

        if duration <= WHISPER_CHUNK_S:
            avg_lp = self._sequence_confidence(model, processor, feats, seq, language)
        else:
            # Long-form output spans several encoder windows, so teacher-forcing
            # the whole transcript against one 30 s window scores most tokens
            # against audio the decoder never saw. Measured effect: a clean 40 s
            # item reported confidence low enough to trip the "not English"
            # signal. Estimate from a single window instead, and say so.
            avg_lp = self._confidence_sample(processor, model, audio, language)

        return ASRResult(
            text=text,
            segments=[],  # derived from forced alignment, not from Whisper
            avg_logprob=avg_lp,
            model_id=self.model_id,
        )

    def _sequence_confidence(self, model, processor, feats, seq, language) -> float:
        """Mean per-token log-probability of the transcript.

        Two traps here, both found by measurement rather than by reading docs:

        1. ``generate(output_scores=True)`` is **silently ignored** on the Whisper
           path in transformers 5.x -- it warns that the flag "is not valid and
           may be ignored" and returns no scores. Reading ``out.scores`` would
           have yielded exactly 0.0 for every item, forever, without erroring.
        2. ``generate`` **strips the forced decoder prefix** from the returned
           sequence, which begins directly at the first text token. Teacher
           forcing on it therefore runs a decoder that has been told neither its
           task nor its language, and scores near chance. Measured: -8.2 on a
           perfectly transcribed clip, against -7.8 for a *silent audio* control.
           The prefix must be restored before scoring.

        A genuinely low value means the audio is not cleanly the language we
        forced, making this both a quality diagnostic and a corroborating channel
        for the foreign-language flag -- but only once both traps are avoided.
        """
        try:
            if seq.shape[1] < 1:
                return 0.0
            # Long-form generation can span several encoder windows; confidence is
            # only meaningful against the window actually encoded.
            if feats.shape[-1] > 3000:
                feats = feats[..., :3000]

            prefix = self._decoder_prefix(processor.tokenizer, model, language)
            if not prefix:
                return 0.0
            pre = torch.tensor([prefix], dtype=seq.dtype, device=seq.device)
            full = torch.cat([pre, seq], dim=1)

            with torch.inference_mode():
                logits = model(input_features=feats, decoder_input_ids=full[:, :-1]).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            targets = full[:, 1:]
            gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]

            # Score only the generated text, not the prefix tokens themselves:
            # those are near-deterministic and would bias the mean toward 0
            # regardless of how poor the audio was.
            scored = gathered[len(prefix) - 1:]
            if scored.numel() == 0:
                return 0.0
            return float(scored.mean())
        except Exception as exc:  # pragma: no cover
            log.debug("confidence computation failed: %s", exc)
            return 0.0

    def _confidence_sample(self, processor, model, audio, language, window_s: float = 30.0):
        """Confidence estimated from the first 30 s, transcribed short-form.

        Costs one extra generate call on a single window. Acceptable here because
        the client processes in batch with no latency requirement, and the
        alternative is a confidence number that is quietly wrong on every
        response longer than 30 s -- which is most of them.
        """
        try:
            clip = np.asarray(audio[: int(window_s * SAMPLE_RATE)], dtype=np.float32)
            if len(clip) < SAMPLE_RATE:
                return 0.0
            inputs = processor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            feats = inputs.input_features.to(self.device, self.dtype)
            kw = {"num_beams": 1}
            if language:
                kw["language"] = language
                kw["task"] = "transcribe"
            with torch.inference_mode():
                seq = model.generate(feats, **kw)
            seq = seq.sequences if hasattr(seq, "sequences") else seq
            return self._sequence_confidence(model, processor, feats, seq, language)
        except Exception as exc:  # pragma: no cover
            log.debug("sampled confidence failed: %s", exc)
            return 0.0

    @staticmethod
    def _decoder_prefix(tok, model, language: str | None) -> list[int]:
        """Rebuild the forced decoder prefix that ``generate`` strips off."""
        def tid(name: str) -> int | None:
            t = tok.convert_tokens_to_ids(name)
            return t if t is not None and t >= 0 else None

        gc = model.generation_config
        sot = getattr(gc, "decoder_start_token_id", None) or tid("<|startoftranscript|>")
        if sot is None:
            return []
        prefix = [int(sot)]
        for name in (f"<|{language or 'en'}|>", "<|transcribe|>", "<|notimestamps|>"):
            t = tid(name)
            if t is not None:
                prefix.append(int(t))
        return prefix

    def transcribe_auto(self, audio: np.ndarray) -> ASRResult:
        """Transcribe with Whisper choosing the language itself."""
        return self.transcribe(audio, language=None)

    def transcribe_suspect_spans(
        self,
        audio: np.ndarray,
        lang_windows: list[LangWindow],
        p_threshold: float = 0.5,
        pad_s: float = 0.4,
    ) -> str:
        """Transcribe **only** the windows the acoustic model reads as non-English.

        Transcribing the whole clip with auto language detection does not work
        for *partial* code-switching, and measurement showed why: on a response
        that was 25% Hindi, Whisper picked English for the clip -- correctly, it
        is the majority language -- and the text detector then voted "English",
        actively lowering the flag score from 35.2 to 25.6. The corroborating
        channel was arguing against the evidence.

        Cutting to the suspect spans first asks the right question: *is the part
        that sounded foreign actually foreign?* On a fully English response no
        spans qualify and this returns "", so the channel abstains.
        """
        if not lang_windows:
            return ""
        suspect = [w for w in lang_windows if w.p_non_english > p_threshold]
        if not suspect:
            return ""

        # Merge overlapping/adjacent windows into contiguous spans.
        spans: list[list[float]] = []
        for w in sorted(suspect, key=lambda x: x.start):
            s, e = max(0.0, w.start - pad_s), w.end + pad_s
            if spans and s <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])

        pieces = []
        for s, e in spans:
            lo, hi = int(s * SAMPLE_RATE), min(int(e * SAMPLE_RATE), len(audio))
            if hi - lo >= SAMPLE_RATE:
                pieces.append(audio[lo:hi])
        if not pieces:
            return ""

        try:
            return self.transcribe(np.concatenate(pieces), language=None).text
        except Exception as exc:  # pragma: no cover
            log.debug("suspect-span transcription failed: %s", exc)
            return ""

    # ---------------------------------------------------------------- #
    # Windowed language identification
    # ---------------------------------------------------------------- #

    def language_windows(self, audio: np.ndarray) -> list[LangWindow]:
        """Language posteriors over sliding windows.

        Returns one :class:`LangWindow` per window, carrying ``p_english``
        rather than a label, so the foreign-language flag can be a continuous
        score with a threshold the client chooses later.
        """
        processor, model = self._ensure()
        tok = processor.tokenizer

        lang_ids, lang_codes = self._language_token_table(tok)
        if not lang_ids:
            log.warning("no language tokens found; skipping windowed LID")
            return []

        sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        win = int(self.cfg.lid_window_s * SAMPLE_RATE)
        hop = int(self.cfg.lid_hop_s * SAMPLE_RATE)
        audio = np.asarray(audio, dtype=np.float32)

        starts = list(range(0, max(len(audio) - win + 1, 1), hop)) or [0]
        chunks = [audio[s: s + win] for s in starts]

        results: list[LangWindow] = []
        batch = 8
        for i in range(0, len(chunks), batch):
            group = chunks[i: i + batch]
            feats = processor(
                group, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(self.device, self.dtype)

            dec = torch.full((feats.shape[0], 1), sot, dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = model(feats, decoder_input_ids=dec).logits[:, 0].float()

            # Renormalise over language tokens only: the raw softmax includes
            # thousands of irrelevant tokens and would understate every language.
            lang_logits = logits[:, lang_ids]
            probs = torch.softmax(lang_logits, dim=-1).cpu().numpy()

            en_idx = lang_codes.index("en") if "en" in lang_codes else None
            for j, p in enumerate(probs):
                k = int(p.argmax())
                s = starts[i + j] / SAMPLE_RATE
                results.append(
                    LangWindow(
                        start=s,
                        end=s + self.cfg.lid_window_s,
                        top_lang=lang_codes[k],
                        top_prob=float(p[k]),
                        p_english=float(p[en_idx]) if en_idx is not None else 0.0,
                    )
                )
        return results

    @staticmethod
    def _language_token_table(tok) -> tuple[list[int], list[str]]:
        """Map Whisper language tokens to ids, skipping any the tokenizer lacks."""
        try:
            from transformers.models.whisper.tokenization_whisper import LANGUAGES
            codes = list(LANGUAGES.keys())
        except Exception:
            codes = ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa",
                     "ur", "tl", "sw", "yo", "ha", "ar", "fr", "es", "zh"]

        ids, kept = [], []
        for c in codes:
            tid = tok.convert_tokens_to_ids(f"<|{c}|>")
            if tid is not None and tid >= 0 and tid != tok.unk_token_id:
                ids.append(tid)
                kept.append(c)
        return ids, kept
