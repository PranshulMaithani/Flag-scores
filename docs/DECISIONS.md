# Decision Log

Architecture decision records. Newest at the bottom. Each records the constraint that
forced it, what was rejected, and what would make us revisit.

---

## ADR-001 — Project venv inherits `mlenv` via `.pth`, never mutates it
**Date:** 2026-09-11 · **Status:** accepted

`mlenv` (`C:\Users\prans\mlenv`) holds a working `torch 2.9.1+rocm7.2.1` build — a
non-trivial thing to reproduce on Windows+AMD. We must not break it.

`python -m venv --system-site-packages` does **not** work here: `mlenv` is itself a
venv, so "system" resolves to the base interpreter, not to `mlenv`. Torch was invisible.

**Decision:** create `.venv` normally, then drop `_mlenv_inherit.pth` into
`.venv/Lib/site-packages` containing mlenv's site-packages path. Project deps install
into `.venv` and *shadow* mlenv (venv path index 6 < mlenv index 7), so mlenv is
read-only from our side.

**Rejected:** installing project deps directly into `mlenv` (risks breaking the client's
working ROCm build); reinstalling ~3GB of ROCm torch into `.venv` (slow, duplicated).

**Incident:** first attempt used `site.getsitepackages()[-1]`, which resolved to the
**base Python install** and wrote the `.pth` machine-wide. Caught and reverted; base
interpreter verified clean. Lesson: on a `--system-site-packages` venv,
`getsitepackages()` returns multiple entries and `[-1]` is the *base*, not the venv.

---

## ADR-002 — Grammar via GEC model + ERRANT, not LanguageTool
**Date:** 2026-09-11 · **Status:** accepted

There is no Java runtime on this machine, and the client's office laptop is locked
down and off-limits (C8). `language-tool-python` requires a JVM.

**Decision:** primary grammar signal is a small seq2seq grammatical-error-correction
model, with **ERRANT** (MIT) aligning original↔corrected to produce *typed* error
counts (article, preposition, agreement, tense, …). Pure Python, no JVM, and the typed
breakdown directly satisfies the explainability requirement.

**Rejected:** LanguageTool as primary — undeployable without Java on the target box.
Retained as an optional dev-only corroborating signal if a JVM ever appears.

**Revisit if:** the GEC model proves to over-flag L2 spoken English so badly that
rule-based checking is more precise.

---

## ADR-003 — No generative LLM; relevance built from embeddings + NLI
**Date:** 2026-09-11 · **Status:** accepted

Client stated the no-LLM constraint twice, then said "your choice" when pressed. We
honour the stated constraint rather than exploit the ambiguity.

**Decision:** relevance is computed from (a) bi-encoder embedding similarity to the
question and to the 3 ideal answers, (b) NLI entailment against ideal-answer claims,
(c) content-word coverage, (d) similarity trajectory across the response (topic drift).
No free-form generative judging.

**Note for the record:** a 7B LLM in 4-bit fits in ~5GB and *would* run inside the 16GB
production budget — the client's stated reason ("won't fit in 16GB") is not actually
correct. Flagged to the client rather than silently acted upon. The relevance module is
structured so an LLM judge can be A/B'd as a drop-in later if we plateau below the bar.

---

## ADR-004 — Whisper large-v3 as ASR, not Parakeet
**Date:** 2026-09-11 · **Status:** provisional, pending benchmark

**Decision:** Whisper large-v3 (MIT).

Reasons beyond raw WER:
- **Multilingual.** The `foreign_language` flag needs the system to actually recognise
  and transcribe Hindi/Tagalog/Swahili when it appears. An English-only model just
  emits garbage English, which is a much weaker signal.
- Built-in language identification, usable per-window for a *continuous* foreign-language
  score (client explicitly asked for a tunable score, not a binary).
- MIT licensed, unambiguously commercial-safe (C5).

**Rejected:** NVIDIA Parakeet TDT 0.6B (CC-BY-4.0, excellent English WER) — English-only,
and NeMo on ROCm is painful (C6).

**Revisit:** benchmark both on accented English; Parakeet may win as a *second* ASR pass
for the English-only scoring path while Whisper handles LID.

---

## ADR-005 — Word timings from CTC forced alignment, not Whisper timestamps
**Date:** 2026-09-11 · **Status:** accepted

Fluency scoring is almost entirely a function of **pause boundaries**. Whisper's native
timestamps are segment-level and drift badly; using them would put noise directly into
the highest-weight fluency features.

**Decision:** `torchaudio.functional.forced_align` (built-in CTC aligner, MIT) against a
wav2vec2 acoustic model, to get word-level start/end times from the Whisper transcript.

**Rejected:** WhisperX — depends on `faster-whisper`/CTranslate2, which has **no ROCm
backend** (C6). Montreal Forced Aligner — heavyweight install, needs a pronunciation
dictionary that will mishandle our accent distribution.

---

## ADR-006 — Relevance matches discourse moves, not content, on personal prompts
**Date:** 2026-09-11 · **Status:** accepted · **Highest-impact decision so far**

The client supplied a real prompt: *"Talk about a friend you were close with but later
lost touch."* The 3 ideal answers are ~140-word C1/C2 responses — full of personal
specifics (a name, a city, a year) that no other candidate will ever reproduce.

Direct similarity between a response and an ideal answer is therefore dominated by
*whose story it is*, not by whether the response answers the question.

**Decision:** derive a per-question relevance rubric automatically —
1. entity-mask the 3 ideal answers,
2. abstract each into predicate–argument discourse moves,
3. keep a move only if it appears in **>= 2 of 3** ideal answers (present in one only ⇒
   personal idiosyncrasy, discard),
4. decompose the question itself by dependency parse into required elements,
5. score coverage of that rubric, on entity-masked representations throughout.

Three independent ideal answers are exactly enough to separate the required from the
idiosyncratic. This is the asset the client already bought for Carnegie and that
Carnegie appears not to have exploited.

**Why this is the opening:** grammar/lexical/fluency are properties of the response
alone and need no question conditioning — hence the incumbent managing 0.60 on all
three. Relevance is the only category requiring reasoning about the prompt, and the
obvious implementation is nearly uncorrelated with relevance on personal-narrative
prompts. 0.40 is about what that would yield. Hypothesis, not proven — but it predicts
the exact 0.6/0.6/0.6/0.4 pattern observed, which is a strong sign.

**Revisit if:** measured relevance fails to exceed the `sim_q` baseline we retain
specifically to test this.

---

## ADR-007 — Question family handled by a continuous blend, not a classifier
**Date:** 2026-09-11 · **Status:** accepted

Client later clarified there are multiple question types — personal/experiential
(*"talk about a friend…"*) **and** opinion/argumentative (*"do you think technological
advancements made humans more dependent?"*). Ideal-answer content is reusable for the
second family and not for the first, so ADR-006 must not be applied blindly.

**Decision:** no question-type classifier and no hand labelling. Compute

```
shareability = mean pairwise entity-masked content overlap(I1, I2, I3)
```

Three model answers to an opinion prompt converge on the same arguments; three about a
lost friend share only structure. So the overlap *is* the measurement of whether content
matching is valid. Both feature blocks (narrative moves, argumentative structure) are
computed for every item and blended continuously by `shareability`.

**Rejected:** a discrete question-type classifier — brittle, needs labels we do not
have, and fails silently on unseen question types. The continuous blend degrades
gracefully instead.

**Bonus property:** `shareability` is a zero-supervision, per-question diagnostic the
client can inspect directly, computed from assets they already own.

---

## ADR-008 — Output format is XML
**Date:** 2026-09-11 · **Status:** accepted

Client: *"leave carnegie you can make xml output with all your scores flag and
everything."* No requirement to mirror the incumbent schema.

**Decision:** emit XML carrying, per item — the four 0–100 scores, each with its
contributing feature values; the four flag scores with recommended thresholds and the
fired/not-fired state at that threshold; quality diagnostics; the transcript; and model
and version provenance. A JSON view of the identical structure ships alongside, since
downstream consumers usually prefer it and the cost is one serializer.

Because every score carries its features, the XML is self-explaining — which is how we
satisfy the candidate-appeals requirement (brief §21) without a separate report path.
