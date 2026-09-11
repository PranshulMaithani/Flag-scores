# voxscore

Self-hosted scoring for spoken open-ended assessment responses. Replaces a
third-party vendor API with a local pipeline that emits four category scores,
four gaming flags, and the features behind every number.

**Status:** research prototype. Validated on public and synthetic data; awaiting
validation against the client's held-out labelled set.

---

## What it does

**In:** a candidate's spoken answer (`.npy`), the question text, and optionally
three ideal answers for that question.

**Out:** XML (and JSON) carrying

| | |
|---|---|
| **Scores**, 0–100 | grammar · lexical · fluency · relevance |
| **Flags**, 0–100 continuous | `prompt_read` · `repetition` · `off_topic` · `foreign_language` |
| **Quality** | SNR, clipping, speech presence, ASR confidence, **scorability** |
| **Explanations** | one sentence per score, naming its largest contributors |
| **Features** | all 84, so nothing is a black box |

Flags are continuous with recommended thresholds, never hard verdicts — the
operating point is the client's to set from the shipped ROC curves.

## Constraints this was built under

- **No generative LLM anywhere.** Encoders, embeddings, NLI, ASR and one small
  task-specific seq2seq only.
- **No external API.** Fully self-hosted.
- **Commercial licensing.** Every shipped model is Apache-2.0 or MIT; see
  `reports/LICENCE_AUDIT.md`, generated live from the Hub.
- **Zero client training data.** Nothing is fitted to client labels; the default
  scorer is a transparent scorecard that cannot overfit.
- **≤16 GB VRAM, batch processing.** Measured 3.1 GB peak, ~1× realtime end to end.

## Quick start

```bash
# 1. environment (inherits torch from an existing ML venv; see docs/DECISIONS.md ADR-001)
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m spacy download en_core_web_sm

# 2. convert audio on the machine where the wavs live
python -c "from voxscore.utils.audio_io import wav_to_npy; wav_to_npy('a.wav','out/',question_id='birthday')"

# 3. score
PYTHONPATH=. .venv/Scripts/python.exe scripts/smoke_pipeline.py
```

### Data contract

```
<item_id>.npy     float32, mono, 16 kHz, range [-1, 1]
<item_id>.json    {item_id, question_id, orig_sr, duration_s}
questions.json    {question_id: {text, ideal_answers: [str, str, str]}}
```

The loader is deliberately forgiving — int16 or float32, mono or stereo, any
sample rate — and reports every coercion rather than performing it silently. A
silent sample-rate error would corrupt every duration-based feature (the whole
fluency category) while producing plausible numbers.

## Layout

```
voxscore/
  config.py            paths, device (ROCm dev / CUDA prod), licence-tagged model registry
  pipeline.py          orchestration; stage 1 cached, features cheap to recompute
  xml_out.py           XML + JSON serialisation
  asr/
    whisper_asr.py     duration-branched transcription, windowed language posteriors
    align.py           own CTC Viterbi forced aligner (see ADR-005)
  features/
    relevance.py       question-derived rubric  <- the core of the project
    fluency.py  lexical.py  grammar.py  quality.py  embed.py
  flags/detectors.py   four gaming detectors, continuous scores
  scoring/aggregate.py scorecard: features -> 0-100, no fitted weights
  eval/synth.py        synthesises labelled edge cases with continuous severity
scripts/               smoke tests, probes, evaluations, licence audit
docs/                  brief, design, decisions (ADRs), engineering journal
reports/               generated results
```

## Reading order

1. `docs/PROJECT_BRIEF.md` — the problem and the constraints
2. `docs/DESIGN.md` — the feature specification
3. `docs/DECISIONS.md` — ADRs, each with what was rejected and why
4. `docs/JOURNAL.md` — what was tried, what broke, what the numbers said

## The central idea

Relevance is the only category requiring reasoning about the prompt, and it is
the one the incumbent scores worst (~0.40 against ~0.60 elsewhere). The obvious
implementation — embed the response, embed the ideal answers, take cosine — is
close to *uncorrelated* with relevance on personal-narrative prompts, because it
is dominated by whose story it is. The client's three ideal answers for "how did
you celebrate your most recent birthday" describe a family dinner, a surprise
party at a café, and a solo day ending with a sister's dinner. Content overlap is
near zero. All three are perfect answers.

So we do not compare content. We derive a rubric per question: required elements
from the question itself, required *discourse moves* from whatever ≥2 of the 3
ideal answers agree on, and a structural profile of what a good answer looks
like. Everything is matched on entity-masked text, so differing personal detail
costs nothing.

How far to trust content matching is decided per question by measuring how much
the ideal answers agree with each other — high for opinion prompts where model
answers converge on the same arguments, near zero for personal prompts where they
share only structure. No question-type labels, no classifier.

## Measured so far

| | result |
|---|---|
| grammar, ordinal validity | Spearman **0.899** (94.4% pairwise) |
| lexical, ordinal validity | Spearman **0.856** (94.4% pairwise) |
| `prompt_read` / `repetition` / `off_topic` | AUC **1.000**, 0% FPR |
| relevance, on-topic vs off-topic | AUC **1.000** |
| throughput | ~1× realtime, 3.1 GB VRAM |

See `docs/JOURNAL.md` for what these do and do not establish. The headline caveat:
relevance discrimination is validated, fine-grained relevance *ranking* among
genuinely on-topic answers is not, and that is what the client's 0–5 labels encode.

## Known gaps

- **`foreign_language` is not yet safe to fail candidates on.** Language-ID models
  misread accented English as the speaker's L1, and the population is entirely L2
  speakers. Calibration against accented-English negatives is in progress.
- Grammar's ceiling is set by licensing, not method — every competitive GEC model
  on the Hub is non-commercial.
- Fluency is unvalidated against proficiency labels (needs graded *audio*, not text).
