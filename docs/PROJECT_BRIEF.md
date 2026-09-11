# Project Brief — Spoken Open-Response Scoring

**Status:** active · **Owner:** Pranshul (Mercer Mettl) · **Started:** 2026-09-11
**Goal:** Replace the incumbent third-party scoring vendor ("Carnegie") for spoken
open-ended assessment responses with a self-hosted pipeline.

---

## 1. Problem

Candidates answer an open-ended question by speaking. We know the question text.
We receive the audio. We must return:

**Four scores** (continuous 0–100, correlated against human 0–5 integer labels):
| Category  | Meaning (per client) | Carnegie baseline (Pearson) |
|-----------|----------------------|------------------------------|
| Grammar   | grammatical accuracy of what was said | ~0.60 |
| Lexical   | vocabulary quality / sophistication    | ~0.60 |
| Fluency   | delivery: rate, pausing, continuity     | ~0.60 |
| Relevance | does the answer address the question    | ~0.40 |

**Four flags** (continuous score + tunable threshold, NOT hard binary):
| Flag | Definition (per client) |
|------|--------------------------|
| `prompt_read`       | Candidate repeats/paraphrases the prompt to fill time |
| `repetition`        | Padding-level content repetition. **Not** disfluency. |
| `off_topic`         | Answer does not address the question |
| `foreign_language`  | Any non-English speech |

Flags exist to detect **gaming/cheating**, not to measure disfluency.

## 2. Success criteria

1. Beat Carnegie's per-category correlation on the client's held-out set of ~350 items.
   Primary target: **relevance > 0.40**. Secondary: the other three > 0.60.
2. Flags validated on client-built edge-case sets; continuous scores so thresholds
   are tunable post-hoc without retraining.
3. Every score explainable — decomposable into named, individually defensible features.

## 3. Hard constraints

| # | Constraint | Source |
|---|------------|--------|
| C1 | **No generative LLM anywhere in the pipeline.** Encoders, embedding models, NLI cross-encoders, ASR, and small task-specific seq2seq only. | Client, stated twice |
| C2 | **No external API calls.** Fully self-hosted. | Client |
| C3 | Production target: on-prem/cloud **NVIDIA GPU, ≤16GB VRAM** | Client |
| C4 | **Batch processing.** No latency requirement — scores are not shown live. | Client |
| C5 | Commercial use. Research-licensed corpora permitted for now, **legal review pending** — every asset must be license-tagged. | Client |
| C6 | Dev machine is **AMD RX 9070 XT / ROCm**, production is CUDA. Code must be device-agnostic. | Observed |
| C7 | Client's labelled data is **test-only**. Zero training data from the client. | Client |
| C8 | Client's office laptop is off-limits to this project. Deliverable must be runnable by the client unaided. | Client |

## 4. Effort allocation (client-directed)

- **70%** → flags + relevance
- **30%** → grammar, lexical, fluency

## 5. What we have and don't have

**Have:**
- Question text at inference time
- **3 ideal answers (text) per question** — originally authored for Carnegie
- ~350 human-labelled client items (held by client, test-only)
- Carnegie's own per-item outputs (for baseline comparison, held by client)

**Do not have:**
- Any client audio, transcripts, or training data
- Any flag labels (client is constructing edge-case sets independently)

## 6. Data contract (proposed by us, client agreed format is open)

Client cannot handle `.wav` on their machine, so audio arrives as `.npy`:

```
item_id.npy       float32, mono, 16 kHz, range [-1, 1]
item_id.json      {item_id, question_id, duration_s, orig_sr}
questions.json    {question_id: {text, ideal_answers: [str, str, str]}}
```

Audio: max 60s, real-world noisy, already quality-filtered by client.

## 7. Population

L2 English speakers — predominantly **Indian, Filipino, African**. Explicitly
**no native US/UK speakers**. Accent must not be penalized; only clarity and
correctness of what is said.

> This is the dominant technical risk. Most public speech-scoring work is built
> and validated on Mandarin/Japanese/European L1 speakers. Our accent distribution
> is badly under-represented in every available corpus.
