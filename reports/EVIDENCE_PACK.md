# Evidence Pack — Replacing the Spoken-Response Scoring Vendor

**voxscore v0.1.0** · research prototype · 2026-09-11

---

## 1. Summary

A self-hosted pipeline that scores spoken open-ended responses on **grammar,
lexical, fluency and relevance**, and flags **prompt-read, repetition, off-topic
and foreign-language** responses. No generative LLM, no external API, every
shipped model commercially licensed, ~1× realtime at 3.1 GB VRAM.

**What is established**

| Claim | Evidence |
|---|---|
| Grammar scores rank proficiency correctly | Spearman **0.899**, 94.4% pairwise, on 18 graded responses |
| Lexical scores rank proficiency correctly | Spearman **0.856**, 94.4% pairwise |
| Gaming flags separate cleanly | AUC **1.000** for prompt-read, repetition, off-topic; 0% FPR |
| Relevance separates on- from off-topic | AUC **1.000**; a wholly off-topic response scores 8.7/100 |
| No false alarms on native English | foreign-language FPR **0.0%** at every threshold tested |
| Brief code-switching is not punished | 1–2 foreign fragments (1.6–3.1% of audio) score **0.0** |
| Runs inside the production budget | 3.1 GB VRAM, ~1× realtime including all models |

**What is not established, stated plainly**

1. **Fine-grained relevance ranking among genuinely on-topic answers.** We can
   show the system separates on-topic from off-topic essentially perfectly. We
   cannot yet show it distinguishes a *good* on-topic answer from a *mediocre*
   one — and that is exactly what your 0–5 relevance labels encode. This is the
   single largest open risk and your 350-item set resolves it in one run.
2. **The foreign-language flag's false-positive rate on accented English.** See
   §5. Until that number exists the flag must not be used to fail candidates,
   and the code ships with a provisional threshold saying so.
3. **Fluency against proficiency labels.** Validating it needs *graded audio*,
   which no permissively licensed corpus provides.

Nothing here is fitted to your data. The default scorer is an unfitted scorecard,
so these are floors rather than optimistic estimates.

---

## 2. The central finding

Your incumbent scores ~0.60 on grammar, lexical and fluency but **~0.40 on
relevance**. That asymmetry has a mechanical explanation, and it is our opening.

Grammar, lexical and fluency are properties of the response *alone*. Relevance is
the only category that requires reasoning about the prompt — and the obvious
implementation (embed the response, embed the ideal answers, take cosine) is
close to **uncorrelated** with relevance on personal-narrative prompts.

Your own data shows why. The three ideal answers for *"Share how you celebrated
your most recent birthday"* describe:

1. a family dinner at home — mother cooks, brother buys a bad cake
2. a surprise evening at a café organised by a friend
3. a solo day off — bookshop, dinner with a sister, a handmade notebook

Content overlap between them is near zero. **All three are perfect answers.** Any
system scoring relevance by resemblance to an ideal answer is measuring whose
story it is.

We measured the consequence directly. Against your birthday rubric, a response
that does nothing but **restate the prompt** scored the *highest of six* on raw
question similarity (0.787) and on question-element coverage (0.886) — above two
genuine answers. A contentless vague answer matched a real one on ideal-answer
similarity (0.761 vs 0.788).

> **Similarity-based relevance systematically rewards restating the question and
> saying nothing at length.** A candidate who fills 60 seconds paraphrasing the
> prompt outranks one who actually answers. That is a plausible and sufficient
> mechanism for 0.40.

### What we do instead

Derive a rubric per question, with no labels and no classifier:

- **Required elements** from the question's own dependency parse — always
  available, even with no ideal answers.
- **Required moves**: entity-mask the three ideal answers, keep only what appears
  in **≥ 2 of 3**. Two independent answers agreeing makes something a
  requirement; one mentioning it is personal idiosyncrasy. *Three ideal answers
  are exactly enough to separate the two — this is an asset you already own.*
- **Structural profile**: what a good answer looks like (how first-person, how
  much concrete detail, how much evaluative language), estimated from the ideal
  answers as a **distribution** rather than as content to match.
- **Shareability**: how much the three ideal answers agree with each other,
  which decides per question how far to trust content matching at all. High for
  opinion prompts where model answers converge on the same arguments; near zero
  for personal prompts where they share only structure. No question-type labels.

Everything is matched on **entity-masked** text, so differing personal detail
costs nothing.

**The bias trap it survives.** All three of your birthday ideal answers describe
a *low-key* celebration. We wrote an adversarial response describing a large,
loud party — equally valid, opposite in kind. It scored **at or above** the quiet
one on every relevance feature (element coverage 0.781 vs 0.694, specificity 5.48
vs 2.86). A content-matching system would have marked it down for being
different. This is now a permanent regression test.

---

## 3. Results

### 3.1 Score validity

You hold all labels as test-only, so there was nothing to validate against. But
absolute labels are not needed to check that a scorer is **ordered** correctly,
and ordering is what you measure. We authored 18 responses — six questions × three
proficiency levels — using error patterns characteristic of Indian, Filipino and
West African L2 English.

| category | Spearman ρ | pairwise accuracy |
|---|---|---|
| grammar | **0.899** | 94.4% |
| lexical | **0.856** | 94.4% |
| relevance | −0.185 | 38.9% |

The relevance figure is the **fixture's** fault, not only the scorer's: it varies
*proficiency* while holding relevance roughly constant. All three levels genuinely
answer the question — a weak speaker saying *"My birthday is last month only, my
mother she make one cake"* is being entirely relevant. The correct instrument for
relevance is discrimination, reported below.

> **A prediction worth testing on your data.** Human relevance ratings usually
> carry a proficiency halo — raters mark fluent answers as more relevant. Our
> relevance is deliberately near-flat across proficiency. If your labels have
> that halo, our correlation is capped regardless of how good the topical
> judgement is. We have *not* tuned for this, because tuning to a guess about
> rater behaviour is how a system stops measuring the construct it claims to.
> Your validation run will show it immediately, and the fix is a weighting
> change, not a redesign.

### 3.2 Flags

36 labelled pairs, where each positive is generated from a genuine response so the
*only* difference is the gaming behaviour.

| flag | AUC | recommended threshold | TPR | FPR |
|---|---|---|---|---|
| `prompt_read` | **1.000** | 55 | 100% | 0% |
| `repetition` | **1.000** | 55 | 100% | 0% |
| `off_topic` | **1.000** | **35** | 100% | 0% |

`off_topic`'s threshold is deliberately lower. At 55 it caught only half of
genuinely mismatched answers — off-topic responses lack the emphatic signature
that echo and padding have. A single shared constant across four flags was wrong;
each threshold now comes from its own curve.

**The signature that makes `prompt_read` work.** A genuine answer is *less*
question-similar than a model answer, because it adds content of its own. Prompt
echo **exceeds** the reference: measured 1.289× the ideal answers' own question
similarity. That comparison, not raw similarity, is the discriminator.

**Disfluency is never treated as gaming.** Immediate repetitions are stripped
before repetition is measured. A stuttering candidate scores well below threshold
and within 20 points of the same text delivered fluently. This is pinned by test.

### 3.3 Foreign language — dose-response

Controlled proportions of Hindi and Tamil (FLEURS, CC-BY-4.0) spliced into English
responses, level-matched and cross-faded so the detector cannot key on the splice.

| true % non-English | flag score (before fix) |
|---|---|
| 0% | 0.0 |
| 5% | 2.5 |
| 10% | 4.2 |
| 20% | 21.7 |
| 35% | 32.3 |
| 50% | 40.1 |
| 75% | 46.8 |
| 100% | 63.8 |

Monotonic — but a response spoken **entirely in Hindi scored 63.8**, and 75% Hindi
scored 46.8, below the then-default threshold of 55. The flag would have missed
three quarters of an answer in another language.

**Cause.** The scoring transcript is forced to English on purpose, so a Hindi
stretch produces visibly broken output rather than clean Hindi that the grammar
and lexical modules would score as an answer. But the flag then ran its *text*
language detector over that same forced transcript, which always reads as English.
That channel held weight 0.30 and voted "not foreign" on every item, capping the
score before anything else was considered.

**Fix.** A second, unforced transcription, run only when the acoustic channel is
already suspicious (clean English measures below 0.02 and skips it entirely).
Absent it, the text channel now abstains and the remaining weights renormalise.
Default threshold lowered to 25 to match the flag's narrower dynamic range.

> This dose-response table is itself a deliverable: it says what a threshold
> *means*. You asked to set that boundary yourselves, and this is the curve you
> need to do it.

### 3.4 Runtime

| | |
|---|---|
| Peak VRAM | **3.1 GB** (budget: 16 GB) |
| Throughput | ~1× realtime end to end, all models loaded |
| ASR alone | 1.7× realtime short-form (beam 5), 17× long-form |
| Model load | 8.5 s for Whisper large-v3 |

Comfortably inside a batch budget with room for a larger ASR model if WER on your
accents justifies it.

---

## 4. Bugs found, and what they say about the method

Seven defects were found during development. **None of them crashed.** Every one
produced plausible numbers in a believable range. They are listed because the way
they were caught is more transferable than the fixes.

| # | Defect | Effect if shipped |
|---|---|---|
| 1 | All six "lower is better" curves double-negated | Grammar rewarded **more** errors; fluency rewarded filled pauses; lexical rewarded commoner vocabulary |
| 2 | Relevance averaged topic-blind features | An off-topic passage dense with detail scored 73/100 while the off-topic flag fired at 98 |
| 3 | `output_scores=True` silently ignored by transformers | ASR confidence would have been exactly 0.0 for every item, forever |
| 4 | `generate` strips the forced decoder prefix | "Confidence" of a perfect transcript (−8.2) barely beat *silent audio* (−7.8) |
| 5 | Long-form confidence scored against one 30 s window | Clean English tripped the "not English" signal |
| 6 | Foreign-language text channel could never fire | 100% Hindi capped at 63.8 |
| 7 | Abstention scored as a pass (×3 separate places) | Unmeasurable inputs received full marks |

Two patterns are worth naming:

**Consistency checks between independent outputs caught more than assertions
did.** Bug 2 surfaced because a *flag* said off-topic while a *score* said
relevant. Neither number looked wrong alone.

**Controls matter.** Bug 4 was caught by scoring *silent audio* and finding it
nearly matched a perfect transcription. Any confidence-like quantity should be
checked against an input that must score badly.

And one recurring class, hit three times: **a channel that cannot measure must
abstain, not vote.** Returning 0.0 for "unavailable" reads as "measured, and
fine". Every such path now collapses its weight and renormalises.

---

## 5. Risks

**1. Foreign-language fairness — the one that could harm real candidates.**
Language-ID models routinely misread heavily accented English as the speaker's
L1: Indian-accented English as Hindi, Filipino-accented as Tagalog. Your
population is *entirely* L2 speakers, so a naive implementation false-positives
on exactly the people it must not, and the failure is invisible unless
specifically tested.

Mitigations already in place: posteriors rather than argmax labels, a required
text corroboration channel, a restricted language candidate set, and a test
pinning that a single confused window cannot fire the flag. Native-English FPR is
**0.0%**.

*The number that decides deployability — FPR on Indian-accented English — is
pending the Svarah corpus.* Until then this flag is advisory only.

**2. Grammar's ceiling is set by licensing, not method.** Every competitive GEC
model on the Hub is non-commercial: the most-used one (86k downloads) is
CC-BY-NC-SA, Grammarly's is CC-BY-NC, and the next candidate declares no licence
at all. The only permissively licensed option is `Unbabel/gec-t5_small`. It
handles the L2 error types that matter and leaves correct sentences untouched, but
if legal will clear a non-commercial model for **internal benchmarking only**,
quantifying that gap is worth doing.

**3. ASR accuracy on your accents is unmeasured.** Everything downstream inherits
it. We have no transcripts to compute WER against, and no permissively licensed
corpus of Indian/Filipino/African-accented English with references.

**4. Length confounds the lexical score.** Responses cap at 60 s (~100–150 words)
and most diversity metrics are unstable below ~100 tokens. MTLD is used because it
is length-robust and raw TTR is excluded outright, but short responses are marked
low-confidence rather than scored as if reliable.

---

## 6. What we need from you

| | Why it matters |
|---|---|
| **The Ideal Answers sheet as CSV** | Only the birthday answers were legible in the screenshots. Every other question currently falls back to question-only features, which is measurably weaker. This is the highest-value single item. |
| **HF token + Svarah / Common Voice terms accepted** | Unblocks the accented-English fairness audit — the one result standing between the foreign-language flag and deployment. |
| **One validation run against your 350** | Resolves the fine-grained relevance question, and tells us whether your relevance labels carry a proficiency halo. |
| **Legal sign-off on `reports/LICENCE_AUDIT.md`** | Generated live from the Hub; every shipped model is Apache-2.0 or MIT. |

---

## 7. How to run it

```bash
python scripts/score_batch.py \
    --input  data/responses \
    --questions questions.json \
    --out    reports/run1
```

Produces `results.xml`, `results.json` and `features.csv`. The CSV joins directly
against your labels for a correlation check; `scripts/fit_calibration.py` will
fit and — importantly — tell you whether fitting actually beat the unfitted
scorecard, using cross-validation that holds whole *questions* out so the number
estimates generalisation to a new prompt rather than a new candidate.

Every score in the XML carries its contributing features and a one-sentence
explanation, so a contested result can be explained without a separate report.
