# Engineering Journal

Running log of the build: what was tried, what broke, what it means.
Append-only. Newest at the bottom.

---

## 2026-09-11 — Day 0: requirements, environment, scaffolding

### Requirements gathering
Two rounds of structured questions with the client before writing any code. Full
outcome captured in `PROJECT_BRIEF.md`. The answers that most changed the design:

- **No client training data at all.** ~350 labelled items exist but are test-only.
  This kills any plan involving a large supervised head. The system has to be
  *feature-first with a thin fitted layer*, where every feature is independently
  defensible. Fortunately this is also exactly what the explainability ask requires.
- **3 ideal answers per question exist.** Originally authored to satisfy Carnegie.
  This is the single most valuable asset we have for relevance, and Carnegie
  apparently squandered it (0.40 correlation).
- **Carnegie baselines: 0.6 / 0.6 / 0.6 / 0.4.** A 0.60 fluency correlation is roughly
  what speech-rate-and-pause features alone achieve in the literature, which suggests
  the incumbent is not doing anything sophisticated. The bar is soft.
- **Accent distribution: Indian, Filipino, African. No native speakers.** This is the
  dominant risk and is under-represented in every public corpus.
- **Flags mean cheating detection**, not disfluency. `repetition` is padding-level;
  `prompt_read` is prompt-echo-to-fill-time. This reframes both as *gaming* detectors
  and moves them close to the relevance module rather than the fluency module.

### Environment findings
| Check | Result |
|---|---|
| GPU | AMD RX 9070 XT (RDNA4), **17.1 GB** VRAM |
| torch | `2.9.1+rocm7.2.1`, HIP 7.2, `cuda.is_available() == True` |
| bf16 GEMM | **138 TFLOPS** (fp32: 15.8) — bf16 is ~8.7× fp32, use it everywhere |
| `torch.compile` | **broken** — no Triton on ROCm/Windows. Irrelevant for inference. |
| CPU / RAM | Ryzen 7 9800X3D (8c/16t) · 30.9 GB |
| Disk | 1.52 TB free |
| Network | HF, PyPI, OpenSLR all reachable |
| Java | **absent** → forced ADR-002 |

Production is NVIDIA/CUDA while dev is AMD/ROCm. Everything must go through
`torch.device` abstraction with no CUDA-specific calls. Noted as constraint C6.

### Incident: stray `.pth` written to the base Python install
While wiring the venv to inherit mlenv's torch, I used `site.getsitepackages()[-1]`
to locate the venv's site-packages. On a `--system-site-packages` venv that call
returns **multiple** entries and `[-1]` is the *base interpreter's*, not the venv's.
The result: `_mlenv_inherit.pth` landed in
`AppData\Local\Programs\Python\Python312\Lib\site-packages`, machine-wide.

Caught immediately, removed, and verified the base interpreter no longer resolves
`torch`. Re-done against the literal `.venv/Lib/site-packages` path. See ADR-001.

Recording this because it is exactly the class of error that silently pollutes a
machine and surfaces weeks later as an unexplainable import.

### Scaffolding
Project laid out as `voxscore/` (asr, features, flags, scoring, data, eval, utils)
with `configs/`, `data/{raw,interim,processed,synthetic}`, `reports/`, `scripts/`,
`tests/`, `docs/`. Old GPU benchmark files moved to `_bench/`.

### The torchcodec wall (twice)
Both `torchaudio.load` (2.9) and `datasets` (5.x) audio decoding now delegate to
**torchcodec**, which wants system FFmpeg and has no usable ROCm/Windows path. There is
no ffmpeg on this machine and none on the client's locked-down laptop either.

Taking the dependency would have made the shipped bundle much harder to install on the
one box that matters. Instead:
- wav reading in `utils/audio_io.py` -> `soundfile`
- HF dataset audio in `data/hf_audio.py` -> `Audio(decode=False)` + `soundfile` on raw bytes

Covers wav/flac/ogg, which is every corpus in the eval plan. Verified on LibriSpeech.

### Client screenshots: the question set (major)
The client sent photographs of `MainSheet.xlsx`. This changed the design more than
anything since the requirements call. Captured to `data/raw/questions_sample.json`.

**24 questions, and they split cleanly into the two families ADR-007 predicted** —
16 personal/experiential (*"Share how you celebrated your most recent birthday"*,
*"Talk about a friend you used to be close with but later lost touch"*) and 8
opinion/argumentative (*"In your opinion, does technology improve human thinking or make
people more dependent?"*, *"Does teamwork always produce better results than working
alone?"*). The continuous-shareability router is not hypothetical; it is needed.

**ADR-006 confirmed on real data.** Three ideal answers for the birthday question were
fully legible. Their content is essentially disjoint:
1. family dinner at home, mother cooks, brother buys a bad cake
2. surprise evening at a café organised by a friend
3. solo day off, bookshop, dinner with sister, handmade notebook gift

Cosine similarity between these would be low, yet all three are *perfect* answers. Any
system scoring relevance by resemblance to an ideal answer is measuring the wrong
quantity. What they actually share is the arc: **low-key framing -> concrete specific
detail -> emotional resolution.**

Two further observations from the same screenshots:

- **The ideal answers are scripted to imitate speech** — "Okay uh", "Um", "Er", "Right",
  "So yeah", "wanna", "I guess". They are not written prose. This settles the grammar
  question: *spoken-adjusted* is the default variant, and strict written-norm is the
  alternate. Scoring these ideal answers with a written-English GEC model would mark
  down the client's own gold standard.
- **All three birthday answers describe a low-key celebration.** That is a latent bias
  trap: content matching would reward "quiet dinner" and penalise a candidate who
  truthfully describes a large party, which is an equally valid answer. Added as an
  explicit adversarial test case rather than left to chance.

### Outstanding asks
1. **HF token + gated dataset access** (`ai4bharat/Svarah`, `mozilla-foundation/common_voice_17_0`).
   These are the only source of *accented-English negatives*. Without them the
   foreign-language flag cannot be calibrated against the population it will actually
   run on, which is the one failure mode most likely to harm real candidates.
2. The **Ideal Answers sheet as CSV** rather than photographs — only the birthday
   answers were legible.

---

## 2026-09-11 — Day 0 (cont.): ASR working end to end

Four non-obvious failures, all found by measurement rather than by reading docs.
Recording them because three of the four would have produced *plausible numbers*
rather than errors, which is the dangerous kind.

### 1. Whisper hung for >10 minutes on 6 seconds of audio
Calling the processor with `padding="longest", truncation=False` produced **585
frames instead of the 3000** Whisper's encoder expects. That is the *long-form*
calling convention; on short input, combined with `return_timestamps` and beam
search, generation degenerated and never terminated.

Fixed by branching on duration: <=30 s uses the standard padded path (beams
allowed), >30 s uses long-form (attention mask + timestamps, greedy only). Client
responses cap at 60 s, so **both paths are live** and both are now exercised.

Measured on large-v3: short path 1.7x realtime at beam 5, long-form 17x realtime
greedy, 3.09 GB VRAM, 8.5 s to load. Comfortable inside the 16 GB production budget.

### 2. `output_scores=True` is silently ignored
transformers 5.x warns that the flag "is not valid and may be ignored" on the
Whisper path and returns no scores. Reading `out.scores` would have yielded
**exactly 0.0 confidence for every item, forever, without raising.**

### 3. `generate` strips the forced decoder prefix
The returned sequence starts at the first *text* token -- no
`<|startoftranscript|><|en|><|transcribe|>`. So teacher-forcing on it scores a
decoder that has been told neither its task nor its language.

Caught by running a **silent-audio control**: the "confidence" of a perfect
transcription was -8.2 while silence scored -7.8. A metric that cannot separate a
correct transcript from no audio at all is measuring nothing. After restoring the
prefix:

| input | avg_logprob |
|---|---|
| clean speech | **-0.112** |
| speech + noise | -0.684 |
| white noise | -0.869 |
| silence | -0.937 |

Monotonic and well separated. The control is the reason this was caught at all --
worth repeating for every confidence-like quantity we add.

### 4. Whisper hallucinates on non-speech
White noise transcribes as **"Thank you."** and pure silence as **"you"**. This is
a well-known Whisper artefact, and it matters commercially: a candidate who says
nothing still yields a scorable transcript. Without a quality gate the system
would award grammar and lexical scores to silence. The `quality` feature block is
therefore a correctness requirement, not a diagnostic nicety.

### 5. torchaudio's forced aligner is unusable for us
`torchaudio.functional.forced_align` has **no CUDA/HIP kernel** (CPU only) *and*
emits a deprecation warning saying it will be removed as torchaudio enters
maintenance. Every fluency feature depends on these timings.

Wrote our own CTC Viterbi aligner (`asr/align.py`, ~70 lines): device-agnostic, no
deprecated dependency, and exact token-boundary recovery via blank-extended state
indices rather than inferring boundaries from runs of repeated ids. Pinned against
the torchaudio reference at >95% frame agreement in `tests/test_align.py` while
that reference still exists.

### Note for the fluency module
Smoke run showed phonation ratio 0.60 on clean read speech, depressed by ~0.5 s of
leading silence. `phonation_time_ratio` must be computed over the **speech span**,
not total duration, for the same reason edge pauses are trimmed: recorder start/stop
latency is not a property of the candidate.

**Status:** 27 tests passing. ASR -> alignment -> pauses -> windowed LID verified
end to end on ROCm. Committed as `9bd2503`.

---

## 2026-09-11 — Day 0 (cont.): relevance, probed against the real ideal answers

Built `features/relevance.py` and probed it on the client's actual birthday ideal
answers against six hand-written responses covering the failure modes we care about.
Three findings, two of them corrections to my own design.

### shareability took three attempts and both failures were the same mistake
The idea (ADR-007) is that agreement *among the three ideal answers* measures whether
their content is reusable across candidates, routing between the narrative and
argumentative strategies with no labels. The idea holds. Measuring it was harder:

| attempt | personal | opinion | verdict |
|---|---|---|---|
| sentence-embedding agreement | 0.711 | 0.676 | **backwards** |
| lemma Jaccard, question terms removed | 0.085 | 0.047 | **backwards** |
| informativeness-weighted overlap | 0.066 | **0.078** | correct |

Attempt 1 failed because bi-encoder similarity conflates form with content: *"It felt
warm, easy, and very real"* and *"It felt personal, unforced, and easy to enjoy"* are
near-identical in form and share no content whatever. Attempt 2 failed for two
compounding reasons -- excluding question terms deleted exactly the shared substance of
an opinion prompt (*technology*, *dependent*, *thinking*), and the residue on both
families was dominated by common verbs any two English texts share.

Weighting by word rarity fixes both. The underlying error was the same twice: **I kept
measuring similarity of form when the quantity I wanted was shared substance.**

The correct margin is thin (0.078 vs 0.066) and rests on opinion ideal answers I
authored as a fixture, since the client's real ones were not legible. Confirming this
against their actual opinion ideal answers is now the main reason to want that CSV.

### Similarity-based relevance actively rewards the two things we must catch
The most useful result of the probe. On the birthday rubric:

- **Prompt echo scored highest of all six responses** on `sim_q` (0.787) and
  `element_coverage` (0.886).
- **A contentless vague answer matched a genuine one** on `move_coverage`
  (0.761 vs 0.788).

Cosine similarity cannot distinguish *answering* the question from *restating* it, nor
from *saying nothing at length*. Any relevance score built primarily on embedding
similarity will rank a candidate who fills 60 seconds paraphrasing the prompt above one
who actually answers. **This is a plausible mechanism for the incumbent's 0.40.**

Added three features that do separate them, all measured on content words rather than
embeddings: `content_novelty_vs_q`, `distinct_content_rate`, `content_word_rate`. With
`specificity` these carry the discrimination that cosine cannot.

### Per-question calibration beats both absolute and self-relative thresholds
`pct_windows_offtopic` fired on nothing (0.000 everywhere, off-topic included) under a
threshold relative to the response's own mean -- a uniformly off-topic answer has a low
mean and so sets itself a low bar. A global constant fails too, since baseline question
similarity varies per prompt (0.45-0.82 across probe cases on one question). Fixed by
referencing the **ideal answers' own** window similarity, which gives the right scale
per question for free.

### Final discrimination, all six cases separated
| response | catching signature |
|---|---|
| off-topic | `sim_q` 0.453, `pct_windows_offtopic` 0.75 |
| drifts off mid-answer | `min_window_sim` 0.376 (lowest) |
| prompt echo | `window_sim_vs_ref` **1.289** |
| vague | `specificity` 0.0 |
| repetitive padding | `distinct_content_rate` 0.474 (lowest) |
| good (quiet) / good (big party) | clean on all |

`window_sim_vs_ref > 1` is close to a definition of prompt echo: a genuine answer is
*less* question-similar than a model answer, because it adds content of its own. Echo
exceeds the reference. This goes straight into the `prompt_read` flag.

---

## 2026-09-11 — Day 0 (cont.): full pipeline, and three bugs the smoke test caught

Wired everything together: `pipeline.py`, the scorecard in `scoring/aggregate.py`, the
remaining feature blocks (fluency, lexical, grammar, quality), and XML/JSON output.
84 features per item. First end-to-end run used ~40 s of concatenated LibriSpeech
paired with the birthday question -- deliberately nonsense as an *answer*, so relevance
should collapse while every other path still executes.

It found three defects. None of them crashed; all three produced plausible numbers.

### 1. Relevance must multiply, not average
The off-topic passage scored `off_topic` **98.2** while relevance came out at **73.3**.
Flatly contradictory.

Cause: `specificity` and `content_novelty_vs_q` are **topic-blind**. They were added to
catch vague and echoed answers and they do that well -- but a wholly off-topic passage
is *dense* with specific, novel content. It is simply about the wrong thing. Averaging
them alongside the topic features let off-topic content buy relevance points back.

Restructured as **anchor x engagement**: is it about the question, times is it a
substantive answer, with a floor on the multiplier. An additive model cannot express
"this is disqualifying", and relevance has exactly one disqualifying condition. Result:
73.3 -> **8.7**, now consistent with the flag.

### 2. Every "lower is better" curve was double-negated
`_curve(x, lo, hi)` already encodes direction: `lo > hi` means lower is better. It also
had an `invert=True` flag, and **all six** such features passed the flag *and* ordered
`lo > hi`, cancelling out.

So the shipped scorecard rewarded:
- **more** grammatical errors
- **more** long pauses and **more** filled pauses
- **commoner** vocabulary
- **more** topic drift

Every score stayed in a believable 0-100 range and nothing raised. Grammar on clean
LibriSpeech prose scored 40.4; after the fix, 72.8.

The flag is now **deleted** rather than fixed, so the mistake is unexpressible.
`tests/test_scoring.py` asserts the direction of all 13 scorecard features, because
this is precisely the bug class that unit tests exist for and smoke tests miss.

### 3. ASR confidence was wrong on every response over 30 s
The teacher-forced confidence pass truncates features to one 30 s encoder window while
scoring the *whole* transcript, so on long-form items most tokens were scored against
audio the decoder never saw. A clean 40 s item reported confidence low enough to trip
the foreign-language flag's "ASR struggled to read this as English" signal (10.0) and
to emit a spurious quality warning.

Fixed by estimating confidence from a separate short-form pass over the first 30 s,
documented as a sample. Costs one extra generate call, which is free in a batch job.
Foreign-language went 10.0 -> **0.0** and quality confidence 0.80 -> **1.00**.

### Final state of the smoke item
| | score | correct? |
|---|---|---|
| grammar | 72.8 | yes -- clean published prose |
| lexical | 73.8 | yes -- rich Victorian vocabulary |
| fluency | 68.7 | yes -- professional read speech |
| relevance | **8.7** | yes -- nothing to do with the question |
| `off_topic` | 98.2 fired | yes |
| `repetition` | 55.7 fired | yes -- the audio was genuinely doubled |
| `prompt_read` / `foreign_language` | 0.0 | yes |

Runtime 38.9 s for 40 s of audio (1.03x realtime) including all models. Well inside a
batch budget.

**Reflection worth keeping:** all three bugs produced *plausible* output. The smoke test
caught them only because I checked whether the numbers were **mutually consistent**
(a flag saying off-topic while the score said relevant) rather than whether the run
succeeded. Consistency checks between independent outputs are worth more here than
any single assertion.

**Status:** 52 tests passing. Full pipeline verified end to end.

---

## 2026-09-11 — Day 0 (cont.): first real measurements

Built the evaluation harness and got numbers instead of opinions.

### Ordinal validation without labels
The client holds every label as test-only, so there is nothing to validate against.
But absolute labels are not required to check that a scorer is **ordered** correctly,
and ordering is what the client measures anyway (they compute correlation).

Authored 18 responses -- six questions x three proficiency levels -- using error
patterns characteristic of Indian, Filipino and West African L2 English. A valid
scorer must rank weak < mid < strong.

| category | Spearman rho | pairwise accuracy |
|---|---|---|
| grammar | **0.899** | 94.4% |
| lexical | **0.856** | 94.4% |
| relevance | -0.185 | 38.9% |

Grammar and lexical are strong. For context, the incumbent manages ~0.60 on these
against human labels -- not the same measurement, and this fixture is ours, so it is
evidence of validity rather than a competitive claim. But a scorer that could not order
its own graded fixture would be disqualified, and these are not.

### The relevance number is the fixture's fault, not (only) the scorer's
Relevance came out uncorrelated. The honest reading is that **the fixture is the wrong
instrument**: it varies *proficiency* while holding relevance roughly constant. All
three levels genuinely answer the question -- a weak speaker saying *"My birthday is
last month only, my mother she make one cake"* is being entirely relevant.

The right test for relevance is **discrimination**, and that is measured separately
below at AUC 1.000.

Worth stating plainly rather than burying: **fine-grained relevance ranking among
genuinely on-topic answers remains unvalidated.** We can show the system separates
on-topic from off-topic essentially perfectly. We cannot yet show it distinguishes a
good on-topic answer from a mediocre one, and that is precisely the discrimination the
client's 0-5 relevance labels encode. This is the main open risk, and the client can
resolve it in one run against their 350.

**A related prediction to test:** human relevance ratings usually carry a proficiency
halo -- raters mark fluent answers as more relevant. Our relevance is deliberately
near-flat across proficiency. If their labels have that halo, our correlation will be
capped no matter how good the topical judgement is. Deliberately *not* tuned for, since
tuning to a guess about rater behaviour is how a system stops measuring the construct
it claims to. Flagged instead as something their validation run will reveal immediately.

### Three defects found by diagnosing the relevance features directly
1. **Topic drift was noise on short responses.** A fixed 4-way split produced ~15-word
   windows; `pct_windows_offtopic` read 0.50 / 0.00 / 0.50 across three responses that
   were all squarely on topic. Replaced with a minimum *window size* (25 words) and a
   minimum response length (75 words), abstaining below that.
2. **`specificity` scored 0.000 on a strong answer dense with detail.** It counted only
   named entities and numerals, and that answer deliberately says *"someone"* and
   *"she"* rather than naming the friend. Declining to name a person is a stylistic
   choice, not an absence of detail -- and penalising it would have systematically
   marked down exactly the discreet, fluent answers we want to reward. Now counts
   concrete low-frequency vocabulary too.
3. **Abstention was being scored as a pass.** Both `profile_match` (unavailable without
   ideal answers) and the drift block (unavailable on short responses) returned 0.0,
   which the scorer read as a measurement rather than a missing value. A 52-word answer
   got a free "0% off-topic" while an 80-word answer was actually assessed. Both now
   collapse their weight and the remainder renormalises, so *unmeasured* and *measured
   as fine* are no longer the same thing.

### Flags: clean separation
36 labelled text pairs, where each positive is generated from a genuine response so the
*only* difference is the gaming behaviour.

| flag | AUC | best threshold | TPR | FPR |
|---|---|---|---|---|
| `prompt_read` | **1.000** | 21 | 100% | 0% |
| `repetition` | **1.000** | 5 | 100% | 0% |
| `off_topic` | **1.000** | 28 | 100% | 0% |

Acted on one result: **`off_topic`'s default threshold moved from 55 to 35.** At 55 it
was catching only half of genuinely mismatched answers. Off-topic responses score lower
than the other gaming behaviours because they lack the emphatic signature that echo and
padding have, so a single shared constant across all four flags was wrong. Per-flag
thresholds now come from curves.

These AUCs are on a small, self-generated set and will fall on real data. The value is
directional: the flags key on the intended signals and do not fire on genuine answers.

---

## 2026-09-11 — Day 0 (cont.): the foreign-language flag was structurally broken

Ran the dose-response experiment: splice controlled proportions of Hindi and Tamil
(FLEURS, CC-BY-4.0) into English responses and plot flag score against true proportion.
This is worth more than a positive/negative set because it says what a threshold
*means*, which is the decision the client kept for themselves.

| true % non-English | flag score |
|---|---|
| 0% | 0.0 |
| 5% | 2.5 |
| 10% | 4.2 |
| 20% | 21.7 |
| 35% | 32.3 |
| 50% | 40.1 |
| 75% | 46.8 |
| **100%** | **63.8** |

Monotonic, which is the good news. The bad news is the ceiling: **a response spoken
entirely in Hindi scored 63.8**, and one that was 75% Hindi scored 46.8 -- below the
default threshold of 55. The flag would have missed three quarters of an answer in
another language.

### Why: the corroborating channel could never fire
The scoring transcript is forced to English on purpose (ADR-004) so that a Hindi stretch
produces visibly broken output rather than clean Hindi that the grammar and lexical
modules would score as though it were an answer. That decision is right.

But the foreign-language flag then ran its *text* language detector over that same
forced-English transcript. Forced English decoding of Hindi audio yields
English-looking tokens, so the text detector reported "English" no matter what. That
channel carried weight 0.30 and voted "not foreign" on every item, capping the
achievable score at 0.70 before any other consideration.

This is the same bug class as the abstention problem found earlier: **a channel that
cannot measure was voting instead of abstaining.** Third instance in this project.
Worth naming as a pattern rather than fixing case by case.

### Fix
A second, unforced transcription, run **only when the acoustic channel is already
suspicious** (mean p(non-English) > 0.12). Clean English responses measure below 0.02
and skip it entirely, so the average cost is small while the items that need an
independent text signal get one. When it is absent the text channel abstains and the
remaining weights renormalise, so it can no longer cap the score.

Also rebalanced toward the acoustic channels and dropped the default threshold from 55
to 25, since the flag's dynamic range is far narrower than the other three.

### The good news, unchanged by any of this
- **US English false-positive rate: 0.0%** at thresholds 40, 55 and 70. Mean score 0.0,
  max 0.0 across 12 native-English responses.
- **Brief code-switching correctly ignored.** One or two foreign fragments (1.6-3.1% of
  audio) scored **0.0**. A candidate dropping a couple of borrowed words is not
  answering in another language, and the flag agrees.

The experiment that actually decides deployability -- false-positive rate on
**Indian-accented English** -- is still blocked on the Svarah download. Until that
number exists, this flag must not be used to fail anyone, and the code says so.

---

## 2026-09-11 — Day 0 (cont.): the fairness risk was real, and it is measured

Svarah finished downloading. Ran the full experiment: five foreign languages at eight
proportions, plus both English groups.

### Dose-response, after the span-targeted fix
| true % non-English | hi | ta | tl | sw | yo | mean |
|---|---|---|---|---|---|---|
| 0% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| 10% | 25.7 | 11.1 | 30.3 | 52.8 | 0.0 | 24.0 |
| 20% | 56.7 | 43.0 | 48.3 | 62.7 | 44.3 | 51.0 |
| 35% | 73.1 | 59.2 | 62.2 | 58.9 | 60.2 | 62.7 |
| 50% | 75.3 | 56.6 | 76.2 | 72.3 | 65.9 | 69.3 |
| 75% | 81.7 | 77.9 | 81.2 | 78.7 | 76.9 | 79.3 |
| **100%** | 92.1 | 96.2 | 90.9 | 99.2 | 97.1 | **95.1** |

Full dynamic range recovered (0 to 95, against 0 to 63.8 before). Detection is weakest
on **Yoruba** -- 0.0 at 10%, where Swahili reaches 52.8 -- so sensitivity is materially
language-dependent and a single global threshold treats languages unequally.

### The result that decides everything
Both groups below are speaking **English**.

| group | mean | p90 | max | FPR@25 | FPR@40 | FPR@55 |
|---|---|---|---|---|---|---|
| US English (FLEURS) | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% | 0.0% |
| **Indian English (Svarah)** | 8.5 | 26.8 | 43.1 | **16.7%** | 8.3% | 0.0% |

**At the default threshold of 25, one in six Indian-accented candidates would be
flagged for speaking a foreign language while speaking English. Native English scored
zero on every single item.**

This is exactly the failure predicted in DESIGN.md before any code was written, and it
is worth being blunt about what it means: a threshold tuned on native English -- the
obvious thing to do, and what any evaluation using only en_us would have produced --
ships a system that penalises Indian, Filipino and African candidates for their accent
and nobody else. The population this product serves is *entirely* L2 speakers. The
error would have been invisible without deliberately seeking out accented negatives.

### Two responses

**1. Structural.** The text channel was an additive term, so the best it could do was
decline to add points. It is now a **multiplier**. If the spans that sounded foreign
are re-transcribed and come back as *English*, that is positive evidence the accent was
misread, and the score should fall rather than merely fail to rise. Acoustic evidence
alone is not safe to act on for this population, and the scoring now encodes that.

**2. Threshold.** Default raised from 25 to **45** — the lowest point at which accented
English is not penalised, while still catching a response roughly a fifth or more spoken
in another language.

### Confirmation run at n=20, after the multiplier
Both groups still speaking English:

| group | mean | p90 | max | FPR@15 | FPR@25 | FPR@30 |
|---|---|---|---|---|---|---|
| US English | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% | 0.0% |
| Indian English, acoustic only *(n=12)* | 8.5 | 26.8 | 43.1 | -- | **16.7%** | -- |
| **Indian English, after the fix** *(n=20)* | **3.4** | **15.7** | **21.9** | 15.0% | **0.0%** | **0.0%** |

The worst accented-English score fell from 43.1 to 21.9, and the false-positive rate at
threshold 25 fell from 16.7% to zero.

**Threshold set to 30**: the measured floor for a zero false-positive rate is 25, and 30
adds margin over the worst observed accented score without giving up much sensitivity.
(It had been raised to 45 on the n=12 data; the multiplier made that unnecessarily blunt.)

### The trade, stated rather than hidden
Suppressing accent false-positives costs real detection:

| true % non-English | before fixes | intermediate | **final** |
|---|---|---|---|
| 10% | 4.2 | 24.0 | 14.0 |
| 20% | 21.7 | 51.0 | **29.3** |
| 35% | 32.3 | 62.7 | 52.8 |
| 50% | 40.1 | 69.3 | 58.6 |
| 100% | 63.8 | 95.1 | **91.9** |

A 20%-foreign response scores 29.3 where the intermediate version gave 51.0. That is
deliberate. Flagging a real candidate for their accent is far worse than missing a
partial code-switch, and the two cannot both be optimised.

### Caveats recorded rather than buried
- **n = 20 per group.** A 0% false-positive rate on twenty items is not a guarantee.
- **Language-dependent sensitivity.** Yoruba detection is much weaker than Swahili; one
  global threshold is not equally fair across L1s.
- **This flag should route to human review, not to an automatic fail.** The code carries
  that statement in its docstring, not only here.

### What this episode is actually about
The prediction was written into `DESIGN.md` before any code existed, the experiment was
built to test it specifically, the prediction was confirmed at 16.7%, and the fix was
measured rather than assumed. None of that would have happened from an evaluation using
only `en_us` -- which is the default any English-language corpus hands you, and which
would have reported a flawless 0.0% false-positive rate while shipping a system that
penalised the entire candidate population for their accent.

### The bias trap held
All three real ideal answers describe a *low-key* birthday. The adversarial "BIG party"
response scored **at or above** the quiet one on every relevance feature
(`element_coverage` 0.781 vs 0.694, `specificity` 5.48 vs 2.86). Entity masking plus
coverage-based matching means a truthful answer that differs in kind from the model
answers is not penalised. Worth keeping as a permanent regression test.


---

## 2026-09-11 — Day 0 (cont.): two measurements that corrected me

The client asked whether the method requires them to author ideal answers. That
deserved a measurement rather than an opinion, and the measurement went against what I
had been telling them.

### Ablation: the ideal answers buy nothing on a personal-narrative question
Every probe response scored twice against the same question — once with the full rubric
built from the three real ideal answers, once with a **question-only** rubric — so the
difference is attributable to the ideal answers alone.

| response | relevance, full rubric | relevance, question-only |
|---|---|---|
| good (quiet) | 79.1 | 81.4 |
| good (big party) | 83.1 | 93.9 |
| off-topic | 21.6 | 24.4 |
| prompt echo | 59.6 | 68.3 |
| vague | 56.7 | 54.2 |
| drifts off topic | 46.7 | 49.9 |
| repetitive padding | 35.5 | 37.3 |

**Separation between genuine and gaming responses: 37.1 with ideal answers, 40.9
without.** The question-only rubric is *better* by 3.8 points. The `off_topic` flag
scored identically under both.

I had told the client the Ideal Answers sheet was "the highest-value single item" they
could send. On this evidence that was wrong, and I have said so rather than quietly
reordering the list.

**Why it is not surprising in hindsight.** `shareability` for the birthday question
measures 0.066, and the scorer is built to down-weight content matching in proportion to
it (ADR-007). The mechanism is working exactly as designed — it looked at three ideal
answers, concluded their content was not reusable, and declined to use it. The features
that survive (`element_coverage`, `specificity`, `content_novelty`) all derive from the
question and the response.

**What the test does not establish.** Only one question was testable, because it is the
only one whose ideal answers were legible. It is a *personal* prompt, where the whole
argument predicted low shareability. The eight **opinion** questions are where content
should be genuinely reusable and where ideal answers should earn their place. That
remains untested, and is now the only reason to want the sheet.

### ASR word-error rate on Indian-accented English
Listed as an unmeasured gap. Svarah ships reference transcripts, so it was measurable
all along.

| | |
|---|---|
| corpus WER | **4.8%** |
| median per-utterance | **0.0%** |
| p90 per-utterance | 20.7% |
| utterances above 30% WER | 1 of 38 |

Whisper large-v3 handles Indian-accented English well, and the transcription layer is
not the bottleneck anyone would assume it to be.

**Caveat that matters:** Svarah is *read* speech — clean, well-formed sentences. Real
responses are spontaneous, disfluent, and recorded on candidate hardware. 4.8% is a
floor, not an expectation. It does establish that the accent itself is not the problem.
