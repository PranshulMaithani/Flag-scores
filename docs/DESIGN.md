# Design — Feature Specification & Pipeline

Every score decomposes into named features. No feature enters a score unless it can be
explained in one sentence to a candidate contesting their result (brief §21).

## Pipeline

```
audio.npy + question_text + ideal_answers[3]
  │
  ├─ A. Conditioning ── resample 16k mono · SNR · clipping · Silero VAD segments
  ├─ B. ASR ─────────── Whisper large-v3 · transcript · per-window language posteriors
  ├─ C. Alignment ───── wav2vec2 + torchaudio CTC forced_align → word (start,end,score)
  ├─ D. Parse ───────── spaCy · POS/lemma/dep/sents · filler & disfluency tagging
  ├─ E. GEC ─────────── seq2seq correction → ERRANT typed edits
  │
  └─ F. Feature extractors → 4 score heads + 4 flag heads + quality diagnostics
```

Stages A–E run once per item and are cached to disk. Feature extraction is pure and
cheap, so the whole feature set can be recomputed without re-running ASR. This matters:
we will iterate on features many times and ASR is the expensive step.

---

## Fluency  *(from C + A)*

| Feature | Definition |
|---|---|
| `speech_rate_wpm` | words / total duration |
| `articulation_rate_wps` | words / phonation time (pauses excluded) |
| `phonation_time_ratio` | speech time / total time |
| `mean_length_of_run` | mean words between pauses >= 250 ms |
| `silent_pause_rate` | pauses >= 250 ms per minute |
| `mean_silent_pause_dur` | mean duration of those pauses |
| `long_pause_rate` | pauses >= 1 s per minute |
| `pause_dur_cv` | coefficient of variation of pause durations |
| `within_clause_pause_ratio` | pauses *not* at a clause boundary / all pauses |
| `filled_pause_rate` | um/uh/er per 100 words |
| `disfluency_repeat_rate` | immediate word/phrase repeats per 100 words |
| `false_start_rate` | abandoned constituents per 100 words |
| `artic_rate_stability` | std of windowed articulation rate |

`mean_length_of_run` and `within_clause_pause_ratio` are the two features that most
separate genuine fluency from mere speed. A fast but choppy speaker beats a slow steady
one on `speech_rate_wpm` alone — likely part of why a naive system plateaus around 0.6.

> **Accent-bias watch:** `mean_word_align_score` (acoustic clarity) is deliberately
> *excluded* from the fluency score. It correlates with accent, and accent must not be
> penalized (brief §12). Computed and retained as a **diagnostic only**.

## Grammar  *(from D + E)*

Emitted in two variants — **strict** (written norm) and **spoken-adjusted** (fillers,
fragments and self-repairs exempted before correction). Client picks whichever
correlates better with their raters.

`errors_per_100_words`, `error_free_clause_ratio`, plus ERRANT-typed rates:
article/determiner · preposition · subject-verb agreement · tense · noun number ·
word order · missing verb. Complexity: `clauses_per_sentence`, `subordination_ratio`,
`mean_dependency_distance`, `mean_sentence_length`.

The typed breakdown *is* the explanation: "3 article errors, 2 agreement errors per 100
words" is defensible to a candidate in a way a single number is not.

## Lexical  *(client calls it "vocab score")*

| Feature | Note |
|---|---|
| `mtld` | length-robust diversity — primary |
| `mattr_50` | moving-average TTR, window 50 |
| `hdd` | hypergeometric diversity |
| `mean_log_freq` | mean `wordfreq` of content lemmas; **lower = more sophisticated** |
| `pct_beyond_2k` / `pct_beyond_5k` | content words outside frequency bands |
| `awl_coverage` | Academic Word List coverage |
| `lexical_density` | content words / total |
| `verb_/noun_/adj_sophistication` | per-POS mean log frequency |
| `collocation_pmi` | mean PMI of observed bigrams vs reference corpus |

> **Length confound — the main risk here.** Responses cap at 60 s, so roughly 100–150
> words. Most diversity metrics are unstable below ~100 tokens. MTLD is chosen precisely
> because it is length-robust, but short responses must still be marked low-confidence
> rather than scored as if reliable. Raw TTR is excluded outright — it is a length
> artefact, not a vocabulary measure.

## Relevance  *(70% priority)*

### Two question families, opposite strategies

Confirmed with the client: prompts are open questions of (at least) two kinds. Real
examples:

| Family | Example | Is ideal-answer *content* reusable? |
|---|---|---|
| **Personal / experiential** | *"Talk about someone you were once close to but no longer see."* | **No** |
| **Opinion / argumentative** | *"Do you think online news makes people better informed?"* | **Yes** |

Ideal answers are 3 independent C1/C2 model responses, ~6–7 lines / ~140 words each.
Candidate responses cap at 60 s, so ~120–150 words — comparable length, which is
convenient.

For the **opinion** family, the ideal answers contain genuinely shareable argument
content — convenience, skill atrophy, social isolation — and content-point coverage
works straightforwardly.

For the **personal** family it collapses. Those ideal answers are full of *personal
specifics* — a friend's name, a city, a university, a year — that no other human being
will ever match. A candidate who gives a flawless answer about *their* lost friend
shares almost no content with an ideal answer about *someone else's*.

### Routing without a router

We do **not** hand-classify question types, and we do not train a classifier. The
decision signal is already in the data: **mutual overlap among the 3 ideal answers.**

Three model answers to an opinion prompt converge on the same arguments. Three model
answers about a lost friend share nothing but structure. So

```
shareability = mean pairwise entity-masked content overlap(I1, I2, I3)   in [0,1]
```

is a direct, per-question, zero-supervision measure of whether content matching is
valid. We compute both strategies and blend them continuously by `shareability`,
which avoids a brittle discrete router and degrades gracefully on question types
we have not seen. `shareability` is logged per question so the client can inspect it.

What a good answer and an ideal answer genuinely share is not content but **discourse
moves**: who the person was · how they met · what the closeness consisted of · how
contact was lost · how the speaker feels about it now.

> **Working hypothesis for why the incumbent vendor sits at 0.40 on relevance while managing 0.60
> elsewhere.** The other three categories are properties of the response alone and need
> no question conditioning. Relevance is the only one that requires reasoning about the
> prompt — and the obvious implementation (embed the response, embed the ideal answers,
> take cosine) is close to *uncorrelated* with relevance on personal-narrative prompts,
> because it is dominated by whose story it is. 0.40 is roughly what you would expect
> from that. This is our opening.

### Deriving the relevance target automatically

Rather than matching `R` against `Ik` directly:

1. **Entity-mask** all 3 ideal answers — replace PERSON / GPE / LOC / ORG / DATE /
   CARDINAL with type placeholders.
2. **Abstract** each masked answer into predicate–argument skeletons (discourse moves).
3. **Intersect.** A move appearing (semantically, not literally) in **>= 2 of 3** ideal
   answers is a **required move**. A move appearing in only one is personal specificity
   and is discarded. *Three independent ideal answers are exactly enough to separate
   signal from idiosyncrasy — this is the asset the incumbent vendor had and did not exploit.*
4. **Decompose `Q`** by dependency parse into required elements — *talk about* `[a
   friend]` `[you were close with]` `[but later lost touch]` → three elements.
5. **Rubric** = union of (3) and (4), derived once per question and cached.

### Features

| Feature | Definition |
|---|---|
| `move_coverage` | each required move soft-matched against R sentences; mean of maxes |
| `element_coverage` | coverage of Q-derived required elements — **always available** |
| `narrative_completeness` | presence of the discourse arc: setup → development → resolution |
| `topical_field_sim` | similarity to the masked ideal answers' lexical *field*, not content |
| `specificity` | NE / concrete-noun density — *that* specifics exist, never *which* |
| `referent_consistency` | does R actually talk about a person, for a person-type prompt |
| `nli_entailment_coverage` | masked hypotheses only |
| `nli_contradiction_rate` | self-contradiction within R |
| `topic_drift_slope` / `min_window_sim` / `pct_windows_offtopic` | similarity trajectory |
| `sim_q` | plain cosine(R, Q) — retained as a floor and as a baseline to beat |

`specificity` deserves a note: a genuinely relevant personal narrative *contains*
concrete detail, while an evasive or generic answer does not. We match on the
*existence* of entities, never their identity — which is what makes it robust to the
fact that every candidate's story is different.

### Additional features for the opinion / argumentative family

High-`shareability` questions are not just "content matching works again" — they carry
their own relevance requirements. An answer to *"do you think technology made humans
more dependent?"* is irrelevant if it never takes a position, however fluent it is.

| Feature | Definition |
|---|---|
| `stance_clarity` | does R commit to a position? NLI against generated stance hypotheses (*"The speaker thinks online news improves understanding"* / *"...did not"*), plus opinion-marker and modality density |
| `reason_count` | distinct supporting reasons, segmented on causal connectives (because, since, therefore, as a result) |
| `argument_coverage` | coverage of argument points mined from the ideal answers — **valid here**, unlike the personal family |
| `evidence_presence` | concrete examples / instances offered in support |
| `counterargument_presence` | acknowledges the opposing side (however, on the other hand, admittedly) |
| `stance_consistency` | does R hold one position, or drift/contradict across windows |

`stance_clarity` and `reason_count` are the argumentative analogues of
`narrative_completeness` — both ask *"did the candidate perform the rhetorical act the
prompt demanded?"*, which is what relevance actually means for open questions and is
precisely what a bag-of-embeddings similarity cannot see.

The two feature blocks are both computed for every item; `shareability` controls their
weighting. Features that do not apply degrade to neutral rather than to zero.

> **Client constraint made concrete:** *"don't make it too dependent that if I just add
> slight variation from ideal answer whole score tanks."* Enforced four ways:
> (a) `element_coverage` derives from `Q` alone, so an answer can score well while
> resembling no ideal answer; (b) required moves need only >= 2 of 3 ideal answers, so no
> single ideal answer can dictate the target; (c) **coverage**, not global similarity —
> extra original content is never penalized; (d) everything is matched on
> **entity-masked** representations, so differing personal detail costs nothing.

> **Confound to guard:** the ideal answers are C1/C2, i.e. the top of the proficiency
> scale. Any feature measuring raw similarity to them leaks *proficiency* into the
> *relevance* score. Masking plus coverage-based matching is the mitigation, and the
> eval plan checks it directly by testing whether relevance correlates with the other
> three scores more than the human labels do.

## Flags

All flags emit a **continuous 0–100 score**, never a hard boolean. The client sets
thresholds afterward from the ROC curves we ship.

### `prompt_read` — prompt echo to fill time

The naive feature (similarity to Q) fails, because a *good* answer is also similar to Q.
The real discriminator is **novelty**: echo is high-Q-similarity with *no added content*.

- `lcs_ratio` — longest common subsequence with Q / |R|
- `ngram_overlap_q` — R 4-grams present in Q
- `q_span_duration_coverage` — fraction of R **duration** covered by Q-matching spans
- `novelty_beyond_q` — content in R not attributable to Q  <- **primary signal**
- `paraphrase_echo` — high sentence-level embedding sim to Q with *low* lexical overlap,
  catching paraphrased echo that pure n-gram matching misses

### `repetition` — padding-level, not disfluency

- `distinct_n` for n = 1..4
- `repeated_ngram_rate`
- `gzip_compression_ratio` of the transcript — cheap and surprisingly strong
- `sent_self_sim_max` / `sent_self_sim_topk` — off-diagonal sentence similarity matrix
- `longest_repeated_span_coverage`, duration-weighted
- `audio_self_similarity` — catches literally re-played or looped audio, a cheating
  vector that transcript-level features cannot see

### `off_topic`

Derived from the relevance feature block, but calibrated as a **classifier** rather than
read off the regression score — the decision boundary and the score curve are different
objectives.

### `foreign_language`

Three independent channels, fused:

- **Acoustic LID (primary)** — Whisper language posteriors over 5 s windows / 2.5 s hop,
  giving a `p(non-English)` trajectory: max, mean, `pct_windows_nonen`, `nonen_duration_s`
- **Second acoustic opinion** — SpeechBrain VoxLingua107 ECAPA
- **Text LID** — `lingua-py` on the transcript, catching romanized or Devanagari output

> **This flag is the single biggest fairness risk in the project.** LID models routinely
> misclassify heavily accented English as the speaker's L1 — Indian-accented English as
> Hindi, Filipino-accented as Tagalog. Our population is *entirely* L2 speakers, so a
> naive implementation would systematically false-positive on exactly the candidates it
> must not. Mitigation: calibrate the decision on **accented-English negatives**
> (Svarah, FLEURS accented sets) rather than on native English, and require agreement
> across channels. Validated explicitly in the eval plan.

### `multi_speaker` *(bonus — client said "nice to have")*

ECAPA speaker embeddings over windows, then within-item embedding variance / cluster
count. Detects someone else answering, or coaching in the background.

## Quality diagnostics *(gate everything)*

`snr_db`, `clipping_rate`, `speech_presence_ratio`, `asr_avg_logprob`,
`asr_no_speech_prob`, `word_count`, `duration_s`.

A 5-word answer must not receive a confident lexical score. These let the client
separate *"bad answer"* from *"bad recording"* — a failure mode that silently corrupts
vendor scores, and one we can expose as a differentiator.

---

## Scoring layer

Features → 0–100 per category. With zero client training data, two paths:

- **(a)** fit on public corpora with proficiency labels (permitted under C5, legal review pending)
- **(b)** rank-normalize + literature-informed weights, no fitting at all

We build **(b) first as a floor that cannot overfit**, then (a) as the candidate
improvement, and report both. A `fit_calibration.py` ships so the client can refit on
their own labels later if they ever choose to.

## Evaluation — with no client data

1. **ASR sanity** — WER on Svarah (Indian English, CC-BY-4.0) and FLEURS accented sets.
   Everything downstream inherits this error, so it is measured first.
2. **Score validity** — correlate against CEFR/proficiency labels on public *spontaneous*
   L2 corpora (Speak & Improve 2025, ICNALE).
3. **Flags — synthetic edge-case construction.** We can manufacture labelled positives:
   - `prompt_read`: responses that are the question re-read or paraphrased
   - `repetition`: genuine responses with segments duplicated or looped
   - `foreign_language`: FLEURS Hindi/Tagalog/Swahili/Yoruba spliced into English
     responses at **controlled proportions**, yielding a proportion-vs-score curve that
     directly drives threshold selection
   - `off_topic`: genuine responses paired with mismatched questions

   Output: ROC/PR curves plus recommended thresholds. This *is* the flag deliverable.
4. **Accent fairness audit** — per-accent score distributions at matched proficiency,
   checking for systematic penalty. Not requested; included because brief §12 requires it
   to be true and nobody has checked whether the incumbent's scores are clean.

---

## As-built deltas

The spec above is the design as written before implementation. Measurement changed
several parts of it. Recorded here so the document does not drift from the code;
full reasoning is in `JOURNAL.md`.

| Spec said | As built | Why |
|---|---|---|
| Relevance = weighted mean of features | **anchor x engagement**, multiplicative | `specificity` and `content_novelty` are topic-blind, so an off-topic passage dense with detail scored 73/100 while the off-topic flag fired at 98. Only a product can express a disqualifying condition. |
| `shareability` from sentence-embedding agreement | **informativeness-weighted content-lemma overlap** | The embedding version ranked the two question families backwards, twice. Bi-encoder similarity conflates form with content. |
| `specificity` = named entities + numerals | **+ concrete low-frequency vocabulary** | Scored 0.000 on a strong answer that deliberately said "someone" and "she" rather than naming a friend. Penalising discretion would mark down exactly the fluent answers we want. |
| Topic drift over a fixed 4 windows | **minimum window size, abstains below 75 words** | A fixed split gave ~15-word windows whose embeddings were noise; the feature read 0.50/0.00/0.50 across three responses that were all on topic. |
| `torchaudio.functional.forced_align` | **own CTC Viterbi aligner** | No CUDA/HIP kernel, and deprecated for removal. Pinned against the torchaudio reference in tests. |
| One shared flag threshold | **per-flag thresholds from ROC curves** | `off_topic` at 55 caught only half of genuinely mismatched answers; its curve puts it at 35. `foreign_language` needed 25. |
| `foreign_language` text channel over the transcript | **over the acoustically suspect spans only** | The scoring transcript is forced to English, so text LID always read English and the channel could never fire. Re-transcribing the whole clip failed too: Whisper picks the majority language. |
| Whisper timestamps for segments | **derived from forced alignment** | Segment timestamps drift by hundreds of milliseconds, which is the difference between "no pause" and "a hesitation" at a 250 ms threshold. |

**The recurring lesson**, hit three separate times: *a channel that cannot measure must
abstain, not vote.* Returning 0.0 for "unavailable" is indistinguishable from
"measured, and fine". Every such path now collapses its weight and lets the remaining
weights renormalise.
