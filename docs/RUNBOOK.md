# Runbook — scoring machine

Copy-paste, in order. Each phase has a **gate**: do not start the next phase
until the gate passes, because every later step assumes the earlier one worked.

Working directory throughout: `~/nn/eval`

---

## Phase 1 — get the GPU back

**Symptom this fixes:** `device=cpu dtype=float32` in the run log, and
`nvidia-smi` reporting *"couldn't communicate with the NVIDIA driver"*.

**Cause:** an unattended upgrade moved the box from kernel 6.17 to 7.0. The
NVIDIA DKMS module (580.126.09) cannot compile against the 7.0 series, so no
module exists for the running kernel. Reinstalling that driver cannot help —
the code does not support that kernel. The fix is to boot the kernel that
already has a working module.

```bash
ENTRY=$(awk -F"'" '/menuentry /{print $2}' /boot/grub/grub.cfg | grep '6.17.0-1019-aws' | head -1)
echo "found: $ENTRY"
```

Check `found:` printed a line containing `6.17.0-1019-aws`. **If it is empty,
stop** — that kernel's boot entry is gone and this route will not work.

```bash
sudo sed -i "s/^GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1>$ENTRY\"/" /etc/default/grub
grep GRUB_DEFAULT /etc/default/grub
sudo update-grub
sudo reboot
```

Reconnect through Instance Connect (the public IP survives a reboot; it only
changes on a stop/start).

```bash
uname -r        # expect 6.17.0-1019-aws
nvidia-smi      # expect the Tesla T4 table
```

Then clear the broken package state and stop this recurring:

```bash
sudo apt-get remove --purge -y linux-image-7.0.0-1010-aws linux-headers-7.0.0-1010-aws
sudo apt-mark hold linux-aws linux-image-aws linux-headers-aws
sudo dpkg --configure -a
```

Order matters. Purging the 7.0 packages is what clears the broken state: with
nothing left trying to build NVIDIA for that kernel, `dpkg --configure -a` has
nothing to fail on. Running it first just repeats the doomed build.

The `apt-mark hold` is not optional housekeeping — an unattended upgrade is what
took the GPU away in the first place, and without the hold it will happen again
on the next restart, most likely mid-run.

> **Gate:** `nvidia-smi` shows the Tesla T4.

---

## Phase 2 — check the inputs before spending GPU time

```bash
cd ~/nn/eval
source wav2vec2_env/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True Tesla T4`. If `which python` points somewhere unexpected, you are in
the wrong environment — torch may be installed in one and the CUDA build in
neither.

Then check the question texts:

```bash
python -c "
import csv
rows = list(csv.DictReader(open('upload.csv', encoding='utf-8-sig')))
t = [r['question_text'] for r in rows]
print('items', len(t))
print('distinct', len(set(t)))
print('distinct stripped', len(set(x.strip() for x in t)))
"
```

The last run reported *135 distinct question texts for 135 items*; the 369-item
run before it reported 24. Relevance rubrics are cached per distinct text, so
135 distinct means the cache never hits and a rubric is built per item instead
of per question — a large amount of wasted GPU time.

If **distinct stripped** is much smaller than **distinct**, it is whitespace
from Excel. Normalise it:

```bash
python - <<'EOF'
import csv
rows = list(csv.DictReader(open('upload.csv', encoding='utf-8-sig')))
for r in rows:
    r['question_text'] = ' '.join(r['question_text'].split())
with open('upload.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print('normalised', len(rows), 'rows ->', len({r['question_text'] for r in rows}), 'distinct')
EOF
```

If the count is still high after stripping, the texts genuinely differ — check
for smart quotes pasted from Word, or a candidate name embedded in the prompt.

> **Gate:** distinct question count is a small number (3, or 24 as before), not
> close to the item count.

---

## Phase 3 — the run

```bash
curl -fsSL -o run_scoring.py \
  https://raw.githubusercontent.com/PranshulMaithani/Flag-scores/main/deploy/run_scoring.py
python run_scoring.py --rescore
```

`--rescore` is needed: the cached per-item results were scored while grammar was
abstaining, and cached items are otherwise reused exactly as they are.

**Two lines to check in the first minute:**

- `device=cuda ... dtype=torch.float16` — **not** bfloat16. A T4 has no bf16
  silicon; CUDA emulates it correctly and far slower than the fp16 the card has
  hardware for.
- No `GEC unavailable` after the Whisper warnings. One
  `fast tokenizer unavailable ...; using the slow one` is fine — that is the
  fallback working, not a failure.

If a GPU is present but torch cannot reach it, the script now stops with both
diagnostic commands rather than quietly spending hours on CPU. To score on CPU
deliberately, pass `--device cpu`.

> **Gate:** `voxscore_results.xlsx` exists and its `Scores` sheet has a row per item.

---

## Phase 4 — the three analyses

All three read the output workbook. None needs a GPU, and all three print
aggregates only — no transcripts, nothing identifying — so the output is safe to
share.

Run them in this order. The first is the open question; the other two refine a
number we already have.

### 4.1 Are the flags alive?

```bash
python scripts/flag_review_sheet.py --file voxscore_results.xlsx --top 25 --controls 25
```

The flags live in the **`Flags` sheet of `voxscore_results.xlsx`**, not a
separate file. The workbook has five tabs: `Scores`, `Flags`, `Diagnostics`,
`Features`, `About`.

Two different things look like "no flag output":

- **Empty or missing `Flags` sheet** — an export fault. The pipeline builds all
  four flags for every item unconditionally, so an empty sheet means the results
  carried no `flags` key, which in practice means cached JSON from an older build
  was reused. Fix with `--rescore`.
- **Numbers present, every `*_fired` column 0** — expected on a few hundred
  ordinary responses. Reciting the prompt, padding by repetition, going off topic
  and switching language are all rare.

The second is not reassurance. A flag that never fires has been shown to be
quiet, not correct — a detector wired to a constant zero is also quiet. The
script prints each flag's score distribution, which separates the two: zero
variance across the whole set means dead, not quiet.

It writes `flag_review.csv` (shuffled, 25 top-ranked plus 25 random controls per
flag) and `flag_review.key.csv`. **Keep the key file away from whoever reviews.**
Knowing an item is a flagged one is enough to make it look suspicious, and the
resulting precision number would measure the review setup rather than the
detector.

### 4.2 What is capping all four scores near 0.4?

```bash
python scripts/diagnose_validation.py --file <your merged sheet>.xlsx \
  --pred relevance,fluency,lexical,grammar \
  --true <your 4 label columns> \
  --question question_id --length duration_s
```

Column names are matched forgivingly — the typo'd headers (`lexcial`, `grammae`)
resolve on their own.

Separates four explanations that need opposite remedies: label range restriction,
between- vs within-question variance, four scores being one latent factor, and
length doing the work. It also reports how many items abstained — an abstained
channel is a constant, and constants drag a pooled correlation toward zero while
looking like poor accuracy.

### 4.3 Is the scorecard the bottleneck, or the features?

```bash
python scripts/fit_aggregation.py --file voxscore_results.xlsx --sheet features \
  --labels <your labels file> --key item_id --group anon_id \
  --true <your 4 label columns>
```

Fits a ridge from the existing features to the labels and reports **out-of-fold**
correlation next to the current unfitted score, on identical folds. Folds are
grouped by candidate: one person's three answers share an accent, a microphone
and a proficiency, so splitting them across folds would let the model recognise
the speaker rather than the speech.

Reading the output:

| pattern | meaning |
|---|---|
| own block ≫ unfitted | the hand-drawn curves were throwing signal away — cheap fix |
| own block ≈ unfitted | the features are the ceiling — expensive, but worth knowing before tuning curves that cannot help |
| all feats ≫ own block | that category's label is better predicted by *other* categories' features: the graders scored one thing and split it four ways, and four dimensions are not recoverable by anyone |

---

## Where this stands

| | incumbent | us |
|---|---|---|
| relevance | 0.32 → **0.41** (after tuning ideal answers) | **0.37** |
| fluency | 0.60 | 0.428 |
| lexical | 0.60 | 0.361 |
| grammar | 0.60 | 0.384 |

**Relevance is at parity and it is the category weighted highest.** We sit
between their untuned 0.32 and their tuned 0.41 while using no ideal answers at
all — the rubric comes from the prompt text. Their 0.41 carries a per-question
authoring and maintenance cost, and reverts toward 0.32 whenever a question
changes. Do not let gap-closing work touch that path.

**The other three are 0.2 behind.** Phase 4.3 decides whether that is cheap
(aggregation) or expensive (features).

**Flags are the real open question.** They are a large share of the stated value
and have been validated only against synthetic fixtures so far. AUC 1.000 on
generated data is evidence about the generator. That is why Phase 4.1 comes
first.

---

## Two things still needed

1. **Grader-grader agreement**, if two people scored any of the same items.
   That is the ceiling. On spontaneous speech it is commonly 0.6–0.8, and if it
   is 0.5 here then nothing reaches 0.8 — the task does not contain it. Without
   this number, 0.37 has no scale.

2. **A decision about the 350 labelled items.** They were declared test-only.
   `fit_aggregation.py` spends them. Out-of-fold CV is the honest way to do that
   and the numbers are not inflated, but "unbiased for new candidates on *these*
   questions with *these* graders" is a weaker claim than a clean holdout, and if
   a fitted number is quoted to the client that distinction has to travel with
   it. The alternative is fitting on Q25–26 and testing on Q27, which costs
   precision and keeps a genuinely untouched holdout.

---

## Quick reference

| problem | check | fix |
|---|---|---|
| `device=cpu` | `nvidia-smi` | Phase 1 |
| `GEC unavailable` | is the workbook's GEC folder complete? | `--rescore`; the fetch self-heals |
| grammar scores look perfect | `gec_available` in the Features sheet | grammar abstains rather than report a perfect score |
| flags all zero | `flag_review_sheet.py` distributions | zero variance = dead; low variance = threshold too high |
| run is slow | `dtype=` in the log | must be `float16` on a T4, never `bfloat16` |
| rubric built per item | `N distinct` in the log | Phase 2 whitespace normalisation |
