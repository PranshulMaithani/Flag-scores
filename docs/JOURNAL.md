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
