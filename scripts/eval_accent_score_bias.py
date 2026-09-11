"""Does accent bias the four category SCORES, not just the flag?

The flag audit is not the whole fairness question. African-accented English
transcribes at 5.3% word-error rate against 0.8% for native speakers, and every
category is computed from the transcript — so ASR error is a channel through
which accent can penalise grammar, lexical and fluency without anyone noticing.

Speech Accent Archive makes this directly measurable, because every speaker reads
**the same paragraph**. Identical words go in. Any difference in grammar or
lexical score between accent groups is therefore not a difference in language
ability — it is the system penalising an accent.

Fluency is reported but cannot be interpreted the same way: reading rate and
pausing genuinely differ between readers, and this is read speech rather than a
spontaneous answer. Relevance is omitted entirely, since the paragraph answers no
question.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_accent_score_bias.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.asr.align import align_words
from voxscore.asr.whisper_asr import WhisperASR
from voxscore.config import RAW_DIR, REPORTS_DIR, SAMPLE_RATE, device_report
from voxscore.features.fluency import fluency_features
from voxscore.features.grammar import GrammarScorer, grammar_features
from voxscore.features.lexical import lexical_features
from voxscore.scoring.aggregate import score_category
from voxscore.utils import textproc as tp

GROUPS = ["native_en", "indian", "filipino", "african"]
LIMIT = 30

REFERENCE = (
    "please call stella ask her to bring these things with her from the store "
    "six spoons of fresh snow peas five thick slabs of blue cheese and maybe a "
    "snack for her brother bob we also need a small plastic snake and a big toy "
    "frog for the kids she can scoop these things into three red bags and we "
    "will go meet her wednesday at the train station"
)


# Punctuated form, for scoring. The lowercase unpunctuated REFERENCE above is
# only correct for word-error-rate computation.
REFERENCE_PUNCTUATED = (
    "Please call Stella. Ask her to bring these things with her from the store: "
    "six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a "
    "snack for her brother Bob. We also need a small plastic snake and a big toy "
    "frog for the kids. She can scoop these things into three red bags, and we "
    "will go meet her Wednesday at the train station."
)


def norm(t: str) -> str:
    t = re.sub(r"[^a-z0-9' ]", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    print(device_report(), "\n", flush=True)
    import jiwer

    asr = WhisperASR()
    nlp = tp.get_nlp()
    gec = GrammarScorer()

    # Ceiling: the same paragraph scored from a clean transcript.
    #
    # NOTE: the first version passed the WER-normalised REFERENCE here, which is
    # lowercased and stripped of punctuation for word-error scoring. spaCy then
    # parses it as one enormous run-on sentence, so the grammar complexity
    # features collapse and the "ceiling" came out at 48.2 -- BELOW the 71.0 that
    # real transcripts scored. A ceiling under the measurements is not a ceiling.
    # Uses the punctuated form now. The group-vs-group comparison below was never
    # affected, since every group is compared to the native control rather than
    # to this number.
    ref_parsed = tp.parse(REFERENCE_PUNCTUATED, nlp)
    ceiling = {
        "grammar": score_category("grammar", grammar_features(ref_parsed, gec, "spoken")).score,
        "lexical": score_category("lexical", lexical_features(ref_parsed)).score,
    }
    print(f"ceiling from a perfect transcript:  grammar {ceiling['grammar']:.1f}  "
          f"lexical {ceiling['lexical']:.1f}\n", flush=True)

    results: dict[str, dict] = {}
    print("=" * 86)
    print("SCORE BIAS BY ACCENT - identical spoken content, so any gap is accent penalty")
    print("=" * 86)
    print(f"\n{'group':12s}{'n':>4s}{'WER':>8s}{'grammar':>10s}{'lexical':>10s}"
          f"{'fluency':>10s}{'g-gap':>8s}{'l-gap':>8s}")
    print("-" * 86)

    for grp in GROUPS:
        files = sorted((RAW_DIR / "eval" / f"saa_{grp}").glob("*.npy"))[:LIMIT]
        if not files:
            continue
        g, l, fl, w = [], [], [], []
        for f in files:
            audio = np.load(f)
            res = asr.transcribe(audio, language="en")
            if not res.text.strip():
                continue
            p = tp.parse(res.text, nlp)
            words = align_words(audio, res.text)
            g.append(score_category("grammar", grammar_features(p, gec, "spoken")).score)
            l.append(score_category("lexical", lexical_features(p)).score)
            fl.append(score_category(
                "fluency", fluency_features(words, p, len(audio) / SAMPLE_RATE)).score)
            w.append(float(jiwer.wer(REFERENCE, norm(res.text))))

        results[grp] = {
            "n": len(g), "wer": float(np.mean(w)),
            "grammar": float(np.mean(g)), "lexical": float(np.mean(l)),
            "fluency": float(np.mean(fl)),
            "grammar_sd": float(np.std(g)), "lexical_sd": float(np.std(l)),
        }
        base = results.get("native_en", results[grp])
        print(f"{grp:12s}{len(g):4d}{np.mean(w):8.1%}{np.mean(g):10.1f}{np.mean(l):10.1f}"
              f"{np.mean(fl):10.1f}{np.mean(g) - base['grammar']:+8.1f}"
              f"{np.mean(l) - base['lexical']:+8.1f}", flush=True)

    print("\nThe paragraph is identical across every speaker, so grammar and lexical")
    print("gaps are transcription error reaching the score. Fluency legitimately varies")
    print("between readers and is shown for context only; relevance is omitted because")
    print("the paragraph answers no question.")

    out = REPORTS_DIR / "accent_score_bias.json"
    out.write_text(json.dumps({"ceiling": ceiling, "groups": results}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
