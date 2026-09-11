"""Two measurements that answer questions we had been answering by assertion.

**1. Ideal-answer ablation.** How much do the three ideal answers actually buy?
Every response is scored twice against the same question — once with the full
rubric, once with a question-only rubric — so the difference is attributable to
the ideal answers alone. This decides whether authoring ideal answers for new
questions is worth the client's effort, and the honest answer might be "not
much".

**2. ASR word-error rate on Indian-accented English.** Everything downstream
inherits ASR error, and we had listed this as unmeasured. Svarah ships reference
transcripts, so it is measurable now.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_ablation_and_wer.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.asr.whisper_asr import WhisperASR
from voxscore.config import RAW_DIR, REPORTS_DIR, device_report
from voxscore.features.embed import NLI, Embedder
from voxscore.features.relevance import build_rubric, relevance_features
from voxscore.flags.detectors import off_topic_flag
from voxscore.scoring.aggregate import score_relevance
from voxscore.utils import textproc as tp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_relevance import RESPONSES  # noqa: E402

GOOD = {"good (quiet, matches ideal style)", "good (BIG party - bias trap)"}


def normalise(t: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — standard WER prep."""
    t = re.sub(r"[^a-z0-9' ]", " ", (t or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def ablation(emb, nli, nlp) -> dict:
    qs = json.loads(Path("data/raw/questions_sample.json").read_text(encoding="utf-8"))
    bday = qs["birthday"]

    full = build_rubric("birthday", bday["text"], bday["ideal_answers"], emb, nlp)
    qonly = build_rubric("birthday", bday["text"], [], emb, nlp)

    print("=" * 86)
    print("ABLATION - what do the three ideal answers actually buy?")
    print("=" * 86)
    print(f"\n{'response':40s}{'relevance':>22s}{'off_topic flag':>24s}")
    print(f"{'':40s}{'full':>10s}{'q-only':>12s}{'full':>12s}{'q-only':>12s}")
    print("-" * 86)

    rows, rel_full, rel_q, lab = [], [], [], []
    for name, text in RESPONSES.items():
        p = tp.parse(text, nlp)
        f_full = relevance_features(p, full, emb, nli)
        f_q = relevance_features(p, qonly, emb, nli)
        r_full = score_relevance(f_full).score
        r_q = score_relevance(f_q).score
        o_full = off_topic_flag(f_full).score
        o_q = off_topic_flag(f_q).score

        is_good = name in GOOD
        rel_full.append(r_full); rel_q.append(r_q); lab.append(is_good)
        rows.append({"response": name, "good": is_good, "rel_full": r_full,
                     "rel_qonly": r_q, "off_full": o_full, "off_qonly": o_q})
        print(f"{name[:39]:40s}{r_full:10.1f}{r_q:12.1f}{o_full:12.1f}{o_q:12.1f}")

    # Separation = how far genuine answers sit above gaming/off-topic ones.
    def gap(scores):
        s = np.array(scores); m = np.array(lab)
        return float(s[m].mean() - s[~m].mean())

    g_full, g_q = gap(rel_full), gap(rel_q)
    print(f"\n  separation (mean good - mean bad):  full {g_full:5.1f}   "
          f"question-only {g_q:5.1f}   delta {g_full - g_q:+.1f}")
    print(f"  rubric derived {len(full.required_moves)} required moves, "
          f"shareability {full.shareability:.4f}")
    print(f"  profile available: full={bool(full.profile)}  q-only={bool(qonly.profile)}")
    return {"rows": rows, "separation_full": g_full, "separation_qonly": g_q}


def wer(asr) -> dict:
    refs = json.loads((RAW_DIR / "eval" / "svarah_en_in" / "refs.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 86)
    print("ASR WORD-ERROR RATE on Indian-accented English (Svarah)")
    print("=" * 86)

    import jiwer

    hyps, gold, n = [], [], 0
    for key in sorted(refs)[:40]:
        f = RAW_DIR / "eval" / "svarah_en_in" / f"{key}.npy"
        if not f.exists():
            continue
        ref = normalise(refs[key])
        if len(ref.split()) < 3:
            continue
        hyp = normalise(asr.transcribe(np.load(f), language="en").text)
        gold.append(ref); hyps.append(hyp); n += 1
        if n <= 3:
            print(f"\n  REF: {ref[:76]}")
            print(f"  HYP: {hyp[:76]}")

    overall = float(jiwer.wer(gold, hyps))
    per_utt = [float(jiwer.wer(g, h)) for g, h in zip(gold, hyps) if g.strip()]
    print(f"\n  n = {n} utterances")
    print(f"  corpus WER      {overall:6.1%}")
    print(f"  median per-utt  {np.median(per_utt):6.1%}")
    print(f"  p90 per-utt     {np.percentile(per_utt, 90):6.1%}")
    print(f"  utterances >30% WER: {sum(1 for w in per_utt if w > 0.3)}/{n}")
    return {"n": n, "corpus_wer": overall,
            "median": float(np.median(per_utt)), "p90": float(np.percentile(per_utt, 90))}


def main() -> int:
    print(device_report(), "\n", flush=True)
    nlp, emb, nli = tp.get_nlp(), Embedder(), NLI()
    out = {"ablation": ablation(emb, nli, nlp), "wer": wer(WhisperASR())}
    p = REPORTS_DIR / "ablation_and_wer.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
