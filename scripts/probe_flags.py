"""Probe the four flags against the same response set used for relevance.

The important check is not that flags fire on gaming attempts -- it is that they
stay quiet on the two genuine answers. A flag that voids a real candidate's
assessment is far more costly than one that misses a cheat.

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/probe_flags.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.features.embed import NLI, Embedder
from voxscore.features.relevance import build_rubric, relevance_features
from voxscore.flags.detectors import (
    off_topic_flag,
    prompt_read_flag,
    repetition_flag,
)
from voxscore.utils import textproc as tp

from probe_relevance import RESPONSES  # reuse the same cases

EXPECT = {
    "good (quiet, matches ideal style)": "clean",
    "good (BIG party - bias trap)": "clean",
    "off-topic (answers a different question)": "off_topic",
    "prompt echo (fills time restating the question)": "prompt_read",
    "vague / no specifics": "clean-ish",
    "starts on topic then drifts away": "off_topic",
    "repetitive padding": "repetition",
}


def main() -> int:
    qs = json.loads(Path("data/raw/questions_sample.json").read_text(encoding="utf-8"))
    nlp = tp.get_nlp()
    emb = Embedder()
    nli = NLI()

    bday = qs["birthday"]
    rub = build_rubric("birthday", bday["text"], bday["ideal_answers"], emb, nlp)

    print("=" * 96)
    print("FLAG SCORES  (0-100, higher = more suspicious; default threshold 55)")
    print("=" * 96)
    print(f"\n{'response':40s}{'prompt_read':>13s}{'repetition':>12s}{'off_topic':>11s}   expected")
    print("-" * 96)

    rows = []
    for name, text in RESPONSES.items():
        p = tp.parse(text, nlp)
        rf = relevance_features(p, rub, emb, nli)
        pr = prompt_read_flag(p, bday["text"], rf)
        rep = repetition_flag(p, emb)
        ot = off_topic_flag(rf)
        rows.append((name, pr, rep, ot))
        mark = lambda f: f"{f.score:6.1f}{'*' if f.fired else ' '}"
        print(f"{name[:39]:40s}{mark(pr):>13s}{mark(rep):>12s}{mark(ot):>11s}   {EXPECT.get(name,'')}")

    print("\n('*' = fired at the default threshold)\n")
    print("EVIDENCE for cases that fired:")
    for name, *flags in rows:
        for f in flags:
            if f.fired:
                print(f"  {name[:44]:46s} {f.name:16s} {f.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
