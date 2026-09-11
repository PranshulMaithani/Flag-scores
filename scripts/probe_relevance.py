"""Probe the relevance rubric on real client data.

Validates the two load-bearing claims of ADR-006/007:
  1. shareability separates personal-narrative prompts from opinion prompts
  2. the derived rubric ranks genuine answers above gaming attempts

Run:  PYTHONPATH=. .venv/Scripts/python.exe scripts/probe_relevance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.features.embed import NLI, Embedder
from voxscore.features.relevance import build_rubric, relevance_features
from voxscore.utils import textproc as tp

# Authored by us in the client's house style (C1/C2, scripted to sound spoken) as
# a DEV FIXTURE ONLY -- the client's real ideal answers for opinion questions were
# not legible in the screenshots. Used to test that shareability behaves
# differently across families, not to fit anything.
TECH_IDEALS = [
    "I think technology has genuinely made us more dependent, and I don't think that's "
    "entirely a bad thing. Um, the obvious example is navigation, most people I know "
    "can't really read a map anymore because the phone just does it. But I'd argue "
    "that's a trade, we've given up one skill and gained a lot of time back. The real "
    "worry for me is attention, because constant notifications make deep thinking "
    "harder. So on balance it improves what we can do but weakens how we concentrate.",

    "Honestly I'd say it's made people more dependent rather than smarter. Take simple "
    "arithmetic, hardly anyone does it in their head now because a calculator is always "
    "there. Right, and the same goes for memory, we outsource phone numbers and dates "
    "to our devices completely. I do accept there's a benefit, because having instant "
    "access to information means you can learn almost anything. But dependence is the "
    "stronger effect, since most people struggle when the technology isn't available.",

    "In my view technology improves thinking in some ways and weakens it in others. It "
    "clearly helps, because you can research a topic in minutes that would have taken "
    "days in a library. Uh, on the other hand people rarely sit with a difficult "
    "problem now, they search for the answer instead. I think the dependence is real, "
    "especially for things like navigation and calculation which we've stopped "
    "practising. So my answer is that it improves access but reduces patience.",
]

RESPONSES = {
    "good (quiet, matches ideal style)":
        "For my last birthday I honestly kept it really simple. I just had dinner at "
        "home with my parents and my younger brother. My mum cooked biryani which is my "
        "favourite, and we sat and talked for a long time after eating. Um, nothing "
        "very exciting happened but I liked that. I felt quite relaxed and happy at the "
        "end of the day.",

    "good (BIG party - bias trap)":
        "My most recent birthday was honestly huge. I booked a hall and invited about "
        "forty people from my college and my office. There was loud music, a DJ, and we "
        "danced until nearly two in the morning. Uh, my closest friend gave a really "
        "embarrassing speech which everyone recorded. It was completely exhausting but I "
        "felt so celebrated and I loved every minute of it.",

    "off-topic (answers a different question)":
        "I usually help my family at home by doing small things. I clean the kitchen "
        "after dinner and I take out the rubbish every evening. Sometimes I help my "
        "sister with her homework when she is struggling with mathematics. I think it is "
        "important to share the work at home so nobody feels tired.",

    "prompt echo (fills time restating the question)":
        "So the question is asking me to share how I celebrated my most recent birthday. "
        "Um, how did I celebrate my most recent birthday. My most recent birthday, how I "
        "celebrated it. Yeah, that is what I need to talk about, the way that I "
        "celebrated my most recent birthday recently.",

    "vague / no specifics":
        "My birthday was nice. It was good. I enjoyed it a lot and it was quite fun. "
        "There were some people and we did some things together. Um, it was a good day "
        "overall and I was happy about it. Yeah it was nice.",

    "starts on topic then drifts away":
        "For my birthday I had a small dinner with my family at home and it was quite "
        "nice. My mother cooked and we ate together in the evening. Anyway, speaking of "
        "food, I think the price of vegetables has gone up a lot this year. The "
        "government should really control inflation because it affects ordinary "
        "families. Public transport is also getting expensive and the buses in my city "
        "are always late and overcrowded these days.",

    "repetitive padding":
        "For my birthday I went out for dinner with my friends. We had a really good "
        "time together. For my birthday I went out for dinner with my friends. We had a "
        "really good time together. It was a nice evening and we enjoyed it. For my "
        "birthday I went out for dinner with my friends.",
}


def main() -> int:
    qs = json.loads(Path("data/raw/questions_sample.json").read_text(encoding="utf-8"))
    nlp = tp.get_nlp()
    emb = Embedder()
    nli = NLI()

    print("loading models...", flush=True)
    _ = emb.dim

    # ---- shareability across the two families ----
    print("\n" + "=" * 74)
    print("SHAREABILITY  (does ideal-answer agreement separate the families?)")
    print("=" * 74)

    bday = qs["birthday"]
    rub_p = build_rubric("birthday", bday["text"], bday["ideal_answers"], emb, nlp)
    rub_o = build_rubric("tech_on_thinking", qs["tech_on_thinking"]["text"], TECH_IDEALS, emb, nlp)

    for name, r in [("personal (birthday, REAL)", rub_p), ("opinion (tech, fixture)", rub_o)]:
        print(f"\n{name}")
        print(f"  shareability   : {r.shareability:.4f}")
        print(f"  required moves : {len(r.required_moves)}")
        for m in r.required_moves[:4]:
            print(f"     - {m[:88]}")
        print(f"  question elements: {r.question_elements}")

    # ---- ranking of responses ----
    print("\n" + "=" * 74)
    print("RESPONSE SCORING  (birthday rubric, real ideal answers)")
    print("=" * 74)
    keys = ["sim_q", "element_coverage", "move_coverage", "profile_match",
            "specificity", "content_novelty_vs_q", "distinct_content_rate",
            "min_window_sim", "window_sim_vs_ref", "topic_drift_slope",
            "pct_windows_offtopic"]
    print(f"\n{'response':38s}" + "".join(f"{k[:11]:>13s}" for k in keys))
    print("-" * (38 + 13 * len(keys)))
    for name, text in RESPONSES.items():
        p = tp.parse(text, nlp)
        f = relevance_features(p, rub_p, emb, nli)
        print(f"{name[:37]:38s}" + "".join(f"{f.get(k, 0.0):13.3f}" for k in keys))

    print("\nKey checks:")
    print("  * shareability(opinion) should be >> shareability(personal)")
    print("  * 'BIG party' must NOT score below 'quiet' -- that is the bias trap")
    print("  * off-topic and prompt-echo must score below both good answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
