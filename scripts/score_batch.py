"""Score a directory of .npy responses. The main entry point for client runs.

    python scripts/score_batch.py --input data/responses --questions questions.json \
                                  --out reports/results

Writes ``results.xml``, ``results.json``, and ``features.csv``. The CSV is the
input to ``scripts/fit_calibration.py`` and is also the easiest thing to join
against your own labels for a correlation check.

Designed for a batch run that must not lose a job to one corrupt file: items that
fail are recorded with the reason and the run continues.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxscore.config import PipelineConfig, device_report
from voxscore.pipeline import Pipeline
from voxscore.scoring.normalise import centre_by_question, summarise
from voxscore.utils.audio_io import load_npy_item, load_questions
from voxscore.xml_out import write_json, write_xml


def write_features_csv(results, path: Path, centred: bool = True) -> Path:
    """Flat feature table: one row per item, one column per feature.

    When ``centred``, also emits ``<category>_centred`` columns z-scored within
    each question. Relevance ranks perfectly *within* a question but only 0.964
    when questions are pooled, so if you compute one correlation over all items
    the cross-question scale difference costs you. Both columns ship because
    which one is right depends on whether your raters graded each response
    against its own prompt or on one absolute scale -- compare them and find out.
    """
    cols: list[str] = []
    for r in results:
        for block, feats in r.features.items():
            for k in feats:
                name = f"{block}__{k}"
                if name not in cols:
                    cols.append(name)
    cats = ("grammar", "lexical", "fluency", "relevance")
    rows_for_centring = [
        {"question_id": r.question_id, "scorable": int(r.quality.get("scorable", 0)),
         **{c: (r.scores[c].score if c in r.scores else None) for c in cats}}
        for r in results
    ]
    if centred:
        centre_by_question(rows_for_centring)

    score_cols = [f"score__{c}" for c in cats]
    if centred:
        score_cols += [f"score__{c}_centred" for c in cats]
    flag_cols = [f"flag__{f.name}" for f in (results[0].flags if results else [])]
    header = (["item_id", "question_id", "duration_s", "word_count", "scorable"]
              + score_cols + flag_cols + cols)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for idx, r in enumerate(results):
            flat = {f"{b}__{k}": v for b, feats in r.features.items() for k, v in feats.items()}
            row = [
                r.item_id, r.question_id or "", round(r.duration_s, 3),
                int(r.quality.get("word_count", 0)), int(r.quality.get("scorable", 0)),
            ]
            row += [round(r.scores[c].score, 3) if c in r.scores else "" for c in cats]
            if centred:
                cr = rows_for_centring[idx]
                row += [round(cr.get(f"{c}_centred"), 3)
                        if cr.get(f"{c}_centred") is not None else "" for c in cats]
            row += [round(f.score, 3) for f in r.flags]
            row += [round(flat.get(c, float("nan")), 6) if c in flat else "" for c in cols]
            w.writerow(row)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="directory of .npy files")
    ap.add_argument("--questions", required=True, help="questions.json")
    ap.add_argument("--out", default="reports/batch", help="output directory")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--no-grammar", action="store_true",
                    help="skip GEC (roughly 2x faster; grammar scores will be 0)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    in_dir, out_dir = Path(args.input), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.questions)
    files = sorted(in_dir.glob("*.npy"))[: args.limit]
    if not files:
        print(f"no .npy files under {in_dir}")
        return 1

    print(device_report())
    print(f"{len(files)} items, {len(questions)} questions\n", flush=True)

    items, load_errors = [], []
    for f in files:
        try:
            items.append(load_npy_item(f, questions=questions))
        except Exception as exc:
            load_errors.append((f.name, f"{type(exc).__name__}: {exc}"))

    if load_errors:
        print(f"{len(load_errors)} files failed to load:")
        for name, err in load_errors[:10]:
            print(f"  {name}: {err}")
        print()

    missing_q = [i.item_id for i in items if not i.question_text]
    if missing_q:
        print(f"WARNING: {len(missing_q)} items have no matching question. Relevance "
              f"and the prompt_read / off_topic flags will be meaningless for these.")
        print(f"  first few: {missing_q[:5]}\n")

    pipe = Pipeline(PipelineConfig(device=args.device), load_grammar=not args.no_grammar)

    t0 = time.perf_counter()
    results = pipe.score_batch(items)
    elapsed = time.perf_counter() - t0

    audio_s = sum(i.duration_s for i in items)
    failed = [r for r in results if not r.scores]
    unscorable = [r for r in results if r.scores and not r.quality.get("scorable", 1)]

    meta = {
        "items": len(results),
        "failed": len(failed),
        "unscorable": len(unscorable),
        "elapsed_s": f"{elapsed:.1f}",
        "audio_s": f"{audio_s:.1f}",
        "realtime_factor": f"{audio_s / max(elapsed, 1e-9):.2f}",
        "device": device_report(),
    }

    xml_p = write_xml(results, out_dir / "results.xml", meta)
    json_p = write_json(results, out_dir / "results.json", meta)
    csv_p = write_features_csv(results, out_dir / "features.csv")

    # Report what per-question centring did, so it is visible rather than silent.
    ok_rows = [{"question_id": r.question_id,
                "scorable": int(r.quality.get("scorable", 0)),
                **{c: (r.scores[c].score if c in r.scores else None)
                   for c in ("grammar", "lexical", "fluency", "relevance")}}
               for r in results]
    centre_by_question(ok_rows)
    n_centred = sum(1 for r in ok_rows if str(r.get("centring", "")).startswith("z within"))
    if n_centred:
        print("")
        print(f"per-question centring applied to {n_centred}/{len(ok_rows)} items")
        print("compute your correlation against BOTH score__<cat> and "
              "score__<cat>_centred; whichever is higher tells you whether your "
              "raters graded relative to the prompt")

    print(f"\nscored {len(results)} items in {elapsed:.0f}s "
          f"({audio_s / max(elapsed, 1e-9):.2f}x realtime)")
    if failed:
        print(f"  {len(failed)} failed; see quality_warnings in the output")
    if unscorable:
        print(f"  {len(unscorable)} unscorable (silence, too short, or ASR "
              f"hallucination) - these carry score 0 and confidence 0, and should "
              f"be excluded from any correlation you compute")

    ok = [r for r in results if r.scores and r.quality.get("scorable", 0)]
    if ok:
        print(f"\n{'category':12s}{'mean':>8s}{'min':>8s}{'max':>8s}")
        print("-" * 36)
        for cat in ("grammar", "lexical", "fluency", "relevance"):
            vals = [r.scores[cat].score for r in ok if cat in r.scores]
            if vals:
                print(f"{cat:12s}{sum(vals) / len(vals):8.1f}"
                      f"{min(vals):8.1f}{max(vals):8.1f}")

        print(f"\n{'flag':20s}{'fired':>8s}{'rate':>8s}")
        print("-" * 36)
        for i, f in enumerate(ok[0].flags):
            fired = sum(1 for r in ok if r.flags[i].fired)
            print(f"{f.name:20s}{fired:8d}{fired / len(ok):8.1%}")

    print(f"\nwrote:\n  {xml_p}\n  {json_p}\n  {csv_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
