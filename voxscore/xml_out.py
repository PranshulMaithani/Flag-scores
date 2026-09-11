"""XML (and JSON) result serialisation.

The client asked for XML carrying everything. Because every score ships with the
features that produced it and a one-sentence explanation, the document is
self-explaining -- which is how the candidate-appeals requirement is satisfied
without a second reporting path.

A JSON view of the identical structure is written alongside, since downstream
consumers generally prefer it and the cost is one extra serialiser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from voxscore import __version__


@dataclass
class ItemResult:
    """Everything known about one scored response."""

    item_id: str
    question_id: str | None = None
    question_text: str | None = None
    transcript: str = ""
    duration_s: float = 0.0

    scores: dict = field(default_factory=dict)        # name -> CategoryScore
    flags: list = field(default_factory=list)         # FlagResult
    features: dict[str, dict[str, float]] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    quality_warnings: list[str] = field(default_factory=list)
    explanations: dict[str, str] = field(default_factory=dict)
    rubric_summary: dict | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    audio_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "duration_s": round(self.duration_s, 3),
            "transcript": self.transcript,
            "scores": {k: v.as_dict() for k, v in self.scores.items()},
            "flags": [f.as_dict() for f in self.flags],
            "quality": {k: round(v, 4) for k, v in self.quality.items()},
            "quality_warnings": self.quality_warnings,
            "audio_warnings": self.audio_warnings,
            "explanations": self.explanations,
            "features": {
                block: {k: round(v, 4) for k, v in feats.items()}
                for block, feats in self.features.items()
            },
            "rubric": self.rubric_summary,
            "provenance": self.provenance,
        }


def _num(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") or "0"


def item_to_xml(result: ItemResult) -> ET.Element:
    """Build the <item> element for one response."""
    item = ET.Element("item", {"id": result.item_id})
    if result.question_id:
        item.set("questionId", result.question_id)
    item.set("durationSeconds", _num(result.duration_s))

    if result.question_text:
        ET.SubElement(item, "question").text = result.question_text
    ET.SubElement(item, "transcript").text = result.transcript

    scores_el = ET.SubElement(item, "scores", {"scale": "0-100"})
    for name, cs in result.scores.items():
        s = ET.SubElement(scores_el, "score", {
            "category": name,
            "value": _num(cs.score),
            "confidence": _num(cs.confidence),
        })
        if name in result.explanations:
            ET.SubElement(s, "explanation").text = result.explanations[name]
        contribs = ET.SubElement(s, "contributions")
        for k, v in sorted(cs.contributions.items(), key=lambda kv: -kv[1]):
            ET.SubElement(contribs, "contribution", {"feature": k, "value": _num(v)})
        for note in cs.notes:
            ET.SubElement(s, "note").text = note

    flags_el = ET.SubElement(item, "flags", {
        "scale": "0-100",
        "note": "continuous scores; thresholds are recommendations and are tunable",
    })
    for f in result.flags:
        fe = ET.SubElement(flags_el, "flag", {
            "name": f.name,
            "score": _num(f.score),
            "threshold": _num(f.threshold),
            "fired": "true" if f.fired else "false",
        })
        if f.evidence:
            ET.SubElement(fe, "evidence").text = f.evidence
        fd = ET.SubElement(fe, "features")
        for k, v in f.features.items():
            ET.SubElement(fd, "feature", {"name": k, "value": _num(v)})

    q = ET.SubElement(item, "quality", {
        "scorable": "true" if result.quality.get("scorable", 1.0) else "false",
        "confidence": _num(result.quality.get("quality_confidence", 0.0)),
    })
    for k, v in result.quality.items():
        if k not in ("scorable", "quality_confidence"):
            ET.SubElement(q, "metric", {"name": k, "value": _num(v)})
    for w in result.quality_warnings + result.audio_warnings:
        ET.SubElement(q, "warning").text = w

    feats_el = ET.SubElement(item, "features")
    for block, feats in result.features.items():
        be = ET.SubElement(feats_el, "block", {"name": block})
        for k, v in feats.items():
            ET.SubElement(be, "feature", {"name": k, "value": _num(v)})

    if result.rubric_summary:
        r = ET.SubElement(item, "rubric")
        r.set("shareability", _num(result.rubric_summary.get("shareability", 0.0)))
        for m in result.rubric_summary.get("required_moves", []):
            ET.SubElement(r, "requiredMove").text = m
        for e in result.rubric_summary.get("question_terms", [])[:20]:
            ET.SubElement(r, "questionTerm").text = e

    return item


def write_xml(results: list[ItemResult], path: str | Path, run_meta: dict | None = None) -> Path:
    """Write a full results document."""
    root = ET.Element("voxscoreResults", {
        "version": __version__,
        "itemCount": str(len(results)),
    })
    meta = ET.SubElement(root, "run")
    for k, v in (run_meta or {}).items():
        ET.SubElement(meta, "meta", {"name": k, "value": str(v)})

    items = ET.SubElement(root, "items")
    for r in results:
        items.append(item_to_xml(r))

    pretty = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty, encoding="utf-8")
    return path


def write_json(results: list[ItemResult], path: str | Path, run_meta: dict | None = None) -> Path:
    """Write the same structure as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": __version__,
                "run": run_meta or {},
                "items": [r.to_dict() for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
