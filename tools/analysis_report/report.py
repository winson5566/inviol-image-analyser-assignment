# pyright: basic
"""Render annotated images and a portable HTML analysis report."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from inviol_image_analyser_assignment.models import AnalysisResult, Detection, RiskLevel

_RELATED_OBJECT_COLOR = "#16a34a"
_RISK_COLORS = {
    RiskLevel.LOW: "#15803d",
    RiskLevel.MEDIUM: "#eab308",
    RiskLevel.HIGH: "#dc2626",
}
_RISK_RANK = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}


@dataclass
class _EventEvidence:
    detection: Detection
    subject_rule_types: list[str] = field(default_factory=list)
    subject_risk_level: RiskLevel | None = None


def annotate_image(image: Image.Image, result: AnalysisResult, destination: Path) -> None:
    """Draw only the detection evidence involved in risk events."""

    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow versions before the optional size argument.
        font = ImageFont.load_default()

    for evidence in _event_evidence(result):
        detection = evidence.detection
        coordinates = _pixel_box(detection, annotated.width, annotated.height)
        if evidence.subject_rule_types:
            if evidence.subject_risk_level is None:
                raise ValueError("subject evidence must include a risk level")
            border_color = _RISK_COLORS[evidence.subject_risk_level]
            label = "\n".join(evidence.subject_rule_types)
            label_color = border_color
        else:
            border_color = _RELATED_OBJECT_COLOR
            label = detection.object_type.value
            label_color = _RELATED_OBJECT_COLOR
        draw.rectangle(coordinates, outline=border_color, width=7)
        _draw_label(
            draw,
            coordinates,
            label,
            label_color,
            font,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(destination, format="JPEG", quality=90, optimize=True)


def write_artifacts(run: dict[str, Any], output_directory: Path) -> None:
    """Write machine-readable results and the static HTML report."""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "results.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_directory / "report.html").write_text(_render_html(run), encoding="utf-8")


def _render_html(run: dict[str, Any]) -> str:
    image_results = run["results"]
    cards = "".join(_render_image_card(item) for item in image_results)
    supported_checks = "".join(f"<code>{html.escape(rule_type)}</code>" for rule_type in run["supported_rule_types"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sample Image Safety Analysis</title>
  <style>
    :root {{ color-scheme: light; --ink: #172033; --muted: #667085; --line: #d9dee8; --panel: #ffffff; --page: #f4f6fa; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--page); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, sans-serif; line-height: 1.45; }}
    main {{ width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 36px 0 64px; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(28px, 4vw, 44px); letter-spacing: -0.035em; }}
    h2, h3, p {{ margin-top: 0; }}
    .muted {{ color: var(--muted); }}
    .supported-checks {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 18px 0 28px; color: var(--muted); }}
    .supported-checks code {{ padding: 5px 9px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); color: var(--ink); }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 22px; }}
    .card {{ display: grid; grid-template-columns: minmax(420px, 560px) minmax(0, 1fr); align-items: start; overflow: hidden; background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 10px 30px rgb(23 32 51 / 6%); }}
    .image-link {{ display: block; grid-column: 1; grid-row: 1; background: #111827; cursor: zoom-in; }}
    .image-link img {{ display: block; width: 100%; height: auto; object-fit: contain; }}
    .content {{ grid-column: 2; grid-row: 1; padding: 22px; }}
    .heading {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 14px; }}
    .heading h2 {{ margin: 0; font-size: 20px; }}
    .risk {{ border-radius: 999px; padding: 5px 10px; color: #fff; font-size: 12px; font-weight: 750; text-transform: uppercase; white-space: nowrap; }}
    .risk-low {{ background: #15803d; }} .risk-medium {{ background: #eab308; color: #422006; }} .risk-high {{ background: #dc2626; }}
    .facts {{ display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 15px; }}
    .tag {{ padding: 4px 8px; background: #eef2f7; border-radius: 7px; color: #475467; font-size: 12px; }}
    .event {{ border-top: 1px solid var(--line); padding: 14px 0 2px; }}
    .event-title {{ display: flex; justify-content: space-between; gap: 12px; font-weight: 700; }}
    .event-title .risk {{ padding: 3px 7px; font-size: 10px; font-weight: 650; }}
    .response-title {{ margin: 16px 0 7px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
    .json-response {{ height: 180px; margin: 0; overflow: auto; padding: 14px; border: 1px solid var(--line); border-radius: 10px; background: #111827; color: #e5e7eb; line-height: 1.5; white-space: pre; }}
    .json-response code {{ color: inherit; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .error {{ margin: 18px; padding: 14px; border-radius: 10px; background: #fef2f2; color: #b91c1c; white-space: pre-wrap; }}
    dialog {{ max-width: 96vw; max-height: 96vh; border: 0; padding: 0; background: transparent; }}
    dialog::backdrop {{ background: rgb(0 0 0 / 88%); }}
    dialog img {{ display: block; max-width: 92vw; max-height: 90vh; border-radius: 10px; }}
    dialog button {{ position: fixed; top: 20px; right: 24px; border: 0; border-radius: 999px; width: 42px; height: 42px; background: #fff; font-size: 26px; cursor: pointer; }}
    @media (max-width: 900px) {{
      .card {{ grid-template-columns: minmax(0, 1fr); }}
      .image-link {{ grid-column: 1; grid-row: 1; }}
      .image-link img {{ min-height: 0; aspect-ratio: 16 / 10; }}
      .content {{ grid-column: 1; grid-row: 2; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Sample Image Safety Analysis</h1>
    <div class="supported-checks"><span>Supported safety checks</span>{supported_checks}</div>
    <section class="grid">{cards}</section>
  </main>
  <dialog id="lightbox"><button type="button" aria-label="Close">×</button><img alt=""></dialog>
  <script>
    const lightbox = document.querySelector('#lightbox');
    const enlarged = lightbox.querySelector('img');
    document.querySelectorAll('.image-link').forEach((link) => {{
      link.addEventListener('click', (event) => {{
        event.preventDefault(); enlarged.src = link.href; enlarged.alt = link.dataset.caption; lightbox.showModal();
      }});
    }});
    lightbox.querySelector('button').addEventListener('click', () => lightbox.close());
    lightbox.addEventListener('click', (event) => {{ if (event.target === lightbox) lightbox.close(); }});
  </script>
</body>
</html>
"""


def _render_image_card(item: dict[str, Any]) -> str:
    if item.get("error"):
        return f"<article class='card'><div class='error'><strong>{html.escape(item['image'])}</strong><br>{html.escape(item['error'])}</div></article>"

    analysis = item["analysis"]
    overall = analysis["overall_risk"]
    event_summaries = "".join(_render_event_summary(event) for event in analysis["events"])
    if not event_summaries:
        event_summaries = "<div class='event'><span class='muted'>No safety events</span></div>"
    analysis_json = html.escape(json.dumps(analysis, indent=2, ensure_ascii=False))
    image_path = html.escape(item["annotated_image"])
    image_name = html.escape(item["image"])

    return f"""<article class="card">
      <a class="image-link" href="{image_path}" data-caption="{image_name}"><img loading="lazy" src="{image_path}" alt="Annotated {image_name}"></a>
      <div class="content">
        <div class="heading"><h2>{image_name}</h2><span class="risk risk-{overall["level"]}">{overall["level"]} · {overall["score"]}</span></div>
        <div class="facts"><span class="tag">{item["elapsed_ms"]:.0f} ms total</span><span class="tag">{item["detection_latency_ms"]} ms detection</span><span class="tag">{len(analysis["events"])} risk events</span></div>
        {event_summaries}
        <div class="response-title">POST /analyse response</div>
        <pre class="json-response"><code>{analysis_json}</code></pre>
      </div>
    </article>"""


def _render_event_summary(event: dict[str, Any]) -> str:
    risk = event["risk"]
    return f"""<div class="event">
      <div class="event-title"><code>{html.escape(event["rule_type"])}</code><span class="risk risk-{risk["level"]}">{risk["level"]} · {risk["score"]}</span></div>
    </div>"""


def _event_evidence(result: AnalysisResult) -> list[_EventEvidence]:
    """Return unique subject and related-object evidence for all risk events."""

    evidence: list[_EventEvidence] = []
    for event in result.events:
        subject_evidence = _find_or_add_evidence(evidence, event.subject)
        if event.rule_type not in subject_evidence.subject_rule_types:
            subject_evidence.subject_rule_types.append(event.rule_type)
        if (
            subject_evidence.subject_risk_level is None
            or _RISK_RANK[event.risk.level] > _RISK_RANK[subject_evidence.subject_risk_level]
        ):
            subject_evidence.subject_risk_level = event.risk.level
        for related_object in event.related_objects:
            _find_or_add_evidence(evidence, related_object)
    return evidence


def _find_or_add_evidence(evidence: list[_EventEvidence], detection: Detection) -> _EventEvidence:
    existing = next((item for item in evidence if item.detection == detection), None)
    if existing is not None:
        return existing
    added = _EventEvidence(detection=detection)
    evidence.append(added)
    return added


def _pixel_box(detection: Detection, width: int, height: int) -> tuple[int, int, int, int]:
    box = detection.bounding_box
    return (
        max(0, min(width - 1, round(box.x_min * width))),
        max(0, min(height - 1, round(box.y_min * height))),
        max(0, min(width - 1, round(box.x_max * width))),
        max(0, min(height - 1, round(box.y_max * height))),
    )


def _draw_label(
    draw: ImageDraw.ImageDraw,
    coordinates: Sequence[int],
    label: str,
    color: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    x_min, y_min = coordinates[0], coordinates[1]
    text_box = draw.multiline_textbbox((x_min, y_min), label, font=font, spacing=3, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_top = max(0, y_min - text_height - 8)
    draw.rectangle((x_min, text_top, x_min + text_width + 8, text_top + text_height + 8), fill=color)
    draw.multiline_text(
        (x_min + 4, text_top + 3),
        label,
        fill="white",
        font=font,
        spacing=3,
        stroke_width=1,
        stroke_fill="#111827",
    )
