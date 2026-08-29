# pyright: basic
"""Render annotated images and a portable static HTML comparison report."""

from __future__ import annotations

import hashlib
import html
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from tools.model_spike.detectors import Detection


def annotate_image(image: Image.Image, detections: Sequence[Detection], destination: Path) -> None:
    """Draw labeled model predictions and save them as a JPEG image."""

    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow versions before the optional size argument.
        font = ImageFont.load_default()

    for detection in detections:
        box = detection.box.clamped(*annotated.size)
        color = _label_color(detection.label)
        coordinates = (box.x_min, box.y_min, box.x_max, box.y_max)
        draw.rectangle(coordinates, outline=color, width=4)

        label = f"{detection.label} {detection.confidence:.2f}"
        text_box = draw.textbbox((box.x_min, box.y_min), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_top = max(0.0, box.y_min - text_height - 8)
        draw.rectangle(
            (box.x_min, text_top, box.x_min + text_width + 8, text_top + text_height + 8),
            fill=color,
        )
        draw.text(
            (box.x_min + 4, text_top + 3),
            label,
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(destination, format="JPEG", quality=88, optimize=True)


def write_artifacts(run: dict[str, Any], output_directory: Path) -> None:
    """Write machine-readable results and the static report."""

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "results.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_directory / "report.html").write_text(_render_html(run), encoding="utf-8")


def _render_html(run: dict[str, Any]) -> str:
    models = run["models"]
    image_names = run["images"]
    summary_rows = "".join(_render_summary_row(model) for model in models)
    comparison_rows = "".join(_render_image_row(image_name, models) for image_name in image_names)
    prompt_badges = "".join(f"<span class='badge'>{html.escape(prompt)}</span>" for prompt in run["prompts"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Object Detection Model Spike</title>
  <style>
    :root {{ color-scheme: light dark; --border: #77808c55; --muted: #7b8490; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 28px; line-height: 1.45; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ color: var(--muted); margin-bottom: 14px; }}
    .badge {{ display: inline-block; border: 1px solid var(--border); border-radius: 999px; padding: 3px 9px; margin: 2px; font-size: 12px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; margin: 18px 0 30px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 900px; }}
    th, td {{ border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); padding: 10px; vertical-align: top; text-align: left; }}
    th:last-child, td:last-child {{ border-right: 0; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ position: sticky; top: 0; background: Canvas; z-index: 1; }}
    .comparison td {{ width: 260px; }}
    .comparison img {{ display: block; width: 100%; min-width: 240px; border-radius: 6px; background: #111; }}
    .zoomable {{ display: block; cursor: zoom-in; }}
    .caption {{ margin-top: 8px; font-size: 13px; }}
    .labels {{ color: var(--muted); overflow-wrap: anywhere; }}
    .error {{ color: #d33; white-space: pre-wrap; overflow-wrap: anywhere; }}
    code {{ font-size: 12px; }}
    dialog {{ max-width: 96vw; max-height: 96vh; border: 0; padding: 0; background: transparent; }}
    dialog::backdrop {{ background: rgb(0 0 0 / 85%); }}
    .lightbox-frame {{ position: relative; padding: 12px; border-radius: 10px; background: #111; color: #fff; }}
    .lightbox-frame img {{ display: block; max-width: 92vw; max-height: 84vh; object-fit: contain; border-radius: 6px; }}
    .lightbox-caption {{ padding: 10px 42px 0 2px; font-size: 14px; }}
    .lightbox-close {{ position: absolute; top: 16px; right: 16px; width: 36px; height: 36px; border: 0; border-radius: 999px; background: rgb(0 0 0 / 72%); color: #fff; font-size: 26px; line-height: 1; cursor: pointer; }}
  </style>
</head>
<body>
  <h1>Object Detection Model Spike</h1>
  <div class="meta">Generated {html.escape(run["created_at"])} · device: <code>{html.escape(run["device"])}</code> · one timed inference per image after warm-up · class-aware NMS IoU: {run["nms_iou"]:.2f}</div>
  <div>{prompt_badges}</div>

  <h2>Run summary</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Model</th><th>Threshold</th><th>Load</th><th>Median inference across sample images (after warm-up)</th><th>Status</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>
  </div>

  <h2>Visual comparison</h2>
  <div class="table-wrap comparison">
    <table>
      <thead><tr>{"".join(f"<th>{html.escape(model['display_name'])}</th>" for model in models)}</tr></thead>
      <tbody>{comparison_rows}</tbody>
    </table>
  </div>

  <p class="meta">This is a qualitative feasibility spike over synthetic sample images, not a statistically meaningful accuracy benchmark. Confidence scores are not calibrated across model families.</p>

  <dialog id="lightbox" aria-label="Enlarged detection result">
    <div class="lightbox-frame">
      <button class="lightbox-close" type="button" aria-label="Close enlarged image">×</button>
      <img alt="">
      <div class="lightbox-caption"></div>
    </div>
  </dialog>
  <script>
    const lightbox = document.querySelector("#lightbox");
    const lightboxImage = lightbox.querySelector("img");
    const lightboxCaption = lightbox.querySelector(".lightbox-caption");

    document.querySelectorAll(".zoomable").forEach((link) => {{
      link.addEventListener("click", (event) => {{
        event.preventDefault();
        lightboxImage.src = link.href;
        lightboxImage.alt = link.dataset.caption;
        lightboxCaption.textContent = link.dataset.caption;
        lightbox.showModal();
      }});
    }});

    lightbox.querySelector(".lightbox-close").addEventListener("click", () => lightbox.close());
    lightbox.addEventListener("click", (event) => {{
      if (event.target === lightbox) lightbox.close();
    }});
    lightbox.addEventListener("close", () => {{
      lightboxImage.src = "";
    }});
  </script>
</body>
</html>
"""


def _render_summary_row(model: dict[str, Any]) -> str:
    timings = [image["inference_ms"] for image in model["images"] if image.get("inference_ms") is not None]
    median = f"{statistics.median(timings):.0f} ms" if timings else "—"
    load = f"{model['load_ms']:.0f} ms" if model.get("load_ms") is not None else "—"
    error = model.get("error")
    status = f"<span class='error'>{html.escape(error)}</span>" if error else "Completed"
    return (
        "<tr>"
        f"<td>{html.escape(model['display_name'])}</td>"
        f"<td>{model['threshold']:.2f}</td>"
        f"<td>{load}</td>"
        f"<td>{median}</td>"
        f"<td>{status}</td>"
        "</tr>"
    )


def _render_image_row(image_name: str, models: Sequence[dict[str, Any]]) -> str:
    cells: list[str] = []
    for model in models:
        result = next((item for item in model["images"] if item["image"] == image_name), None)
        if result is None:
            cells.append("<td class='error'>Not run</td>")
            continue
        if result.get("error"):
            cells.append(f"<td class='error'>{html.escape(result['error'])}</td>")
            continue

        counts = Counter(item["label"] for item in result["detections"])
        labels = ", ".join(f"{label} × {count}" for label, count in sorted(counts.items())) or "No detections"
        path = html.escape(result["annotated_image"])
        caption = f"{model['display_name']} on {image_name}"
        cells.append(
            f"<td><a class='zoomable' href='{path}' data-caption='{html.escape(caption)}'>"
            f"<img loading='lazy' src='{path}' alt='{html.escape(caption)}'></a>"
            f"<div class='caption'><strong>{html.escape(image_name)}</strong> · {result['inference_ms']:.0f} ms · {len(result['detections'])} detections</div>"
            f"<div class='caption labels'>{html.escape(labels)}</div></td>"
        )
    return f"<tr>{''.join(cells)}</tr>"


def _label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (
        70 + digest[0] % 150,
        70 + digest[1] % 150,
        70 + digest[2] % 150,
    )
