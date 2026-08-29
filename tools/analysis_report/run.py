# pyright: basic
"""Analyse sample images and generate a static visual safety report."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from inviol_image_analyser_assignment.config import load_analysis_config
from inviol_image_analyser_assignment.services.object_detection import ObjectDetector, create_object_detector
from inviol_image_analyser_assignment.services.risk_assessment import RiskAssessmentService
from inviol_image_analyser_assignment.services.safety_detection import SafetyRuleEngine, create_safety_rule_engine
from tools.analysis_report.report import annotate_image, write_artifacts

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=_PROJECT_ROOT / "sample_images",
        help="Directory containing JPEG/PNG images (default: sample_images).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "tools" / "analysis_report" / "reports",
        help="Root directory for timestamped reports (default: tools/analysis_report/reports).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "analysis.json",
        help="Active analysis configuration (default: config/analysis.json).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the production analysis pipeline and write a visual report."""

    args = parse_args()
    image_paths = _find_images(args.images)
    if not image_paths:
        print(f"No JPEG or PNG images found in {args.images}", file=sys.stderr)
        return 2

    config = load_analysis_config(args.config.resolve())
    detector = create_object_detector(config.object_detection)
    rule_engine = create_safety_rule_engine(config.safety_rules)
    risk_assessment = RiskAssessmentService(config.risk_assessment)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    output_directory = args.output.resolve() / timestamp
    output_directory.mkdir(parents=True, exist_ok=False)
    run: dict[str, Any] = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "config": _display_path(args.config.resolve()),
        "detector": config.object_detection.detector_type.value,
        "model_source": config.object_detection.model_source,
        "supported_rule_types": list(rule_engine.supported_rule_types),
        "results": [],
    }

    for image_path in image_paths:
        result = _analyse_image(
            image_path=image_path,
            output_directory=output_directory,
            detector=detector,
            rule_engine=rule_engine,
            risk_assessment=risk_assessment,
        )
        run["results"].append(result)
        if result.get("error"):
            print(f"{image_path.name}: ERROR {result['error']}", file=sys.stderr, flush=True)
        else:
            analysis = result["analysis"]
            print(
                f"{image_path.name}: {result['elapsed_ms']:.0f} ms, "
                f"{len(analysis['detected_objects'])} detections, {len(analysis['events'])} events, "
                f"risk {analysis['overall_risk']['score']}/{analysis['overall_risk']['level']}",
                flush=True,
            )
        write_artifacts(run, output_directory)

    print(f"\nReport: {output_directory / 'report.html'}")
    print(f"Results: {output_directory / 'results.json'}")
    return 0


def _analyse_image(
    *,
    image_path: Path,
    output_directory: Path,
    detector: ObjectDetector,
    rule_engine: SafetyRuleEngine,
    risk_assessment: RiskAssessmentService,
) -> dict[str, Any]:
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        started = perf_counter()
        detection_result = detector.detect(image)
        safety_result = rule_engine.evaluate(detection_result)
        analysis_result = risk_assessment.assess(detection_result, safety_result)
        elapsed_ms = (perf_counter() - started) * 1000

        annotated_relative = Path("annotated") / f"{image_path.stem}.jpg"
        annotate_image(image, analysis_result, output_directory / annotated_relative)
        return {
            "image": image_path.name,
            "elapsed_ms": round(elapsed_ms, 3),
            "detection_latency_ms": detection_result.latency_ms,
            "annotated_image": annotated_relative.as_posix(),
            "analysis": analysis_result.model_dump(mode="json"),
            "error": None,
        }
    except Exception as error:
        return {
            "image": image_path.name,
            "elapsed_ms": None,
            "detection_latency_ms": None,
            "annotated_image": None,
            "analysis": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
