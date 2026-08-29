# pyright: basic
"""Batch-run candidate detectors and generate a static comparison report."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from tools.model_spike.config import (
    DEFAULT_MODELS,
    DEFAULT_PROMPTS,
    REPORTS_DIRECTORY,
    SAMPLE_IMAGES_DIRECTORY,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from tools.model_spike.detectors import Detector, build_detector, class_aware_nms
from tools.model_spike.report import annotate_image, write_artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a reproducible spike run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=SAMPLE_IMAGES_DIRECTORY,
        help="Directory containing JPEG/PNG images (default: sample_images).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIRECTORY,
        help="Root directory for timestamped reports (default: tools/model_spike/reports).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="Candidate models to run in order.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=list(DEFAULT_PROMPTS),
        help="Open-vocabulary class prompts.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch/Ultralytics device, for example cpu, mps, or 0 for CUDA.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="MODEL=VALUE",
        help="Override one model's global confidence threshold; may be repeated.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.50,
        help="Class-aware NMS IoU threshold applied to every adapter (default: 0.50).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.nms_iou <= 1.0:
        parser.error("--nms-iou must be between 0 and 1")
    args.thresholds = _parse_thresholds(parser, args.threshold)
    return args


def main() -> int:
    """Run selected models sequentially and write a portable static report."""

    args = parse_args()
    image_paths = _find_images(args.images)
    if not image_paths:
        print(f"No JPEG or PNG images found in {args.images}", file=sys.stderr)
        return 2

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    selected_models = tuple(args.models)
    is_full_comparison = selected_models == DEFAULT_MODELS
    run_name = timestamp if is_full_comparison else f"{timestamp}--{'-'.join(selected_models)}"
    output_directory = args.output.resolve() / run_name
    output_directory.mkdir(parents=True, exist_ok=False)

    run: dict[str, Any] = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "device": args.device,
        "nms_iou": args.nms_iou,
        "prompts": args.prompts,
        "images": [path.name for path in image_paths],
        "models": [],
    }

    for model_name in args.models:
        detector = build_detector(
            model_name,
            device=args.device,
            prompts=args.prompts,
            threshold_override=args.thresholds.get(model_name),
        )
        model_result: dict[str, Any] = {
            "name": detector.name,
            "display_name": detector.display_name,
            "threshold": detector.threshold,
            "load_ms": None,
            "error": None,
            "images": [],
        }
        run["models"].append(model_result)
        print(f"\n[{detector.display_name}] loading...", flush=True)

        try:
            load_started = perf_counter()
            detector.load()
            model_result["load_ms"] = (perf_counter() - load_started) * 1000
            print(f"[{detector.display_name}] loaded in {model_result['load_ms']:.0f} ms", flush=True)

            with Image.open(image_paths[0]) as warmup_source:
                detector.detect(warmup_source.convert("RGB"))

            for image_path in image_paths:
                image_result = _evaluate_image(
                    detector=detector,
                    image_path=image_path,
                    output_directory=output_directory,
                    nms_iou=args.nms_iou,
                )
                model_result["images"].append(image_result)
                if image_result.get("error"):
                    print(f"  {image_path.name}: ERROR {image_result['error']}", flush=True)
                else:
                    print(
                        f"  {image_path.name}: {image_result['inference_ms']:.0f} ms, "
                        f"{len(image_result['detections'])} detections",
                        flush=True,
                    )
        except Exception as error:  # Continue so one failed candidate does not discard the whole spike.
            model_result["error"] = f"{type(error).__name__}: {error}"
            print(f"[{detector.display_name}] ERROR: {model_result['error']}", file=sys.stderr, flush=True)
        finally:
            detector.close()
            write_artifacts(run, output_directory)

    report_path = output_directory / "report.html"
    print(f"\nReport: {report_path}")
    print(f"Results: {output_directory / 'results.json'}")
    return 0


def _find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _parse_thresholds(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        model_name, separator, raw_threshold = value.partition("=")
        if not separator or model_name not in DEFAULT_MODELS:
            parser.error(f"invalid --threshold {value!r}; expected MODEL=VALUE")
        try:
            threshold = float(raw_threshold)
        except ValueError:
            parser.error(f"invalid confidence threshold: {raw_threshold!r}")
        if not 0.0 <= threshold <= 1.0:
            parser.error("confidence thresholds must be between 0 and 1")
        thresholds[model_name] = threshold
    return thresholds


def _evaluate_image(
    *,
    detector: Detector,
    image_path: Path,
    output_directory: Path,
    nms_iou: float,
) -> dict[str, Any]:
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        started = perf_counter()
        detections = class_aware_nms(detector.detect(image), nms_iou)
        inference_ms = (perf_counter() - started) * 1000

        annotated_relative = Path("annotated") / detector.name / f"{image_path.stem}.jpg"
        annotate_image(image, detections, output_directory / annotated_relative)
        return {
            "image": image_path.name,
            "inference_ms": round(inference_ms, 3),
            "annotated_image": annotated_relative.as_posix(),
            "detections": [detection.to_dict() for detection in detections],
            "error": None,
        }
    except Exception as error:
        return {
            "image": image_path.name,
            "inference_ms": None,
            "annotated_image": None,
            "detections": [],
            "error": f"{type(error).__name__}: {error}",
        }


if __name__ == "__main__":
    raise SystemExit(main())
