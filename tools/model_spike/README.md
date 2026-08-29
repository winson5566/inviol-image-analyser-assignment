# Object detection model spike

This qualitative spike compares four pretrained detectors on the six supplied workplace images. All candidates use fixed settings, normalized detections, and the same class-aware IoU 0.50 NMS.

The open-vocabulary prompts are `person`, `material handling vehicle`, `safety hat`, and `safety vest`. The images have no ground-truth annotations, so this is model-selection evidence rather than an accuracy benchmark.

## Decision

YOLOv8s-World-v2 was selected because it provided the best balance of relevant-object coverage, latency, and operational simplicity.

| Model | Observed coverage | Median CPU inference | Outcome |
|---|---|---:|---|
| YOLOv8n | `person` only | ~40 ms | Fixed-vocabulary baseline |
| YOLOv8s-World-v2 | All four target categories | ~69 ms | Selected |
| OWLv2 base | All four target categories | ~5.8 s | Too slow for this application |
| Grounding DINO tiny | High detection volume; no `person` detections at the selected threshold | ~9.5 s | Slower and less reliable on this sample |

Test environment: MacBook Air, Apple M4 (10-core CPU), 32 GB memory, macOS 15.7.4 (`arm64`). All reported timings used CPU inference.

Timings are indicative results from one local run, not a performance benchmark. Model definitions, prompts, and thresholds are in `config.py`.

### Forklift prompt selection

YOLOv8s-World-v2 was also tested with four prompts on three sample images containing a forklift. Values are the highest detection confidence in each image.

| Prompt | `image_01` | `image_03` | `image_05` |
|---|---:|---:|---:|
| `forklift` | 0.0310 | 0.0033 | 0.0267 |
| `forklift truck` | 0.1425 | 0.0093 | 0.0564 |
| `fork lift truck` | 0.1025 | 0.0149 | 0.0512 |
| `material handling vehicle` | **0.5487** | **0.0359** | **0.2648** |

`material handling vehicle` was selected because it produced the strongest response consistently. The application still uses `forklift` as its domain and API label; the broader phrase is only the model prompt.

## Run

From the repository root:

```bash
uv run --group model-evaluation python -m tools.model_spike.run
```

Missing weights are downloaded on first use. YOLO weights use `tools/model_spike/models/`, CLIP uses `weights/clip/`, and Transformers models use the Hugging Face cache.

## Output

Each run creates `tools/model_spike/reports/<timestamp>/` with:

- `report.html`: visual comparison
- `results.json`: normalized detections and run metadata
- `annotated/`: images with labels and bounding boxes

Each model receives one unmeasured warm-up inference followed by one timed inference per image. Generated reports and downloaded weights are gitignored; the decision above records the result used by the application.
