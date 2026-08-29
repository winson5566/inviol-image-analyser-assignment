# Sample analysis report

Run the same detector, safety rules, and risk assessment used by the `/analyse`
endpoint over every JPEG or PNG image in `sample_images/`:

```bash
uv run python -m tools.analysis_report.run
```

Each run creates a timestamped directory under `tools/analysis_report/reports/`
containing:

- `report.html` — offline visual report
- `results.json` — structured analysis responses and timing data
- `annotated/` — images showing only subjects and related objects from risk events

Custom input and output roots can be supplied when needed:

```bash
uv run python -m tools.analysis_report.run \
  --images sample_images \
  --output tools/analysis_report/reports
```
