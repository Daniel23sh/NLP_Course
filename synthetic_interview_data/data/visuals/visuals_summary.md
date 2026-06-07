# Final Visuals Summary

These files were generated from existing official_v1 reports and reviewed split files.

## Figures

- `model_comparison_mean_exact.png`: Mean exact accuracy by model and split.
- `model_comparison_mae.png`: Mean absolute error by model and split.
- `weak_aspect_f1_by_model.png`: Weak-aspect F1 by model and split.
- `ood_drop_by_model.png`: OOD deltas relative to reviewed test.
- `encoder_per_aspect_errors.png`: Encoder per-aspect error patterns.
- `dataset_split_counts.png`: Official dataset split counts.

## Tables

- `model_comparison_table.csv`: Normalized model metric comparison.
- `ood_drop_table.csv`: OOD minus reviewed-test metric deltas.
- `encoder_aspect_error_table.csv`: Encoder per-aspect MAE and severe-error rates.
- `dataset_split_table.csv`: Official train/dev/test/OOD split counts.

## Inputs

- `baseline_report`: `data/reports/baseline_results.json`
- `llm_report`: `data/reports/llm_baseline_results.json`
- `encoder_report`: `data/reports/encoder_baseline_results.json`
- `error_report`: `data/reports/error_analysis_results.json`
- `train`: `data/processed/train.jsonl`
- `dev`: `data/reviewed/dev_project_team_reviewed.jsonl`
- `test`: `data/reviewed/test_project_team_reviewed.jsonl`
- `ood`: `data/reviewed/ood_project_team_reviewed.jsonl`
