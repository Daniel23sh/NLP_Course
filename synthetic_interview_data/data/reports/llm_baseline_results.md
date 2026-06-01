# LLM Baseline Results

These are initial real API LLM baseline results on project-team reviewed evaluation files.

## Purpose

Evaluate zero-shot and few-shot LLM prompting infrastructure using the same metrics as the classical baselines.

## Dataset Splits Used

| Split | Path | Records |
| --- | --- | ---: |
| train | `data/processed/train.jsonl` | 265 |
| dev | `data/reviewed/dev_project_team_reviewed.jsonl` | 64 |
| test | `data/reviewed/test_project_team_reviewed.jsonl` | 106 |
| ood | `data/reviewed/ood_project_team_reviewed.jsonl` | 55 |

## Model And Prompt Configuration

- Run type: `real-api`
- Model: `gpt-5.4-mini`
- Prompt version: `llm_baseline_v1`
- Modes: `zero-shot`, `few-shot`
- Few-shot k: `3`

## Results Summary

| Mode | Split | Coverage | Mean Exact Accuracy | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | dev | 1.0000 | 0.7708 | 0.7150 | 0.2474 | 0.9024 |
| zero-shot | test | 1.0000 | 0.6698 | 0.5789 | 0.3318 | 0.8163 |
| zero-shot | ood | 1.0000 | 0.6121 | 0.2792 | 0.4182 | 0.8753 |
| few-shot | dev | 1.0000 | 0.7448 | 0.7193 | 0.2812 | 0.9412 |
| few-shot | test | 1.0000 | 0.6352 | 0.5808 | 0.3664 | 0.8092 |
| few-shot | ood | 1.0000 | 0.5848 | 0.2780 | 0.5121 | 0.8518 |

## Method Comparison

| Method | Split | Mean Exact Accuracy | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| majority | dev | 0.3411 | 0.2372 | 0.9453 | 0.4667 |
| majority | test | 0.2830 | 0.1852 | 1.1384 | 0.4422 |
| majority | ood | 0.2242 | 0.0956 | 1.9424 | 0.3333 |
| tfidf_logistic_regression | dev | 0.7318 | 0.6036 | 0.2917 | 0.8121 |
| tfidf_logistic_regression | test | 0.6038 | 0.5020 | 0.4182 | 0.7646 |
| tfidf_logistic_regression | ood | 0.3061 | 0.2621 | 0.7848 | 0.8636 |
| zero-shot LLM | dev | 0.7708 | 0.7150 | 0.2474 | 0.9024 |
| zero-shot LLM | test | 0.6698 | 0.5789 | 0.3318 | 0.8163 |
| zero-shot LLM | ood | 0.6121 | 0.2792 | 0.4182 | 0.8753 |
| few-shot LLM | dev | 0.7448 | 0.7193 | 0.2812 | 0.9412 |
| few-shot LLM | test | 0.6352 | 0.5808 | 0.3664 | 0.8092 |
| few-shot LLM | ood | 0.5848 | 0.2780 | 0.5121 | 0.8518 |

## Prediction Coverage

- `zero-shot` on `dev`: 64 / 64 successful predictions; 0 parse or prediction failures.
- `zero-shot` on `test`: 106 / 106 successful predictions; 0 parse or prediction failures.
- `zero-shot` on `ood`: 55 / 55 successful predictions; 0 parse or prediction failures.
- `few-shot` on `dev`: 64 / 64 successful predictions; 0 parse or prediction failures.
- `few-shot` on `test`: 106 / 106 successful predictions; 0 parse or prediction failures.
- `few-shot` on `ood`: 55 / 55 successful predictions; 0 parse or prediction failures.

## Classical Baseline Reference

Existing majority and TF-IDF summary rows are included above and in the JSON report.

## OOD Observations

- Zero-shot is stronger overall than few-shot on exact accuracy and MAE in this run.
- OOD is the hardest split by low/mid/high macro-F1 for both LLM modes.
- Weak-aspect F1 remains comparatively strong on OOD (0.8518-0.8753) even while exact score metrics drop.

## Limitations

- These are initial baseline results, not final deployment performance.
- Real API runs may incur cost and require `OPENAI_API_KEY` plus a selected model.
- Few-shot examples are selected deterministically from train only.
- The evaluation labels are project-team reviewed, not external expert annotations.

## Next Experiments

- Perform error analysis by aspect, split, project domain, and answer length.
- Compare LLM behavior against majority and TF-IDF baselines using representative mistakes.
- Add and evaluate a supervised encoder baseline.
