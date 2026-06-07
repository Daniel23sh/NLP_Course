# Supervised Encoder Baseline Results

These results evaluate a lightweight supervised encoder baseline on the official project splits.

## Purpose

Train an answer-only neural baseline for six ordinal rubric scores and compare behavior across dev, test, and OOD splits.

## Dataset Splits Used

| Split | Path | Records Used | Full Records |
| --- | --- | ---: | ---: |
| train | `data/processed/train.jsonl` | 265 | 265 |
| dev | `data/reviewed/dev_project_team_reviewed.jsonl` | 64 | 64 |
| test | `data/reviewed/test_project_team_reviewed.jsonl` | 106 | 106 |
| ood | `data/reviewed/ood_project_team_reviewed.jsonl` | 55 | 55 |

## Model Configuration

- Model: `distilbert-base-uncased`
- Formulation: one shared encoder with one five-class classification head per rubric aspect.
- Input features: `answer` text only.
- Labels: `final_scores` only, with scores `1`-`5` mapped to classes `0`-`4` during training.
- Epochs: `3`
- Batch size: `8`
- Learning rate: `2e-05`
- Max length: `256`
- Seed: `42`
- Device: `cpu`

## Result Summary

| Split | Mean Exact Accuracy | Mean Macro-F1 | Mean Weighted-F1 | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak Precision | Weak Recall | Weak F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 0.6823 | 0.3910 | 0.6195 | 0.6652 | 0.3516 | 0.8696 | 0.9302 | 0.8989 |
| test | 0.6777 | 0.3439 | 0.6245 | 0.5721 | 0.3428 | 0.8529 | 0.9223 | 0.8862 |
| ood | 0.2879 | 0.0813 | 0.1935 | 0.2560 | 0.7152 | 1.0000 | 0.8000 | 0.8889 |

## Per-Split Observations

- `dev`: mean exact accuracy is 0.6823; highest exact accuracy aspect is `technical_depth` and lowest is `personal_contribution`.
- `test`: mean exact accuracy is 0.6777; highest exact accuracy aspect is `impact` and lowest is `clarity`.
- `ood`: mean exact accuracy is 0.2879; highest exact accuracy aspect is `problem_solving` and lowest is `role_relevance`.
- OOD mean exact accuracy differs from reviewed test by -0.3898 (0.2879 OOD vs 0.6777 test).

## Limitations

- This is a baseline training run, not a tuned neural system.
- The model treats each aspect score as a five-way class and does not explicitly optimize ordinal distance.
- The dataset is synthetic and evaluation labels are project-team reviewed rather than external expert annotations.
- Error analysis by answer type, domain, and aspect is intentionally left for a later project stage.
