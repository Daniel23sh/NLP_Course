# Baseline Experiment Results

These are initial baseline results on project-team reviewed evaluation files.

## Purpose

Establish simple reference points before adding LLM or supervised encoder baselines.

## Dataset Splits Used

| Split | Path | Records |
| --- | --- | ---: |
| train | `data/processed/train.jsonl` | 265 |
| dev | `data/reviewed/dev_project_team_reviewed.jsonl` | 64 |
| test | `data/reviewed/test_project_team_reviewed.jsonl` | 106 |
| ood | `data/reviewed/ood_project_team_reviewed.jsonl` | 55 |

Training uses synthetic accepted training data. Evaluation uses project-team reviewed evaluation files.

## Baselines Implemented

- `majority`: predicts the most common training score for each aspect.
- `tfidf_logistic_regression`: answer-only TF-IDF features with one logistic regression classifier per aspect.

## Metrics Used

- Per-aspect exact accuracy, macro-F1, weighted-F1, MAE, and low/mid/high macro-F1.
- Weak-aspect precision, recall, and F1 where weak means score `<= 2`.
- Per-aspect confusion matrices are included in the JSON report.

## Results Summary

| Model | Split | Mean Exact Accuracy | Mean Macro-F1 | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| majority | dev | 0.3411 | 0.0974 | 0.2372 | 0.9453 | 0.4667 |
| majority | test | 0.2830 | 0.0719 | 0.1852 | 1.1384 | 0.4422 |
| majority | ood | 0.2242 | 0.0533 | 0.0956 | 1.9424 | 0.3333 |
| tfidf_logistic_regression | dev | 0.7318 | 0.4677 | 0.6036 | 0.2917 | 0.8121 |
| tfidf_logistic_regression | test | 0.6038 | 0.3596 | 0.5020 | 0.4182 | 0.7646 |
| tfidf_logistic_regression | ood | 0.3061 | 0.0964 | 0.2621 | 0.7848 | 0.8636 |

## Per-Aspect Observations

- `majority` on `dev`: highest exact accuracy aspect was `impact`; lowest exact accuracy aspect was `technical_depth`.
- `majority` on `test`: highest exact accuracy aspect was `impact`; lowest exact accuracy aspect was `personal_contribution`.
- `majority` on `ood`: highest exact accuracy aspect was `impact`; lowest exact accuracy aspect was `personal_contribution`.
- `tfidf_logistic_regression` on `dev`: highest exact accuracy aspect was `impact`; lowest exact accuracy aspect was `problem_solving`.
- `tfidf_logistic_regression` on `test`: highest exact accuracy aspect was `impact`; lowest exact accuracy aspect was `technical_depth`.
- `tfidf_logistic_regression` on `ood`: highest exact accuracy aspect was `impact`; lowest exact accuracy aspect was `role_relevance`.

## OOD Observations

- `majority` OOD mean exact accuracy differs from reviewed test by -0.0588 (0.2242 OOD vs 0.2830 test).
- `tfidf_logistic_regression` OOD mean exact accuracy differs from reviewed test by -0.2977 (0.3061 OOD vs 0.6038 test).

## Limitations

- These are first-pass baselines, not tuned systems.
- TF-IDF uses only the answer text and does not model ordinal distance directly.
- The evaluation labels come from rubric-based project-team review of synthetic examples.

## Next Planned Baselines

- Add zero-shot LLM baseline using the same evaluation framework.
- Add few-shot LLM baseline using the same evaluation framework.
