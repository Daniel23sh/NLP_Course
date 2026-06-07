# Error Analysis Report

## Purpose

Analyze existing prediction artifacts for the junior interview answer scoring task, with the supervised encoder baseline as the primary focus.

## Inputs And Model Availability

| Input | Path |
| --- | --- |
| encoder_report | `data/reports/encoder_baseline_results.json` |
| llm_report | `data/reports/llm_baseline_results.json` |
| baseline_report | `data/reports/baseline_results.json` |
| train | `data/processed/train.jsonl` |
| dev | `data/reviewed/dev_project_team_reviewed.jsonl` |
| test | `data/reviewed/test_project_team_reviewed.jsonl` |
| ood | `data/reviewed/ood_project_team_reviewed.jsonl` |

| Model Source | Status | Detail |
| --- | --- | --- |
| encoder | `available` | per-example predictions loaded |
| baseline | `available` | aggregate rows only |
| llm | `aligned` | included detailed modes: zero-shot, few-shot |

## Aggregate Results Comparison

| Model | Split | Mean Exact Accuracy | Mean Low/Mid/High Macro-F1 | Mean MAE | Weak-Aspect F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| classical:majority | dev | 0.3411 | 0.2372 | 0.9453 | 0.4667 |
| classical:majority | ood | 0.2242 | 0.0956 | 1.9424 | 0.3333 |
| classical:majority | test | 0.2830 | 0.1852 | 1.1384 | 0.4422 |
| classical:tfidf_logistic_regression | dev | 0.7318 | 0.6036 | 0.2917 | 0.8121 |
| classical:tfidf_logistic_regression | ood | 0.3061 | 0.2621 | 0.7848 | 0.8636 |
| classical:tfidf_logistic_regression | test | 0.6038 | 0.5020 | 0.4182 | 0.7646 |
| encoder | dev | 0.6823 | 0.6652 | 0.3516 | 0.8989 |
| encoder | ood | 0.2879 | 0.2560 | 0.7152 | 0.8889 |
| encoder | test | 0.6777 | 0.5721 | 0.3428 | 0.8862 |
| few-shot | dev | 0.7448 | 0.7193 | 0.2812 | 0.9412 |
| few-shot | ood | 0.5848 | 0.2780 | 0.5121 | 0.8518 |
| few-shot | test | 0.6352 | 0.5808 | 0.3664 | 0.8092 |
| zero-shot | dev | 0.7708 | 0.7150 | 0.2474 | 0.9024 |
| zero-shot | ood | 0.6121 | 0.2792 | 0.4182 | 0.8753 |
| zero-shot | test | 0.6698 | 0.5789 | 0.3318 | 0.8163 |

## Encoder Error Summary

| Split | Records | Mean Exact Accuracy | Mean MAE | Exact-All Rate | Severe Error Rate | Weak F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 64 | 0.6823 | 0.3516 | 0.1250 | 0.0339 | 0.8989 |
| test | 106 | 0.6777 | 0.3428 | 0.0849 | 0.0189 | 0.8862 |
| ood | 55 | 0.2879 | 0.7152 | 0.0000 | 0.0030 | 0.8889 |

## OOD Drop

- `classical:majority`: exact accuracy delta -0.0588; MAE delta 0.8040; weak-aspect F1 delta -0.1089.
- `classical:tfidf_logistic_regression`: exact accuracy delta -0.2977; MAE delta 0.3666; weak-aspect F1 delta 0.0990.
- `encoder`: exact accuracy delta -0.3898; MAE delta 0.3724; weak-aspect F1 delta 0.0027.
- `few-shot`: exact accuracy delta -0.0504; MAE delta 0.1457; weak-aspect F1 delta 0.0426.
- `zero-shot`: exact accuracy delta -0.0577; MAE delta 0.0864; weak-aspect F1 delta 0.0590.

## Per-Aspect Failure Patterns

- `technical_depth`: MAE 0.4933, signed error 0.1378, severe rate 0.0000; common confusions: 1->2 (55), 5->4 (24), 3->2 (16).
- `personal_contribution`: MAE 0.4089, signed error 0.2044, severe rate 0.0356; common confusions: 1->2 (35), 3->4 (13), 3->2 (12).
- `clarity`: MAE 0.4667, signed error -0.4311, severe rate 0.0000; common confusions: 4->3 (72), 5->4 (29), 3->4 (4).
- `problem_solving`: MAE 0.3778, signed error 0.0044, severe rate 0.0133; common confusions: 1->2 (42), 5->4 (26), 3->2 (8).
- `impact`: MAE 0.3689, signed error -0.3333, severe rate 0.0444; common confusions: 2->1 (40), 3->2 (9), 5->4 (8).
- `role_relevance`: MAE 0.5022, signed error 0.4844, severe rate 0.0222; common confusions: 2->3 (54), 3->4 (47), 2->4 (3).

## Top Error Examples

1. `official_strong_patch_impact_000213` (test): total absolute error 6, max error 2; missed weak: none; false weak: impact.
2. `official_weak_patch_personal_contribution_000043` (ood): total absolute error 6, max error 1; missed weak: role_relevance; false weak: none.
3. `official_weak_patch_personal_contribution_000045` (ood): total absolute error 6, max error 1; missed weak: role_relevance; false weak: none.
4. `official_high_impact_patch_impact_000019` (test): total absolute error 5, max error 3; missed weak: none; false weak: impact.
5. `official_high_impact_patch_impact_000032` (dev): total absolute error 5, max error 2; missed weak: none; false weak: none.
6. `official_weak_patch_role_relevance_000007` (ood): total absolute error 5, max error 2; missed weak: role_relevance; false weak: none.
7. `official_high_impact_patch_impact_000046` (dev): total absolute error 5, max error 1; missed weak: none; false weak: none.
8. `official_high_impact_patch_impact_000050` (dev): total absolute error 5, max error 1; missed weak: none; false weak: none.
9. `official_weak_patch_personal_contribution_000047` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
10. `official_weak_patch_personal_contribution_000053` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
11. `official_weak_patch_personal_contribution_000055` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
12. `official_weak_patch_personal_contribution_000058` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
13. `official_weak_patch_personal_contribution_000060` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
14. `official_weak_patch_personal_contribution_000061` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
15. `official_weak_patch_personal_contribution_000063` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
16. `official_weak_patch_personal_contribution_000065` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
17. `official_weak_patch_personal_contribution_000066` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
18. `official_weak_patch_personal_contribution_000069` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
19. `official_weak_patch_personal_contribution_000078` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.
20. `official_weak_patch_personal_contribution_000081` (ood): total absolute error 5, max error 1; missed weak: role_relevance; false weak: none.

## Optional LLM Comparison Notes

Detailed aligned LLM analysis is available for: `zero-shot`, `few-shot`.

## Limitations

- This report analyzes existing predictions only; it does not run new models or create new experiments.
- Classical baselines are aggregate-only here because their report does not include per-example predictions.
- Slice summaries are descriptive and may be noisy for small groups.
- No charts are generated in this stage.
