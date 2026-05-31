# Target Project Structure

This file is the structure guide for the next stages of the project. It is not a request to create every directory immediately. Keep the current working pipeline stable, and when adding training, evaluation, notebooks, reports, or new scripts, place new files according to this structure.

The current repository may still contain a flatter `src/` layout for the already-working data-generation pipeline. Do not reorganize that working code just for aesthetics unless there is a real need.

```text
interview_answer_scoring/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── generation_config.yaml
│   ├── rubrics.yaml
│   ├── profiles.yaml
│   ├── model_config.yaml
│   └── evaluation_config.yaml
│
├── data/
│   ├── raw/
│   │   └── full_synthetic_all.jsonl
│   │
│   ├── processed/
│   │   ├── train.jsonl
│   │   ├── dev_review_candidates.jsonl
│   │   ├── test_review_candidates.jsonl
│   │   ├── ood_test_review_candidates.jsonl
│   │   └── not_selected.jsonl
│   │
│   ├── reviewed/
│   │   ├── dev_human_reviewed.jsonl
│   │   ├── test_human_reviewed.jsonl
│   │   ├── ood_test_human_reviewed.jsonl
│   │   ├── dev_project_team_reviewed.jsonl
│   │   ├── test_project_team_reviewed.jsonl
│   │   ├── ood_project_team_reviewed.jsonl
│   │   ├── manual_review_audit.jsonl
│   │   ├── manual_review_summary.md
│   │   ├── manual_review_summary.json
│   │   └── manual_review_sheet.csv
│   │
│   └── reports/
│       ├── data_quality.json
│       ├── data_quality.md
│       ├── manual_review_quality.json
│       ├── manual_review_quality.md
│       └── course_bot_final_dataset_report.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── load_dataset.py
│   │   ├── validate_schema.py
│   │   ├── split_dataset.py
│   │   └── review_tools.py
│   │
│   ├── generation/
│   │   ├── profile_sampler.py
│   │   ├── prompt_templates.py
│   │   ├── llm_client.py
│   │   ├── generator.py
│   │   ├── labeler.py
│   │   ├── validator.py
│   │   └── postprocess.py
│   │
│   ├── baselines/
│   │   ├── rule_based.py
│   │   ├── zero_shot_llm.py
│   │   └── few_shot_llm.py
│   │
│   ├── models/
│   │   ├── train_encoder.py
│   │   ├── predict_encoder.py
│   │   ├── dataset_module.py
│   │   └── model_heads.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate_predictions.py
│   │   ├── weak_aspect_metrics.py
│   │   ├── ordinal_metrics.py
│   │   └── error_analysis.py
│   │
│   └── utils/
│       ├── io.py
│       ├── constants.py
│       └── logging_utils.py
│
├── experiments/
│   ├── 01_rule_based_baseline.yaml
│   ├── 02_zero_shot_llm.yaml
│   ├── 03_few_shot_llm.yaml
│   ├── 04_encoder_multitask.yaml
│   └── 05_ood_evaluation.yaml
│
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_manual_review_analysis.ipynb
│   ├── 03_model_results.ipynb
│   └── 04_error_analysis.ipynb
│
├── scripts/
│   ├── generate_dataset.py
│   ├── create_splits.py
│   ├── prepare_manual_review.py
│   ├── train_rule_based.py
│   ├── run_zero_shot.py
│   ├── run_few_shot.py
│   ├── train_encoder.py
│   ├── evaluate_all.py
│   └── make_final_tables.py
│
├── outputs/
│   ├── predictions/
│   │   ├── rule_based/
│   │   ├── zero_shot/
│   │   ├── few_shot/
│   │   └── encoder/
│   │
│   ├── metrics/
│   │   ├── rule_based_metrics.json
│   │   ├── zero_shot_metrics.json
│   │   ├── few_shot_metrics.json
│   │   ├── encoder_metrics.json
│   │   └── comparison_table.csv
│   │
│   ├── figures/
│   │   ├── score_distribution.png
│   │   ├── model_comparison.png
│   │   ├── weak_aspect_f1.png
│   │   └── confusion_matrices/
│   │
│   └── error_analysis/
│       ├── impact_errors.csv
│       ├── personal_contribution_errors.csv
│       └── ood_errors.csv
│
├── reports/
│   ├── proposal_summary.md
│   ├── dataset_report.md
│   ├── final_results.md
│   └── presentation_outline.md
│
└── tests/
    ├── test_schema.py
    ├── test_metrics.py
    ├── test_weak_aspects.py
    ├── test_split_leakage.py
    └── test_data_loading.py
```

## How To Use This Guide

- Keep data-generation artifacts under `data/`.
- Put project-team reviewed evaluation labels and review audit files under `data/reviewed/`.
- Put training/evaluation outputs under `outputs/`, not under `data/`.
- Put course-facing summaries under `reports/`.
- Keep future baseline, model, and evaluation code separated by purpose.
- Do not move the current working generation pipeline until the training/evaluation code is ready for a broader package reorganization.
