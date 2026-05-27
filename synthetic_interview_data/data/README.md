# Official Synthetic Interview Dataset

This directory contains the frozen `official_v1` synthetic dataset for the junior developer interview-answer scoring project. The dataset is ready for baseline training, but the review splits are not gold evaluation data until they are manually checked.

## Project Task

Input: a free-text answer from a junior candidate about a personal, academic, or portfolio project.

Output:

- `technical_depth`: 1-5
- `personal_contribution`: 1-5
- `clarity`: 1-5
- `problem_solving`: 1-5
- `impact`: 1-5
- `role_relevance`: 1-5
- `weak_aspects`: aspects with score `<= 2`
- `strong_aspects`: aspects with score `>= 4`
- validation metadata, labeler evidence, and validator evidence

The dataset supports model training and later model comparison for aspect-based NLP/LLM scoring of interview answers.

## Why Synthetic Data

There is no large public labeled dataset for junior developer project-experience interview answers with this six-aspect score schema. Synthetic data makes it possible to create controlled examples across weak, mid, and strong answer behaviors, including missing ownership, unclear structure, shallow technical content, low impact, and weak role relevance.

## Methodology

The pipeline uses label-after-generation:

1. Generate a candidate answer from a controlled profile.
2. Label the answer with a rubric-based LLM labeler.
3. Independently validate and rescore the answer.
4. Accept only records whose labels are reliable under agreement, evidence, leakage, and profile-success checks.
5. Export train data with accepted synthetic labels.
6. Export dev/test/OOD splits as review candidates.
7. Require manual review before final reported evaluation.

Labels are based on observable evidence in the answer, not copied from target scores.

## Official Version

The official dataset version is `official_v1`.

Older `v2`, `v3`, and `v4` paths were experimental development stages. They are not the official dataset location. Stable official files now use purpose-based folders:

- `data/raw/`: raw merged official records and intermediate generation-phase outputs.
- `data/processed/`: train/dev/test/OOD splits and processed status buckets.
- `data/reviewed/`: human-reviewed dev/test/OOD files, created after manual review.
- `data/reports/`: quality reports and course-facing summaries.

## Generation Phases

- `broad`: realistic normal, mixed-quality, and strong junior project answers.
- `weak_patch`: targeted examples for missing weak labels such as low role relevance, low technical depth, low clarity, low impact, and absent personal contribution.
- `strong_patch`: attempted high-quality examples. This phase produced only one clean accepted official example, but it is preserved for traceability.
- `high_impact_patch`: targeted repair for high-impact examples after impact high coverage was identified as the final blocker.

Current generation sources:

- `broad`: 135
- `weak_patch`: 100
- `strong_patch`: 1
- `high_impact_patch`: 25

## Final Official Counts

- Dataset size: 261
- Accepted records: 261
- Train: 129
- Dev review candidates: 31
- Test review candidates: 33
- OOD test review candidates: 29
- Not selected: 39
- Duplicate IDs: 0
- Duplicate answers: 0
- Train/test leakage: none
- OOD/train leakage: none
- Accepted labeler-validator delta `>= 2`: 0

## Score Coverage

Low means scores `1-2`, mid means score `3`, and high means scores `4-5`.

| Aspect | Low | Mid | High |
| --- | ---: | ---: | ---: |
| `technical_depth` | 119 | 64 | 78 |
| `personal_contribution` | 133 | 22 | 106 |
| `clarity` | 20 | 62 | 179 |
| `problem_solving` | 135 | 49 | 77 |
| `impact` | 205 | 30 | 26 |
| `role_relevance` | 75 | 37 | 149 |

## Impact Repair Note

High-impact coverage was the last serious blocker before baseline training. The `high_impact_patch` phase added evidence-based examples with concrete project-level outcomes such as before/after metrics, saved time, reduced manual work, QA/error reduction, final-demo usage, and improved accuracy/performance.

Final impact distribution:

- `impact = 1`: 25
- `impact = 2`: 180
- `impact = 3`: 30
- `impact = 4`: 17
- `impact = 5`: 9
- High impact total: 26

## File Guide

- `data/raw/full_synthetic_all.jsonl`: all official accepted records, including selected and not-selected records.
- `data/raw/generated/<phase>/`: intermediate outputs for `broad`, `weak_patch`, `strong_patch`, and `high_impact_patch`.
- `data/processed/train.jsonl`: clean accepted synthetic training records.
- `data/processed/dev_review_candidates.jsonl`: development review candidates; not gold until manually verified.
- `data/processed/test_review_candidates.jsonl`: test review candidates; not gold until manually verified.
- `data/processed/ood_test_review_candidates.jsonl`: out-of-domain review candidates; not gold until manually verified.
- `data/processed/not_selected.jsonl`: accepted records kept out of the selected train/dev/test/OOD splits by balancing and split logic.
- `data/processed/full_synthetic_clean_accepted.jsonl`: clean accepted records from all official generation phases.
- `data/processed/full_synthetic_manual_review.jsonl`: manual-review bucket for the final merge. Currently empty for `official_v1`.
- `data/processed/full_synthetic_profile_mismatch.jsonl`: profile-mismatch bucket for the final merge. Currently empty for `official_v1`.
- `data/processed/full_synthetic_rejected.jsonl`: rejected/audit bucket for the final merge. Currently empty for `official_v1`.
- `data/reviewed/dev_human_reviewed.jsonl`: manually reviewed development labels, created before final evaluation.
- `data/reviewed/test_human_reviewed.jsonl`: manually reviewed test labels, created before final evaluation.
- `data/reviewed/ood_test_human_reviewed.jsonl`: manually reviewed OOD labels, created before final evaluation.
- `data/reports/data_quality.md`: human-readable quality report.
- `data/reports/data_quality.json`: machine-readable quality report.
- `data/reports/course_bot_final_dataset_report.md`: course-facing final dataset summary.

## Readiness

- Usable for synthetic training: yes.
- Ready to start baseline training: yes.
- Usable for final evaluation without manual labels: no.
- Manual review required for dev/test/OOD: yes.

The train split can be used for baseline training. Dev, test, and OOD files must be manually reviewed or corrected before final reported evaluation.

## Known Limitations

- Synthetic data may contain generation-style bias.
- Exact 1-5 scoring is subjective, even with rubric evidence.
- Dev/test/OOD are review candidates, not gold labels.
- Impact remains relatively skewed low even after repair.
- Clarity score `1` is absent, but low clarity is represented by score `2`.
- Some exact extreme scores are rare, so reports should include both exact-score metrics and low/mid/high band metrics.

## Recommended Next Steps

1. Freeze `official_v1`.
2. Start baseline training.
3. Build a rule-based baseline.
4. Build zero-shot and few-shot LLM baselines.
5. Train the first supervised model.
6. Manually review dev/test/OOD before final evaluation.
7. Report exact-score metrics, low/mid/high metrics, weak-aspect metrics, and error analysis.

## Reproduction Commands

Run commands from `synthetic_interview_data/`.

Broad generation:

```bash
python3 -m src.main_generate \
  --mode openai \
  --dataset-version official_v1 \
  --generation-kind broad \
  --n 220 \
  --model gpt-5.4-mini \
  --seed 42 \
  --output-dir data/raw/generated/broad
```

Weak patch:

```bash
python3 -m src.main_generate \
  --mode openai \
  --n 0 \
  --dataset-version official_v1 \
  --generation-kind weak_patch \
  --targeted-patch \
  --patch-target-n 20 \
  --patch-max-attempts-per-target 30 \
  --model gpt-5.4-mini \
  --seed 43 \
  --output-dir data/raw/generated/weak_patch
```

Strong patch:

```bash
python3 -m src.main_generate \
  --mode openai \
  --n 0 \
  --dataset-version official_v1 \
  --generation-kind strong_patch \
  --patch-target-n 15 \
  --patch-max-attempts-per-target 20 \
  --model gpt-5.4-mini \
  --seed 44 \
  --output-dir data/raw/generated/strong_patch
```

High-impact patch:

```bash
python3 -m src.main_generate \
  --mode openai \
  --n 0 \
  --dataset-version official_v1 \
  --generation-kind high_impact_patch \
  --patch-target-n 25 \
  --patch-max-attempts-per-target 80 \
  --model gpt-5.4-mini \
  --seed 46 \
  --output-dir data/raw/generated/high_impact_patch
```

Merge official phases:

```bash
python3 -m src.merge_datasets \
  --broad data/raw/generated/broad/full_synthetic_clean_accepted.jsonl \
  --weak data/raw/generated/weak_patch/full_synthetic_clean_accepted.jsonl \
  --strong data/raw/generated/strong_patch/full_synthetic_clean_accepted.jsonl \
  --high-impact data/raw/generated/high_impact_patch/full_synthetic_clean_accepted.jsonl \
  --output-dir data/processed \
  --raw-output-dir data/raw \
  --max-domain-share 0.25 \
  --max-profile-share 0.25 \
  --max-question-type-share 0.30 \
  --split-group-key scenario_family \
  --train-ratio 0.70 \
  --dev-ratio 0.10 \
  --test-ratio 0.10 \
  --ood-ratio 0.10 \
  --ood-domains "general coursework,non-software team activity,simple school assignment,presentation project,planning-only project,design mockup without implementation,volunteer event organization"
```

Generate quality report:

```bash
python3 -m src.quality_report \
  --input data/raw/full_synthetic_all.jsonl \
  --output-md data/reports/data_quality.md \
  --output-json data/reports/data_quality.json
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Full orchestrated build:

```bash
python3 scripts/build_official_dataset.py --mode mock --output-root /tmp/official_dataset_mock --force --run-tests
```

For the real OpenAI build, use:

```bash
python3 scripts/build_official_dataset.py \
  --mode openai \
  --model gpt-5.4-mini \
  --output-root data \
  --force
```

The build script automatically applies `OPENAI_TIMEOUT_SECONDS=180` and `OPENAI_MAX_RETRIES=2` to its subprocesses unless those variables are already set in your environment.
