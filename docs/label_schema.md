# Label Schema

## Task Summary

The project predicts aspect-level ordinal scores for free-text interview answers from junior software developer candidates. Each answer describes a personal, academic, portfolio, or team project.

Input:

- interview question metadata
- candidate answer text
- target role context

Output:

- six ordinal scores from `1` to `5`
- derived `weak_aspects`
- derived `strong_aspects`
- labeler, validator, and validation metadata

The current official dataset version is `official_v1`.

## Target Role

For `official_v1`, `target_role` is fixed to:

- `Junior Software Developer`

Keeping one role reduces task drift and makes the baseline experiments easier to interpret.

## Aspects

The dataset uses exactly six scoring aspects:

1. `technical_depth`
2. `personal_contribution`
3. `clarity`
4. `problem_solving`
5. `impact`
6. `role_relevance`

Definitions:

- `technical_depth`: junior-appropriate technical reasoning, implementation detail, debugging, tradeoffs, constraints, or meaningful tool use.
- `personal_contribution`: how clearly the candidate explains what they personally did.
- `clarity`: organization, coherence, concreteness, and interview-readiness.
- `problem_solving`: challenge, reasoning, troubleshooting, decisions, iteration, or solution quality.
- `impact`: project outcome, user/team value, delivery, metrics, improvement, or meaningful learning.
- `role_relevance`: fit to the interview question and to junior software-development work.

## Score Meaning

| Score | Meaning |
| --- | --- |
| 1 | absent or almost absent |
| 2 | weak |
| 3 | partial / acceptable |
| 4 | good |
| 5 | strong |

Scores are ordinal labels, not interval measurements. Adjacent labels can be subjective, especially for `impact` and `role_relevance`.

## Official v1 Exported Record Schema

Official `official_v1` training and evaluation records use these required top-level fields:

| Field | Meaning |
| --- | --- |
| `example_id` | Stable unique example identifier. |
| `dataset_version` | Dataset version, currently `official_v1`. |
| `target_role` | Role context, currently `Junior Software Developer`. |
| `question_id` | Stable question identifier. |
| `question` | Interview question shown to the candidate. |
| `question_type` | Question category used for analysis and split design. |
| `project_domain` | Project/domain category, such as `backend API` or `React web app`. |
| `scenario_family` | Grouping key used to prevent train/test scenario leakage. |
| `profile` | Controlled generation profile and constraints used before answer generation. |
| `answer` | Generated free-text interview answer. |
| `labeler` | Rubric-based scoring pass with scores, evidence, rationale, and confidence. |
| `validator` | Independent validation/scoring pass with evidence, rationale, confidence, and audit flags. |
| `final_scores` | Canonical exported target labels for training and evaluation. |
| `weak_aspects` | Derived list of aspects with `final_scores[aspect] <= 2`. |
| `strong_aspects` | Derived list of aspects with `final_scores[aspect] >= 4`. |
| `validation` | Final validation status, rejection reasons, score deltas, leakage flags, and related checks. |
| `split` | Split assignment, such as `train`, `dev_review_candidates`, `test_review_candidates`, or `ood_test_review_candidates`. |
| `metadata` | Generation backend, model, seed, timestamps, and generation source metadata. |

Reviewed evaluation records additionally include:

- `manual_review`

The `manual_review` field appears in:

- `synthetic_interview_data/data/reviewed/dev_project_team_reviewed.jsonl`
- `synthetic_interview_data/data/reviewed/test_project_team_reviewed.jsonl`
- `synthetic_interview_data/data/reviewed/ood_project_team_reviewed.jsonl`

Some reviewed records also retain backward-compatible helper fields such as `human_reviewed`, `human_final_scores`, and `human_notes`. These are not the canonical training/evaluation targets. Use `final_scores` and `manual_review` for `official_v1` evaluation logic.

## Derived Label Rules

`final_scores` are the exported target labels for `official_v1`.

`weak_aspects` must be derived from `final_scores`:

```text
weak_aspects = [aspect for aspect, score in final_scores.items() if score <= 2]
```

`strong_aspects` must be derived from `final_scores`:

```text
strong_aspects = [aspect for aspect, score in final_scores.items() if score >= 4]
```

These derived fields should be recomputed after any approved manual correction to `final_scores`.

## Labeler and Validator Fields

`labeler` stores the first rubric scoring pass. It contains:

- `scores`: aspect scores proposed by the labeler
- `evidence`: answer-text evidence supporting each aspect score
- `rationale`: short explanation for each score
- `confidence`: confidence values for each aspect

`validator` stores an independent scoring and audit pass. It contains:

- `scores`: validator aspect scores
- `evidence`: validator evidence per aspect
- `rationale`: validator explanations and audit notes
- `confidence`: validator confidence values

`validation` stores export-level checks, including:

- `final_status`
- `agreement_tolerance`
- `rejection_reasons`
- `label_leakage_terms`
- `evidence_contradictions`
- `actual_word_count`
- `score_deltas`
- `flags`

Accepted synthetic training records have `validation.final_status = "accepted"`.

## Manual Review Fields for Reviewed Evaluation Splits

The reviewed dev/test/OOD files are project-team reviewed evaluation sets created from the original synthetic review-candidate splits. They are rubric-based manual review outputs, not external expert annotations.

`manual_review` contains:

- `status`: review outcome, such as `project_team_approved` or `project_team_approved_with_corrections`
- `reviewer_role`: reviewer role metadata
- `review_type`: review method, currently `rubric_based_manual_review`
- `review_scope`: review target, currently `final_evaluation_candidate`
- `label_action`: whether labels were confirmed or corrected
- `corrected_fields`: score fields changed during review
- `exclude_from_final_evaluation`: whether the record was excluded from final reviewed evaluation files
- `review_notes`: concise review rationale

The reviewed evaluation package has:

- reviewed dev: `64`
- reviewed test: `106`
- reviewed OOD: `55`
- final reviewed evaluation size: `225`

One near-duplicate test candidate was excluded from final evaluation. Four approved records had unsupported `impact` labels corrected.

## Notes and Limitations

- `official_v1` is synthetic accepted training data plus project-team reviewed evaluation data.
- Reviewed labels are project-level rubric review labels, not external expert-certified labels.
- `final_scores` are the canonical exported labels for `official_v1` training and evaluation.
- The original dev/test/OOD files under `data/processed/` are preserved as synthetic review candidates for traceability.
- Final evaluation should use the reviewed files under `data/reviewed/`.
- Exact `1` to `5` labels remain subjective even with evidence and review.
- Reports should include both exact-score metrics and low/mid/high band metrics.
- `target_scores`, `candidate_level`, `answer_profile_id`, `generation_metadata`, and `evaluation_subset` may appear in older planning or generation-internal artifacts. They are not canonical exported training/evaluation fields for `official_v1`.
