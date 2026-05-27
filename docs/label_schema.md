# Label Schema

## Task Summary

The project predicts aspect-level ordinal scores for free-text interview answers given by candidates for a `Junior Software Developer` role.

### Input

- `target_role`
- `question`
- `answer`

### Output

- six ordinal scores from `1` to `5`
- derived `weak_aspects` list where every aspect with score `<= 2` is marked as weak
- `evaluation_subset` for subset-aware evaluation slices such as `main` and `weak_answer_challenge`

## Target Role

For the current project scope, `target_role` is fixed to:

- `Junior Software Developer`

Later versions may test role transfer, but the main dataset should stay focused on one role to avoid mixing task difficulty with role variation.

## Aspects

The dataset uses exactly six aspects:

1. `technical_depth`
2. `personal_contribution`
3. `clarity`
4. `problem_solving`
5. `impact`
6. `role_relevance`

Earlier drafts used `relevance`, `specificity`, `structure`, `technical_depth`, `ownership`, and `impact`.
The active canonical schema keeps the same course requirement of six interpretable ordinal aspects, but uses names that better match the final project framing.

## Aspect Definitions

1. `technical_depth`: whether the answer demonstrates junior-appropriate technical reasoning, implementation detail, debugging, tradeoffs, or constraints.
2. `personal_contribution`: whether it is clear what the candidate personally did, without requiring senior-level ownership.
3. `clarity`: whether the answer is understandable, concrete enough, and organized well enough to follow.
4. `problem_solving`: whether the answer describes a reasoning, troubleshooting, decision, or iteration process.
5. `impact`: whether the answer explains an outcome, value, improvement, or learning.
6. `role_relevance`: whether the answer addresses the interview question and fits a Junior Software Developer role.

## Score Meaning

| Score | Meaning |
|---|---|
| 1 | absent or almost absent |
| 2 | weak |
| 3 | partial / acceptable |
| 4 | good |
| 5 | strong |

## Weak-Aspect Rule

An aspect is considered weak if:

```text
score <= 2
```

## Controlled Attribute Levels

The gold scores are derived from hidden control variables before answer generation. The modular generator freezes these controls first, computes scores deterministically in code, and only then asks the LLM to write the answer text.

| Attribute level | Score |
|---|---:|
| `absent` | 1 |
| `low` | 2 |
| `medium` | 3 |
| `good` | 4 |
| `high` | 5 |

## Required Example Fields

Each dataset record should contain:

- `example_id`
- `dataset_version`
- `target_role`
- `question_id`
- `question`
- `question_type`
- `project_domain`
- `candidate_level`
- `answer_profile_id`
- `same_question_group_size`
- `attributes`
- `target_scores`
- `answer`
- `weak_aspects`
- `evaluation_subset`
- `generation_metadata`
- `validation`
- `split`

## Notes

- `target_scores` are canonical and should be used for training/evaluation.
- `weak_aspects` is a derived field and must always be recomputed from `target_scores` before export.
- `strong_aspects` is a derived field and must always be recomputed from `target_scores` where score `>= 4`.
- `evaluation_subset` is required in the normal v3 generator/export flow and the strict export path should fail closed if it is missing.
- Older records may be normalized to `main`, but only through an explicitly separate legacy-normalization helper.
- `weak_answer_challenge` is reserved for intentionally narrow stress-test examples rather than the default evaluation population.
- `question_type` and `project_domain` should be preserved for split design, EDA, and error analysis.
- The main exported dataset should include only records with `validation.final_status = "accepted"`.
- Records that require regeneration or manual follow-up should be written to a separate audit JSONL with full validation evidence.
- The current active planning artifacts are the `v3` schema, generation config, and question bank.
