# Project-Team Manual Review Summary

## Purpose

Create reviewed evaluation files from the synthetic dev/test/OOD review-candidate splits using the project rubric. This is a project-team reviewed evaluation set, not external expert annotation.

## Source Files Reviewed

- dev: `data/processed/dev_review_candidates.jsonl` (64 records)
- test: `data/processed/test_review_candidates.jsonl` (107 records)
- ood: `data/processed/ood_test_review_candidates.jsonl` (55 records)

## Rubric Used

- Score scale: 1 = absent or almost absent, 2 = weak, 3 = partial / acceptable, 4 = good, 5 = strong.
- Aspects: `technical_depth`, `personal_contribution`, `clarity`, `problem_solving`, `impact`, `role_relevance`.
- Derived labels were recomputed so `weak_aspects` equals scores `<= 2` and `strong_aspects` equals scores `>= 4`.

## Review Counts

- Total records reviewed: 226
- Approved unchanged: 221
- Approved with corrections: 4
- Excluded: 1

## Records Reviewed Per Split

- dev: 64
- test: 107
- ood: 55

## Correction Counts By Label

- impact: 4

## Exclusion Counts By Reason

- duplicate_or_near_duplicate: 1

## Final Reviewed Split Sizes

- dev: 64
- test: 106
- ood: 55

## Validation Checks After Review

- JSONL files are valid one-object-per-line files.
- Reviewed records preserve the original required fields and include `manual_review` metadata.
- Scores are integers from 1 to 5 for all six aspects.
- `weak_aspects` and `strong_aspects` were recomputed from reviewed `final_scores`.
- Excluded examples appear only in `manual_review_audit.jsonl`, not in reviewed split files.
- No duplicate `example_id` appears inside or across reviewed split files.
- No duplicate normalized answers were found inside the reviewed split files.
- Existing train/test and OOD/train leakage checks remain empty for reviewed evaluation splits.

## Remaining Limitations

- The review is a structured project-team rubric review, not external expert annotation.
- Adjacent ordinal labels can remain subjective, especially for impact and role relevance.
- The reviewed examples are synthetic and may retain generation-style bias.

## Recommended Usage

Use `dev_project_team_reviewed.jsonl`, `test_project_team_reviewed.jsonl`, and `ood_project_team_reviewed.jsonl` for final evaluation. Use the original candidate files only as pre-review sources. The reviewed files should be treated as project-level reviewed evaluation sets, not as external expert annotations.
