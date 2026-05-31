# Official Synthetic Interview Data Quality Report

- Dataset size: 558
- Clean accepted count: 558
- OOD count: 55
- Manual review candidate count: 226

## Status Counts

- accepted: 558

## Split Counts

- dev_review_candidates: 64
- not_selected: 67
- ood_test_review_candidates: 55
- test_review_candidates: 107
- train: 265

## Score Distributions

- technical_depth: {1: 130, 2: 106, 3: 134, 4: 146, 5: 42}
- personal_contribution: {1: 47, 2: 222, 3: 44, 4: 137, 5: 108}
- clarity: {1: 6, 2: 38, 3: 97, 4: 329, 5: 88}
- problem_solving: {1: 99, 2: 180, 3: 93, 4: 131, 5: 55}
- impact: {1: 75, 2: 334, 3: 50, 4: 74, 5: 25}
- role_relevance: {1: 8, 2: 129, 3: 75, 4: 152, 5: 194}

## Low/Mid/High Coverage

- technical_depth: {'low': 236, 'mid': 134, 'high': 188}
- personal_contribution: {'low': 269, 'mid': 44, 'high': 245}
- clarity: {'low': 44, 'mid': 97, 'high': 417}
- problem_solving: {'low': 279, 'mid': 93, 'high': 186}
- impact: {'low': 409, 'mid': 50, 'high': 99}
- role_relevance: {'low': 137, 'mid': 75, 'high': 346}
- Impact high coverage: 99 / 20

## Remaining Coverage Gaps

- technical_depth: []
- personal_contribution: []
- clarity: []
- problem_solving: []
- impact: []
- role_relevance: []

## Diversity

- Domains: {'database system': 39, 'React web app': 96, 'analytics reporting tool': 49, 'backend API': 31, 'automation script': 40, 'general coursework': 21, 'mobile app': 24, 'full-stack student platform': 25, 'data analysis dashboard': 7, 'planning-only project': 37, 'presentation project': 6, 'design mockup without implementation': 6, 'volunteer event organization': 6, 'non-software team activity': 25, 'simple school assignment': 23, 'browser extension': 33, 'API integration': 21, 'testing framework': 4, 'machine learning project': 8, 'NLP course project': 9, 'DevOps/deployment task': 9, 'CLI tool': 8, 'data pipeline': 8, 'embedded Arduino project': 1, 'data cleaning script': 1, 'cloud deployment task': 1, 'game development project': 1, 'accessibility improvement': 3, 'Linux CLI tool': 1, 'authentication feature': 1, 'open-source bug fix': 2, 'poster design activity': 1, 'club scheduling activity': 1, 'business plan without implementation': 1, 'survey assignment': 1, 'study group coordination': 1, 'simple calculator app': 1, 'spreadsheet macro': 1, 'cybersecurity class project': 2, 'frontend performance project': 1, 'web scraping data pipeline': 1, 'cloud infrastructure project': 1}
- Profiles: {'strong_project_answer': 53, 'generic_confident_answer': 28, 'vague_but_relevant': 56, 'structured_but_shallow': 48, 'high_contribution_low_impact': 48, 'short_underdeveloped_answer': 19, 'off_topic_answer': 17, 'team_focused_low_personal_contribution': 2, 'project_manager_observer_role': 7, 'presentation_only_project': 7, 'learning_reflection_without_project': 4, 'design_only_no_implementation': 6, 'irrelevant_volunteer_project': 6, 'non_software_team_activity': 6, 'generic_school_assignment_no_coding': 6, 'observer_no_personal_task': 42, 'severely_unclear_but_realistic_answer': 38, 'no_outcome_answer': 17, 'unfinished_project_no_result': 18, 'vague_tools_no_explanation': 35, 'official_strong_technical_depth_46': 1, 'official_strong_technical_depth_47': 1, 'official_strong_technical_depth_49': 1, 'official_strong_technical_depth_70': 1, 'official_strong_technical_depth_76': 1, 'official_strong_technical_depth_82': 1, 'official_strong_technical_depth_84': 1, 'official_strong_personal_contribution_90': 1, 'official_strong_personal_contribution_97': 1, 'official_strong_personal_contribution_103': 1, 'official_strong_personal_contribution_104': 1, 'official_strong_personal_contribution_111': 1, 'official_strong_personal_contribution_112': 1, 'official_strong_personal_contribution_130': 1, 'official_strong_personal_contribution_134': 1, 'official_strong_clarity_157': 1, 'official_strong_clarity_163': 1, 'official_strong_clarity_164': 1, 'official_strong_clarity_166': 1, 'official_strong_clarity_172': 1, 'official_strong_problem_solving_188': 1, 'official_strong_problem_solving_189': 1, 'official_strong_problem_solving_190': 1, 'official_strong_problem_solving_193': 1, 'official_strong_problem_solving_214': 1, 'official_strong_problem_solving_218': 1, 'official_strong_problem_solving_221': 1, 'official_strong_impact_226': 1, 'official_strong_impact_229': 1, 'official_strong_impact_243': 1, 'official_strong_impact_244': 1, 'official_strong_impact_256': 1, 'official_strong_impact_257': 1, 'official_strong_impact_267': 1, 'official_strong_role_relevance_270': 1, 'official_strong_role_relevance_284': 1, 'official_strong_role_relevance_285': 1, 'official_strong_role_relevance_295': 1, 'official_strong_role_relevance_303': 1, 'official_strong_role_relevance_304': 1, 'before_after_metric_improvement': 9, 'measurable_performance_improvement': 6, 'reduced_manual_work': 6, 'successful_demo_or_client_value': 5, 'deployed_feature_with_usage': 3, 'accuracy_or_quality_improvement': 3, 'qa_or_error_reduction_fix': 3, 'unclear_but_relevant_backend': 1, 'unclear_but_relevant_deployment': 1, 'unclear_game_project': 1, 'unclear_accessibility_project': 1, 'unclear_cli_project': 1, 'mid_score_basic_project': 1, 'mid_score_shared_feature': 1, 'mid_score_clear_low_impact': 1, 'mid_score_high_contrib_shallow': 1, 'mid_score_borderline_relevance': 1, 'mid_score_debugging': 1, 'mid_score_unmeasured_impact': 1, 'mid_score_shared_security': 1, 'mid_score_tradeoff': 1, 'mid_score_accessibility': 1, 'high_impact_security': 1, 'high_impact_data_pipeline': 1, 'high_impact_open_source': 1, 'high_impact_cloud': 1, 'high_impact_accessibility': 1}
- Question types: {'performance_improvement': 41, 'project_overview': 101, 'feature_delivery_under_constraint': 50, 'technical_challenge': 35, 'leadership_or_ownership': 39, 'general_reflection': 21, 'debugging_story': 26, 'tradeoff_explanation': 26, 'design_decision': 5, 'planning_activity': 38, 'presentation_activity': 7, 'design_mockup': 6, 'volunteer_activity': 6, 'team_activity': 27, 'school_assignment': 24, 'extension_project': 33, 'integration_story': 21, 'testing_story': 5, 'teamwork_conflict': 7, 'failure_or_learning': 9, 'deployment_story': 11, 'automation_ownership': 10, 'data_pipeline_story': 10}

## Split Leakage Report

- train_test_leakage: []
- ood_train_leakage: []

## Labeler-Validator Agreement

- delta 0: 2893
- delta 1: 455

## Readiness Statement

- Usable for synthetic training: yes
- Usable for final evaluation without manual labels: no
- Manual review required for dev/test/OOD: yes
- Ready to start baseline training: yes

Note: This report describes the synthetic accepted dataset and original review-candidate splits. For final evaluation, use the project-team reviewed files under `data/reviewed/` and the separate `manual_review_quality` report.

## Recommended Next Actions

- Use the project-team reviewed dev/test/OOD files under `data/reviewed/` for final evaluation.
- Inspect any remaining low/mid/high coverage gaps before model training.
- Keep profile_mismatch and rejected examples for error analysis only.
