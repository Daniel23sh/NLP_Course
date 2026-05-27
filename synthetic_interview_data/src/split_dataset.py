from __future__ import annotations

import math
import random

from src.schemas import (
    STATUS_ACCEPTED,
    STATUS_ACCEPTED_BORDERLINE,
    STATUS_AUDIT_ONLY,
    STATUS_MANUAL_REVIEW,
    DatasetRecord,
)


def split_records(records: list[DatasetRecord], seed: int, manual_review_size: int, ood_domains: set[str]) -> dict[str, list[DatasetRecord]]:
    ood = [record for record in records if record.project_domain in ood_domains]
    in_domain = [record for record in records if record.project_domain not in ood_domains]
    groups: dict[str, list[DatasetRecord]] = {}
    for record in in_domain:
        family = f"{record.question_type}|{record.project_domain}"
        groups.setdefault(family, []).append(record)
    rng = random.Random(seed)
    families = list(groups.values())
    rng.shuffle(families)
    train: list[DatasetRecord] = []
    dev: list[DatasetRecord] = []
    test: list[DatasetRecord] = []
    for family in families:
        if len(dev) < manual_review_size:
            dev.extend(family)
        elif len(test) < manual_review_size:
            test.extend(family)
        else:
            train.extend(family)
    if not train and (dev or test):
        train = (test or dev)[manual_review_size // 2 :]
        if test:
            test = test[: manual_review_size // 2]
        else:
            dev = dev[: manual_review_size // 2]
    splits = {
        "train": train,
        "dev_review_candidates": dev[:manual_review_size],
        "test_review_candidates": test[:manual_review_size],
        "ood_test_review_candidates": ood[:manual_review_size],
    }
    for split, items in splits.items():
        for record in items:
            record.split = split
    return splits


def apply_domain_cap(
    records: list[DatasetRecord],
    domain_cap: float,
    cap_base_size: int | None = None,
) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    if not records:
        return [], []
    base_size = cap_base_size or len(records)
    max_per_domain = max(1, int(base_size * domain_cap))
    kept: list[DatasetRecord] = []
    audit: list[DatasetRecord] = []
    counts: dict[str, int] = {}
    for record in records:
        count = counts.get(record.project_domain, 0)
        if count < max_per_domain:
            kept.append(record)
            counts[record.project_domain] = count + 1
        else:
            record.validation.final_status = STATUS_AUDIT_ONLY
            if "domain_cap_excess" not in record.validation.rejection_reasons:
                record.validation.rejection_reasons.append("domain_cap_excess")
            record.validation.flags.append("domain_cap_excess")
            audit.append(record)
    return kept, audit


def _mark_audit(record: DatasetRecord, reason: str) -> None:
    record.validation.final_status = STATUS_AUDIT_ONLY
    if reason not in record.validation.rejection_reasons:
        record.validation.rejection_reasons.append(reason)
    if reason not in record.validation.flags:
        record.validation.flags.append(reason)


def apply_v4_diversity_caps(
    records: list[DatasetRecord],
    target_size_min: int,
    domain_cap: float,
    question_type_cap: float,
    profile_cap: float,
) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    limits = {
        "project_domain": max(1, math.floor(target_size_min * domain_cap)),
        "question_type": max(1, math.floor(target_size_min * question_type_cap)),
        "profile_id": max(1, math.floor(target_size_min * profile_cap)),
    }
    counts = {"project_domain": {}, "question_type": {}, "profile_id": {}}
    kept: list[DatasetRecord] = []
    audit: list[DatasetRecord] = []
    for record in records:
        values = {
            "project_domain": record.project_domain,
            "question_type": record.question_type,
            "profile_id": record.profile.profile_id,
        }
        exceeded = [
            key
            for key, value in values.items()
            if counts[key].get(value, 0) >= limits[key]
        ]
        if exceeded:
            for key in exceeded:
                _mark_audit(record, f"diversity_cap_excess:{key}")
            audit.append(record)
            continue
        kept.append(record)
        for key, value in values.items():
            counts[key][value] = counts[key].get(value, 0) + 1
    return kept, audit


def split_v3_records(
    records: list[DatasetRecord],
    seed: int,
    manual_review_size: int,
    ood_domains: set[str],
    ood_minimum: int = 20,
    target_size_min: int | None = None,
) -> dict[str, list[DatasetRecord]]:
    reviewable = [
        record
        for record in records
        if record.validation.final_status in {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE, STATUS_MANUAL_REVIEW}
    ]
    train_eligible = [
        record
        for record in records
        if record.validation.final_status in {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE}
    ]
    ood = [record for record in reviewable if record.project_domain in ood_domains]
    in_domain = [record for record in reviewable if record.project_domain not in ood_domains]
    rng = random.Random(seed)
    rng.shuffle(in_domain)
    rng.shuffle(ood)
    dev = in_domain[:manual_review_size]
    test = in_domain[manual_review_size : manual_review_size * 2]
    train: list[DatasetRecord] = []
    ood_count = min(len(ood), max(ood_minimum, manual_review_size))
    ood_review = ood[:ood_count]
    held_out = {record.example_id for record in dev + test + ood_review}
    train.extend([record for record in train_eligible if record.example_id not in held_out])
    if not train:
        for candidate_list in (test, dev, ood_review):
            for index in range(len(candidate_list) - 1, -1, -1):
                candidate = candidate_list[index]
                if candidate.validation.final_status in {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE}:
                    train.append(candidate_list.pop(index))
                    break
            if train:
                break
    held_out = {record.example_id for record in dev + test + ood_review + train}
    if target_size_min is not None:
        extra_candidates = [record for record in in_domain if record.example_id not in held_out]
        while extra_candidates and len(train) + len(dev) + len(test) + len(ood_review) < target_size_min:
            target_split = dev if len(dev) <= len(test) else test
            target_split.append(extra_candidates.pop(0))
    splits = {
        "train": train,
        "dev_review_candidates": dev,
        "test_review_candidates": test,
        "ood_test_review_candidates": ood_review,
    }
    for split_name, split_records_ in splits.items():
        for record in split_records_:
            record.split = split_name
    return splits


def _scenario_family(record: DatasetRecord) -> str:
    return record.scenario_family or "|".join(
        [
            record.question_type,
            record.project_domain,
            record.profile.profile_id,
        ]
    )


def _group_records(records: list[DatasetRecord]) -> list[list[DatasetRecord]]:
    grouped: dict[str, list[DatasetRecord]] = {}
    for record in records:
        grouped.setdefault(_scenario_family(record), []).append(record)
    return list(grouped.values())


def _assign_split(records: list[DatasetRecord], split_name: str) -> list[DatasetRecord]:
    for record in records:
        record.split = split_name
    return records


def _leakage_report(splits: dict[str, list[DatasetRecord]]) -> dict[str, list[str]]:
    family_splits: dict[str, set[str]] = {}
    for split_name, records in splits.items():
        for record in records:
            family_splits.setdefault(_scenario_family(record), set()).add(split_name)
    train_test = [
        family
        for family, family_split_names in family_splits.items()
        if "train" in family_split_names and "test_review_candidates" in family_split_names
    ]
    ood_train = [
        family
        for family, family_split_names in family_splits.items()
        if "train" in family_split_names and "ood_test_review_candidates" in family_split_names
    ]
    return {
        "train_test_leakage": sorted(train_test),
        "ood_train_leakage": sorted(ood_train),
    }


def split_v4_records(
    records: list[DatasetRecord],
    seed: int,
    target_size_min: int,
    target_size_max: int,
    manual_review_size: int,
    ood_domains: set[str],
    domain_cap: float,
    question_type_cap: float,
    profile_cap: float,
    ood_minimum: int = 20,
) -> tuple[dict[str, list[DatasetRecord]], dict[str, list[str]]]:
    reviewable_statuses = {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE, STATUS_MANUAL_REVIEW}
    train_statuses = {STATUS_ACCEPTED, STATUS_ACCEPTED_BORDERLINE}
    candidates = [record for record in records if record.validation.final_status in reviewable_statuses]
    kept, _audit = apply_v4_diversity_caps(
        candidates,
        target_size_min=target_size_min,
        domain_cap=domain_cap,
        question_type_cap=question_type_cap,
        profile_cap=profile_cap,
    )
    rng = random.Random(seed)
    ood_groups = _group_records([record for record in kept if record.project_domain in ood_domains])
    in_domain_groups = _group_records([record for record in kept if record.project_domain not in ood_domains])
    rng.shuffle(ood_groups)
    rng.shuffle(in_domain_groups)

    ood_review: list[DatasetRecord] = []
    unused_ood_groups: list[list[DatasetRecord]] = []
    for group in ood_groups:
        if len(ood_review) >= min(ood_minimum, target_size_max):
            break
        ood_review.extend(group)
    used_ood_families = {_scenario_family(record) for record in ood_review}
    unused_ood_groups.extend(
        group
        for group in ood_groups
        if group and _scenario_family(group[0]) not in used_ood_families
    )
    ood_review = ood_review[:target_size_max]

    remaining_capacity = max(target_size_max - len(ood_review), 0)
    dev_target = min(manual_review_size, remaining_capacity)
    test_target = min(manual_review_size, max(remaining_capacity - dev_target, 0))
    train_target = max(target_size_min - len(ood_review) - dev_target - test_target, 0)

    train: list[DatasetRecord] = []
    dev: list[DatasetRecord] = []
    test: list[DatasetRecord] = []
    for group in in_domain_groups:
        train_eligible = [record for record in group if record.validation.final_status in train_statuses]
        if len(dev) < dev_target:
            dev.extend(group)
        elif len(test) < test_target:
            test.extend(group)
        elif len(train) < train_target and train_eligible:
            train.extend(train_eligible)
        elif len(dev) <= len(test):
            dev.extend(group)
        else:
            test.extend(group)
        if len(train) + len(dev) + len(test) + len(ood_review) >= target_size_max:
            break

    while unused_ood_groups and len(train) + len(dev) + len(test) + len(ood_review) < target_size_min:
        group = unused_ood_groups.pop(0)
        if len(train) + len(dev) + len(test) + len(ood_review) + len(group) > target_size_max:
            break
        ood_review.extend(group)

    splits = {
        "train": _assign_split(train[:target_size_max], "train"),
        "dev_review_candidates": _assign_split(dev[:target_size_max], "dev_review_candidates"),
        "test_review_candidates": _assign_split(test[:target_size_max], "test_review_candidates"),
        "ood_test_review_candidates": _assign_split(ood_review[:target_size_max], "ood_test_review_candidates"),
    }
    return splits, _leakage_report(splits)
