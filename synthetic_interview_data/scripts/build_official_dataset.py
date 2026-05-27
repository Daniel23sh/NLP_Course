from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import read_json, read_jsonl  # noqa: E402
from src.schemas import ASPECTS  # noqa: E402


OOD_DOMAINS = (
    "general coursework,"
    "non-software team activity,"
    "simple school assignment,"
    "presentation project,"
    "planning-only project,"
    "design mockup without implementation,"
    "volunteer event organization"
)
OPENAI_SUBPROCESS_DEFAULTS = {
    "OPENAI_TIMEOUT_SECONDS": "180",
    "OPENAI_MAX_RETRIES": "2",
}


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    for key, value in OPENAI_SUBPROCESS_DEFAULTS.items():
        env.setdefault(key, value)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _resolve_output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _raw_dir(output_root: Path) -> Path:
    return output_root / "raw"


def _processed_dir(output_root: Path) -> Path:
    return output_root / "processed"


def _reports_dir(output_root: Path) -> Path:
    return output_root / "reports"


def _reviewed_dir(output_root: Path) -> Path:
    return output_root / "reviewed"


def _phase_output_dir(output_root: Path, phase: str) -> Path:
    return _raw_dir(output_root) / "generated" / phase


def _phase_clean_path(output_root: Path, phase: str) -> Path:
    return _phase_output_dir(output_root, phase) / "full_synthetic_clean_accepted.jsonl"


def _required_validation_files(output_root: Path) -> list[Path]:
    processed_dir = _processed_dir(output_root)
    return [
        processed_dir / "train.jsonl",
        processed_dir / "dev_review_candidates.jsonl",
        processed_dir / "test_review_candidates.jsonl",
        processed_dir / "ood_test_review_candidates.jsonl",
        processed_dir / "not_selected.jsonl",
        _raw_dir(output_root) / "full_synthetic_all.jsonl",
        processed_dir / "full_synthetic_clean_accepted.jsonl",
        _reports_dir(output_root) / "data_quality.json",
    ]


def _ensure_required_structure(output_root: Path) -> None:
    for path in [
        _raw_dir(output_root),
        _raw_dir(output_root) / "generated",
        _processed_dir(output_root),
        _reviewed_dir(output_root),
        _reports_dir(output_root),
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for name in [
        "dev_human_reviewed.jsonl",
        "test_human_reviewed.jsonl",
        "ood_test_human_reviewed.jsonl",
    ]:
        (_reviewed_dir(output_root) / name).touch(exist_ok=True)


def _ensure_safe_to_write(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = "\n".join(str(path) for path in existing)
        raise SystemExit(
            "Refusing to overwrite existing official outputs without --force:\n"
            f"{joined}"
        )
    if force:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _main_generate_base(args: argparse.Namespace, output_root: Path, phase: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.main_generate",
        "--mode",
        args.mode,
        "--dataset-version",
        args.dataset_version,
        "--generation-kind",
        phase,
        "--model",
        args.model,
        "--seed",
        str(seed),
        "--output-dir",
        str(_phase_output_dir(output_root, phase)),
    ]


def _run_generation(args: argparse.Namespace, output_root: Path) -> None:
    broad_cmd = _main_generate_base(args, output_root, "broad", args.seed)
    broad_cmd.extend(["--n", str(args.broad_n)])
    _run(broad_cmd)

    weak_cmd = _main_generate_base(args, output_root, "weak_patch", args.seed + 1)
    weak_cmd.extend(
        [
            "--n",
            "0",
            "--targeted-patch",
            "--patch-target-n",
            str(args.weak_patch_target_n),
            "--patch-max-attempts-per-target",
            str(args.weak_patch_max_attempts),
        ]
    )
    _run(weak_cmd)

    strong_cmd = _main_generate_base(args, output_root, "strong_patch", args.seed + 2)
    strong_cmd.extend(
        [
            "--n",
            "0",
            "--patch-target-n",
            str(args.strong_patch_target_n),
            "--patch-max-attempts-per-target",
            str(args.strong_patch_max_attempts),
        ]
    )
    _run(strong_cmd)

    high_cmd = _main_generate_base(args, output_root, "high_impact_patch", args.seed + 4)
    high_cmd.extend(
        [
            "--n",
            "0",
            "--patch-target-n",
            str(args.high_impact_patch_target_n),
            "--patch-max-attempts-per-target",
            str(args.high_impact_patch_max_attempts),
        ]
    )
    _run(high_cmd)


def _require_phase_inputs(output_root: Path) -> None:
    missing = [
        _phase_clean_path(output_root, phase)
        for phase in ["broad", "weak_patch", "strong_patch", "high_impact_patch"]
        if not _phase_clean_path(output_root, phase).exists()
    ]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Cannot merge because required phase files are missing:\n{joined}")


def _run_merge(output_root: Path) -> None:
    _require_phase_inputs(output_root)
    _run(
        [
            sys.executable,
            "-m",
            "src.merge_datasets",
            "--broad",
            str(_phase_clean_path(output_root, "broad")),
            "--weak",
            str(_phase_clean_path(output_root, "weak_patch")),
            "--strong",
            str(_phase_clean_path(output_root, "strong_patch")),
            "--high-impact",
            str(_phase_clean_path(output_root, "high_impact_patch")),
            "--output-dir",
            str(_processed_dir(output_root)),
            "--raw-output-dir",
            str(_raw_dir(output_root)),
            "--max-domain-share",
            "0.25",
            "--max-profile-share",
            "0.25",
            "--max-question-type-share",
            "0.30",
            "--split-group-key",
            "scenario_family",
            "--train-ratio",
            "0.70",
            "--dev-ratio",
            "0.10",
            "--test-ratio",
            "0.10",
            "--ood-ratio",
            "0.10",
            "--ood-domains",
            OOD_DOMAINS,
        ]
    )


def _run_quality_report(output_root: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "src.quality_report",
            "--input",
            str(_raw_dir(output_root) / "full_synthetic_all.jsonl"),
            "--output-md",
            str(_reports_dir(output_root) / "data_quality.md"),
            "--output-json",
            str(_reports_dir(output_root) / "data_quality.json"),
        ]
    )


def _scenario_family(row: dict) -> str:
    return row.get("scenario_family") or "|".join(
        [
            row.get("question_type", ""),
            row.get("project_domain", ""),
            row.get("profile", {}).get("profile_id", ""),
        ]
    )


def _validate_dataset(output_root: Path) -> dict:
    raw_dir = _raw_dir(output_root)
    processed_dir = _processed_dir(output_root)
    report_json = _reports_dir(output_root) / "data_quality.json"
    required_files = _required_validation_files(output_root)
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("Missing required output files:\n" + "\n".join(str(path) for path in missing))

    rows = read_jsonl(raw_dir / "full_synthetic_all.jsonl")
    train = read_jsonl(processed_dir / "train.jsonl")
    dev = read_jsonl(processed_dir / "dev_review_candidates.jsonl")
    test = read_jsonl(processed_dir / "test_review_candidates.jsonl")
    ood = read_jsonl(processed_dir / "ood_test_review_candidates.jsonl")
    report = read_json(report_json)
    if not rows:
        raise SystemExit("full_synthetic_all.jsonl is empty")
    if not train:
        raise SystemExit("train.jsonl is empty")

    ids = [row["example_id"] for row in rows]
    normalized_answers = [" ".join(row.get("answer", "").lower().split()) for row in rows]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    duplicate_answers = [item for item, count in Counter(normalized_answers).items() if item and count > 1]
    if duplicate_ids:
        raise SystemExit(f"Duplicate IDs found: {duplicate_ids[:5]}")
    if duplicate_answers:
        raise SystemExit(f"Duplicate answers found: {len(duplicate_answers)}")

    for row in rows:
        scores = row.get("final_scores", {})
        if set(scores) != set(ASPECTS):
            raise SystemExit(f"Missing final scores for {row.get('example_id')}")
        for aspect, score in scores.items():
            if not isinstance(score, int) or score < 1 or score > 5:
                raise SystemExit(f"Invalid score for {row.get('example_id')} {aspect}: {score}")
        if not row.get("answer", "").strip():
            raise SystemExit(f"Empty answer for {row.get('example_id')}")
        expected_weak = [aspect for aspect in ASPECTS if scores[aspect] <= 2]
        expected_strong = [aspect for aspect in ASPECTS if scores[aspect] >= 4]
        if row.get("weak_aspects") != expected_weak:
            raise SystemExit(f"weak_aspects mismatch for {row.get('example_id')}")
        if row.get("strong_aspects") != expected_strong:
            raise SystemExit(f"strong_aspects mismatch for {row.get('example_id')}")
        if row.get("validation", {}).get("final_status") == "accepted":
            deltas = row.get("validation", {}).get("score_deltas", {})
            if any(delta >= 2 for delta in deltas.values()):
                raise SystemExit(f"Accepted row has delta >= 2: {row.get('example_id')}")

    family_splits: dict[str, set[str]] = {}
    for row in rows:
        split = row.get("split")
        if split:
            family_splits.setdefault(_scenario_family(row), set()).add(split)
    train_test_leakage = [
        family for family, splits in family_splits.items() if "train" in splits and "test_review_candidates" in splits
    ]
    ood_train_leakage = [
        family for family, splits in family_splits.items() if "train" in splits and "ood_test_review_candidates" in splits
    ]
    if train_test_leakage:
        raise SystemExit(f"Train/test leakage found: {train_test_leakage[:5]}")
    if ood_train_leakage:
        raise SystemExit(f"OOD/train leakage found: {ood_train_leakage[:5]}")

    return {
        "dataset_size": len(rows),
        "split_counts": {
            "train": len(train),
            "dev_review_candidates": len(dev),
            "test_review_candidates": len(test),
            "ood_test_review_candidates": len(ood),
        },
        "status_counts": dict(Counter(row["validation"]["final_status"] for row in rows)),
        "remaining_gaps": report.get("remaining_low_mid_high_gaps", {}),
        "readiness": report.get("readiness", {}),
    }


def _run_tests() -> None:
    _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate the official_v1 synthetic interview dataset.")
    parser.add_argument("--mode", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--dataset-version", default="official_v1")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--run-quality-report", action="store_true", default=True)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--broad-n", type=int, default=220)
    parser.add_argument("--weak-patch-target-n", type=int, default=20)
    parser.add_argument("--weak-patch-max-attempts", type=int, default=30)
    parser.add_argument("--strong-patch-target-n", type=int, default=15)
    parser.add_argument("--strong-patch-max-attempts", type=int, default=20)
    parser.add_argument("--high-impact-patch-target-n", type=int, default=25)
    parser.add_argument("--high-impact-patch-max-attempts", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_version != "official_v1":
        raise SystemExit("This build script is only for dataset-version official_v1.")
    output_root = _resolve_output_root(args.output_root)
    write_targets = [_processed_dir(output_root), _reports_dir(output_root), _raw_dir(output_root) / "full_synthetic_all.jsonl"]
    if not args.skip_generation:
        write_targets.extend(
            [
                _phase_output_dir(output_root, "broad"),
                _phase_output_dir(output_root, "weak_patch"),
                _phase_output_dir(output_root, "strong_patch"),
                _phase_output_dir(output_root, "high_impact_patch"),
            ]
        )
    _ensure_safe_to_write(write_targets, args.force)
    _ensure_required_structure(output_root)

    if not args.skip_generation:
        _run_generation(args, output_root)
    _run_merge(output_root)
    if args.run_quality_report:
        _run_quality_report(output_root)
    validation = _validate_dataset(output_root)
    if args.run_tests:
        _run_tests()
    print(json.dumps({"validation": validation}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
