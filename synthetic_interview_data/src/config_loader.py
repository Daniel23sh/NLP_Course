from __future__ import annotations

from pathlib import Path

from src.io_utils import read_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_generation_config(config_dir: Path | None = None) -> dict:
    return read_json((config_dir or CONFIG_DIR) / "generation_config.yaml")


def load_profiles(config_dir: Path | None = None) -> dict:
    return read_json((config_dir or CONFIG_DIR) / "profiles.yaml")


def load_rubrics(config_dir: Path | None = None) -> dict:
    return read_json((config_dir or CONFIG_DIR) / "rubrics.yaml")

