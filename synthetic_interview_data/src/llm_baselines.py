from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.llm_client import LLMClient
from src.llm_prompting import PROMPT_VERSION, build_few_shot_prompt, build_zero_shot_prompt, select_few_shot_examples
from src.schemas import ASPECTS, SCORE_MAX, SCORE_MIN, compute_strong_aspects, compute_weak_aspects


VALID_MODES = {"zero-shot", "few-shot"}


def derive_weak_strong(final_scores: dict[str, int]) -> tuple[list[str], list[str]]:
    return compute_weak_aspects(final_scores), compute_strong_aspects(final_scores)


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    raise ValueError("no JSON object found")


def _failed(raw_text: str, error: str) -> dict[str, Any]:
    return {
        "parse_status": "failed",
        "error": error,
        "raw_response": raw_text,
    }


def parse_llm_prediction(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(_extract_json_object(raw_text))
    except Exception as exc:
        return _failed(raw_text, f"invalid_json: {exc}")
    if not isinstance(payload, dict):
        return _failed(raw_text, "top-level JSON value must be an object")
    final_scores = payload.get("final_scores")
    if not isinstance(final_scores, dict):
        return _failed(raw_text, "missing final_scores")
    missing = [aspect for aspect in ASPECTS if aspect not in final_scores]
    if missing:
        return _failed(raw_text, f"missing scores: {', '.join(missing)}")
    extra = [aspect for aspect in final_scores if aspect not in ASPECTS]
    if extra:
        return _failed(raw_text, f"unknown scores: {', '.join(extra)}")
    parsed_scores: dict[str, int] = {}
    for aspect in ASPECTS:
        value = final_scores[aspect]
        if type(value) is not int:
            return _failed(raw_text, f"{aspect} score must be an integer")
        if not SCORE_MIN <= value <= SCORE_MAX:
            return _failed(raw_text, f"{aspect} score must be from {SCORE_MIN} to {SCORE_MAX}")
        parsed_scores[aspect] = value
    weak_aspects, strong_aspects = derive_weak_strong(parsed_scores)
    rationale = payload.get("rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}
    return {
        "parse_status": "ok",
        "final_scores": parsed_scores,
        "weak_aspects": weak_aspects,
        "strong_aspects": strong_aspects,
        "rationale": {aspect: str(rationale.get(aspect, "")) for aspect in ASPECTS},
        "raw_response": raw_text,
    }


@dataclass
class DryRunLLMPredictor:
    model_name: str = "dry-run-mock"

    def generate(self, prompt: str, record: dict[str, Any], mode: str) -> str:
        scores = {aspect: int(record["final_scores"][aspect]) for aspect in ASPECTS}
        return json.dumps(
            {
                "final_scores": scores,
                "weak_aspects": [],
                "strong_aspects": [],
                "rationale": {aspect: f"Dry-run {mode} prediction; not a real LLM result." for aspect in ASPECTS},
            },
            ensure_ascii=True,
        )


class OpenAILLMPredictor:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client = LLMClient(mode="openai", model=model_name)

    def generate(self, prompt: str, record: dict[str, Any], mode: str) -> str:
        return self.client.generate(prompt)


class LLMBaseline:
    def __init__(
        self,
        mode: str,
        predictor,
        train_records: list[dict[str, Any]] | None = None,
        few_shot_k: int = 3,
        seed: int = 42,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        self.mode = mode
        self.predictor = predictor
        self.train_records = train_records or []
        self.few_shot_k = few_shot_k
        self.seed = seed
        self.few_shot_examples = (
            select_few_shot_examples(self.train_records, k=few_shot_k, seed=seed)
            if mode == "few-shot"
            else []
        )

    def _prompt(self, record: dict[str, Any]) -> str:
        if self.mode == "zero-shot":
            return build_zero_shot_prompt(record)
        if not self.few_shot_examples:
            raise ValueError("few-shot mode requires at least one training example")
        return build_few_shot_prompt(record, self.few_shot_examples)

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        predictions = []
        for index, record in enumerate(records):
            prompt = self._prompt(record)
            try:
                raw_response = self.predictor.generate(prompt, record=record, mode=self.mode)
                parsed = parse_llm_prediction(raw_response)
            except Exception as exc:
                parsed = _failed("", f"prediction_error: {exc}")
            parsed.update(
                {
                    "example_id": record.get("example_id", f"row_{index}"),
                    "mode": self.mode,
                    "model": getattr(self.predictor, "model_name", "unknown"),
                    "prompt_version": PROMPT_VERSION,
                    "prediction_index": index,
                }
            )
            predictions.append(parsed)
        return predictions
