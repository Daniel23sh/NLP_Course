from __future__ import annotations

from collections import Counter
from typing import Any

from src.experiment_data import extract_texts_and_labels
from src.schemas import ASPECTS


class MajorityScoreBaseline:
    def __init__(self) -> None:
        self.majority_scores: dict[str, int] = {}

    def fit(self, train_records: list[dict[str, Any]]) -> "MajorityScoreBaseline":
        if not train_records:
            raise ValueError("train_records must not be empty")
        _, labels = extract_texts_and_labels(train_records)
        for aspect in ASPECTS:
            counts = Counter(labels[aspect])
            self.majority_scores[aspect] = max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
        return self

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, int]]:
        if set(self.majority_scores) != set(ASPECTS):
            raise ValueError("MajorityScoreBaseline must be fit before predict")
        return [dict(self.majority_scores) for _ in records]


class TfidfLogisticRegressionBaseline:
    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int = 5000,
        random_state: int = 42,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = None
        self.classifiers: dict[str, Any] = {}

    def _load_sklearn(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scikit-learn is required for the TF-IDF baseline. Install with: python3 -m pip install -r requirements.txt"
            ) from exc
        return TfidfVectorizer, LogisticRegression

    def fit(self, train_records: list[dict[str, Any]]) -> "TfidfLogisticRegressionBaseline":
        if not train_records:
            raise ValueError("train_records must not be empty")
        TfidfVectorizer, LogisticRegression = self._load_sklearn()
        texts, labels = extract_texts_and_labels(train_records)
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
        )
        features = self.vectorizer.fit_transform(texts)
        self.classifiers = {}
        for aspect in ASPECTS:
            observed_classes = sorted(set(labels[aspect]))
            if len(observed_classes) < 2:
                raise ValueError(f"{aspect}: TF-IDF baseline requires at least two observed train classes")
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
            classifier.fit(features, labels[aspect])
            self.classifiers[aspect] = classifier
        return self

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, int]]:
        if self.vectorizer is None or set(self.classifiers) != set(ASPECTS):
            raise ValueError("TfidfLogisticRegressionBaseline must be fit before predict")
        texts = [str(record["answer"]) for record in records]
        features = self.vectorizer.transform(texts)
        predictions = [{aspect: 0 for aspect in ASPECTS} for _ in records]
        for aspect in ASPECTS:
            aspect_predictions = self.classifiers[aspect].predict(features)
            for index, value in enumerate(aspect_predictions):
                predictions[index][aspect] = int(value)
        return predictions
