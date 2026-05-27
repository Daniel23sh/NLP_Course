from __future__ import annotations

import re


FIRST_PERSON_ACTIONS = [
    "i implemented",
    "i wrote",
    "i tested",
    "i debugged",
    "i fixed",
    "i built",
    "i created",
    "i added",
    "i reviewed",
    "i suggested",
    "i handled",
    "i reproduced",
    "i checked",
    "i changed",
    "i compared",
    "i personally owned",
    "i documented",
    "i was responsible for",
    "my role involved",
    "my main responsibility",
]

TECH_TERMS = [
    "api",
    "backend",
    "frontend",
    "database",
    "query",
    "index",
    "logs",
    "inputs",
    "code",
    "endpoint",
    "payload",
    "test",
    "implementation",
    "data",
    "python",
    "react",
    "tensorflow",
    "opencv",
    "cache",
    "normaliz",
    "edge-case",
    "edge case",
]

IMPLEMENTATION_TERMS = [
    "implemented",
    "wrote",
    "changed the code",
    "checked logs",
    "reproduced",
    "compared",
    "documented",
    "debugged",
    "tested",
    "fixed",
    "optimized",
    "reviewed",
    "merged",
]

PROBLEM_TERMS = ["issue", "bug", "problem", "challenge", "slow", "error", "failed", "did not work", "constraint"]
SOLUTION_TERMS = ["debug", "logs", "tested", "fixed", "solution", "checked", "reproduced", "confirmed", "adjusted", "changed"]
IMPACT_TERMS = [
    "improved",
    "reduced",
    "fewer errors",
    "finish the milestone",
    "deliver the milestone",
    "delivered",
    "easier to use",
    "faster",
    "accuracy",
    "stable",
    "stability",
    "user",
    "users",
    "load",
    "measured",
    "30 percent",
    "saved time",
    "manual work",
    "final demo",
    "used by",
    "qa checklist",
    "response time",
    "error rate",
    "from",
    "to",
]
PROJECT_TERMS = ["project", "app", "system", "dashboard", "api", "model", "platform", "feature"]

SOFTWARE_DEVELOPMENT_EVIDENCE_TERMS = [
    "coding",
    "coded",
    "code",
    "implementation",
    "implemented",
    "debugging",
    "debugged",
    "testing",
    "tested",
    "api",
    "database",
    "frontend",
    "front end",
    "backend",
    "back end",
    "model training",
    "deployment",
    "deployed",
    "technical decision",
    "technical design",
    "architecture",
    "performance optimization",
    "optimized",
]

CONCRETE_PROJECT_OUTCOME_TERMS = [
    "metric improvement",
    "improved accuracy",
    "improved performance",
    "reduced latency",
    "reduced errors",
    "reduced load time",
    "reduced response time",
    "saved time",
    "reduced manual work",
    "manual work",
    "deployed feature",
    "delivered feature",
    "completed deliverable",
    "client demo",
    "used in the demo",
    "final demo",
    "used by the team",
    "used by users",
    "user adoption",
    "used by",
    "users saw",
    "faster for users",
    "fewer errors",
    "qa checklist",
    "passed qa",
    "response time",
    "error rate",
    "accuracy increased",
    "automated",
    "reduced from",
    "increased from",
    "production",
]

TECHNICAL_PROBLEM_SOLVING_EVIDENCE_TERMS = [
    "identified the root cause",
    "root cause",
    "checked logs",
    "inspected logs",
    "wrote tests",
    "tested the fix",
    "reproduced the bug",
    "reproduced the issue",
    "debugged code",
    "debugged the code",
    "changed the implementation",
    "compared alternatives",
    "verified the fix",
    "fixed the code",
    "traced the error",
]

NEGATION_CUES = ["no ", "not ", "without ", "did not ", "was not ", "were not ", "is not ", "isn't ", "wasn't "]


def count_any(text: str, patterns: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for pattern in patterns if pattern in lowered)


def contains_any(text: str, patterns: list[str]) -> bool:
    return count_any(text, patterns) > 0


def contains_positive_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    for pattern in patterns:
        start = lowered.find(pattern)
        while start != -1:
            window = lowered[max(0, start - 32) : start]
            if not any(cue in window for cue in NEGATION_CUES):
                return True
            start = lowered.find(pattern, start + len(pattern))
    return False


def has_metric_outcome(text: str) -> bool:
    lowered = text.lower()
    if any(cue in lowered for cue in ["i learned", "helped me understand", "personal growth"]) and not any(
        term in lowered for term in ["from", "to", "reduced", "increased", "saved", "used by"]
    ):
        return False
    number = r"\d+(?:\.\d+)?"
    unit = r"(?:%|percent|seconds?|minutes?|ms|milliseconds?|errors?|items?|checks?)?"
    before_after_patterns = [
        rf"\bfrom\s+(?:about\s+|around\s+)?{number}\s*{unit}\s+to\s+(?:about\s+|around\s+)?{number}\s*{unit}\b",
        rf"\b{number}\s*{unit}\s+to\s+{number}\s*{unit}\b",
        rf"\b(?:reduced|increased|improved|went)\s+[^.?!]{{0,80}}\bfrom\s+{number}\s*{unit}\s+to\s+{number}\s*{unit}\b",
    ]
    for pattern in before_after_patterns:
        for match in re.finditer(pattern, lowered):
            window = lowered[max(0, match.start() - 32) : match.start()]
            if not any(cue in window for cue in NEGATION_CUES):
                return True
    return False
