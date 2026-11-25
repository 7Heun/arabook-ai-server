"""오프라인 추천 평가 하네스.

라벨셋(JSON)과 카탈로그를 읽어 변형별 지표를 계산한다. 실제 수치는 라벨과
임베딩이 있어야만 나온다. 없는 상태에서는 `evaluate()`를 호출하지 않고
"미산출"로 남긴다 — 수치를 만들어내지 않기 위한 구조적 장치다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from eval.catalog import Catalog
from eval.embedders import Embedder
from eval.metrics import DEFAULT_KS, UserResult, aggregate, coverage, evaluate_user
from eval.recommenders import (
    QueryContext,
    Recommender,
    VectorIndex,
    build_recommenders,
)

#: 라벨 파일에서 "실제 측정용이 아님"을 표시하는 값들
SYNTHETIC_SOURCES = {"example", "synthetic", "synthetic-format-example", "dummy", "sample"}


class LabelError(ValueError):
    """라벨 파일 형식이 잘못되었을 때."""


@dataclass(frozen=True)
class EvalUser:
    member_id: str
    preferences: tuple[str, ...]
    relevant: tuple[str, ...]
    graded: dict[str, float] = field(default_factory=dict)

    @property
    def query_text(self) -> str:
        # 운영 코드와 동일: 선호 서브카테고리 이름을 공백으로 이어 붙인다
        return " ".join(self.preferences)


@dataclass
class LabelSet:
    path: str
    meta: dict[str, Any]
    users: list[EvalUser]

    @property
    def label_source(self) -> str:
        return str(self.meta.get("label_source", "unknown"))

    @property
    def is_synthetic(self) -> bool:
        return self.label_source.lower() in SYNTHETIC_SOURCES


@dataclass
class ValidationReport:
    n_users: int
    n_users_without_preferences: list[str]
    n_users_without_relevant: list[str]
    unknown_relevant_isbns: dict[str, list[str]]
    unknown_preference_labels: dict[str, list[str]]
    n_relevant_values: list[int]

    @property
    def has_blocking_issues(self) -> bool:
        """측정 자체가 불가능한 문제만 차단으로 본다."""
        return self.n_users == 0 or len(self.n_users_without_preferences) == self.n_users


@dataclass
class VariantResult:
    key: str
    label: str
    notes: str
    aggregate: dict[str, Any]
    coverage: dict[int, float]
    ranked: dict[str, list[str]]


def load_labels(path: str | Path) -> LabelSet:
    path = Path(path)
    if not path.exists():
        raise LabelError(
            f"라벨 파일이 없습니다: {path}\n"
            "eval/labels.example.json 형식을 참고해 eval/labels.json을 만드세요."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelError(f"라벨 파일이 JSON이 아닙니다: {exc}") from exc

    if not isinstance(payload, dict) or "users" not in payload:
        raise LabelError("라벨 파일에는 최상위 'users' 배열이 있어야 합니다.")

    users: list[EvalUser] = []
    for index, raw in enumerate(payload["users"]):
        if not isinstance(raw, dict):
            raise LabelError(f"users[{index}]가 객체가 아닙니다.")
        if "member_id" not in raw:
            raise LabelError(f"users[{index}]에 member_id가 없습니다.")
        preferences = raw.get("preferences", [])
        relevant = raw.get("relevant_isbns", [])
        if not isinstance(preferences, list) or not isinstance(relevant, list):
            raise LabelError(
                f"users[{index}]의 preferences/relevant_isbns는 배열이어야 합니다."
            )
        graded_raw = raw.get("graded_relevance", {}) or {}
        if not isinstance(graded_raw, dict):
            raise LabelError(f"users[{index}].graded_relevance는 객체여야 합니다.")
        users.append(
            EvalUser(
                member_id=str(raw["member_id"]),
                preferences=tuple(str(p) for p in preferences),
                relevant=tuple(str(i) for i in relevant),
                graded={str(k): float(v) for k, v in graded_raw.items()},
            )
        )

    if not users:
        raise LabelError("라벨 파일에 사용자가 없습니다.")
    return LabelSet(path=str(path), meta=dict(payload.get("meta", {})), users=users)


def validate_labels(labels: LabelSet, catalog: Catalog) -> ValidationReport:
    known_preferences = {sub for item in catalog.items for sub in item.sub_categories}
    by_isbn = {item.isbn: item for item in catalog.items}

    unknown_relevant: dict[str, list[str]] = {}
    unknown_preferences: dict[str, list[str]] = {}
    for user in labels.users:
        missing = [isbn for isbn in user.relevant if isbn not in by_isbn]
        if missing:
            unknown_relevant[user.member_id] = missing
        unknown_pref = [p for p in user.preferences if p not in known_preferences]
        if unknown_pref:
            unknown_preferences[user.member_id] = unknown_pref

    return ValidationReport(
        n_users=len(labels.users),
        n_users_without_preferences=[u.member_id for u in labels.users if not u.preferences],
        n_users_without_relevant=[u.member_id for u in labels.users if not u.relevant],
        unknown_relevant_isbns=unknown_relevant,
        unknown_preference_labels=unknown_preferences,
        n_relevant_values=[len(u.relevant) for u in labels.users],
    )


def evaluate(
    labels: LabelSet,
    catalog: Catalog,
    embedder: Embedder,
    variant_keys: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
    alpha: float = 0.15,
) -> dict[str, VariantResult]:
    """라벨과 카탈로그로 변형별 지표를 계산한다.

    사용자별 선호 텍스트는 운영과 같은 방식(" ".join(preferences))으로 만들어
    임베딩한다. 임베딩은 사용자당 한 번만 호출하고 모든 변형이 공유한다.
    """
    index = VectorIndex(catalog)
    recommenders: list[Recommender] = build_recommenders(index, variant_keys, alpha=alpha)
    limit = max(ks)

    query_embeddings = embedder.embed([user.query_text for user in labels.users])

    contexts = [
        QueryContext(
            member_id=user.member_id,
            preferences=user.preferences,
            query_text=user.query_text,
            query_embedding=query_embeddings[i],
        )
        for i, user in enumerate(labels.users)
    ]

    results: dict[str, VariantResult] = {}
    categories_by_isbn = catalog.categories_by_isbn()
    for recommender in recommenders:
        ranked: dict[str, list[str]] = {}
        user_results: list[UserResult] = []
        for user, context in zip(labels.users, contexts):
            ranking = recommender.rank(context, limit)
            ranked[user.member_id] = ranking
            user_results.append(
                evaluate_user(
                    member_id=user.member_id,
                    ranked=ranking,
                    relevant=user.relevant,
                    categories_by_isbn=categories_by_isbn,
                    ks=ks,
                    graded_relevance=user.graded or None,
                )
            )
        results[recommender.key] = VariantResult(
            key=recommender.key,
            label=recommender.label,
            notes=recommender.notes,
            aggregate=aggregate(user_results, ks=ks),
            coverage={
                k: coverage([r[:k] for r in ranked.values()], catalog.isbns) for k in ks
            },
            ranked=ranked,
        )
    return results


def catalog_gaps(catalog: Catalog, labels: LabelSet) -> dict[str, Any]:
    """라벨의 정답 ISBN이 카탈로그에 없을 때, 평가에서 무엇이 빠지는지 요약."""
    known = {item.isbn for item in catalog.items}
    missing_by_user = {
        user.member_id: [isbn for isbn in user.relevant if isbn not in known]
        for user in labels.users
        if any(isbn not in known for isbn in user.relevant)
    }
    return {
        "n_labeled_relevant": sum(len(u.relevant) for u in labels.users),
        "n_known_relevant": sum(1 for u in labels.users for isbn in u.relevant if isbn in known),
        "users_with_missing": missing_by_user,
    }
