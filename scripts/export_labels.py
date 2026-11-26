#!/usr/bin/env python3
"""MySQL 행동 로그 → 오프라인 평가 라벨(`eval/labels.json`) 추출.

무엇을 만드나
-------------
`eval/labels.schema.json` 형식의 JSON. 사용자별로
- `preferences`: 회원이 고른 서브카테고리 이름 (`member_sub_category_selections`)
- `relevant_isbns`: 양성 행동(대여·구매·좋아요 등)이 있었던 책의 ISBN13
- `graded_relevance`: 행동별 가중치의 최댓값 (2=강한 관심, 1=관심)

를 뽑는다. 이 파일이 있으면 `python scripts/eval_recommend.py --labels eval/labels.json`
으로 Recall·MRR 등을 실제로 측정할 수 있다.

왜 테이블 이름을 인자로 받나
---------------------------
이 저장소가 아는 스키마는 `app/services/database.py`에 쓰인 테이블뿐이고, 대여/구매/
좋아요 테이블의 실제 이름·컬럼은 여기 없다. 그래서 이름을 코드에 박아 넣지 않고
`--action`으로 받는다(식별자는 정규식으로 검증한 뒤에만 SQL에 넣는다).
스키마가 바뀌어도 스크립트를 고치지 않아도 되고, 잘못된 이름으로 조용히 0건을
만드는 일도 없다(실행 즉시 SQL 오류로 드러난다).

사용 예
-------
::

    python scripts/export_labels.py \\
        --since 2026-09-01 --until 2026-12-01 \\
        --action "table=rentals,member=member_id,book=book_id,time=rented_at,weight=2" \\
        --action "table=book_likes,member=member_id,book=book_id,time=created_at,weight=1" \\
        --exclude-recommended \\
        --out eval/labels.json

- `book=`은 `book_id` 컬럼(→ `books` 테이블과 조인해 ISBN을 얻음),
  `isbn=`은 ISBN을 직접 들고 있는 컬럼이다. 둘 중 하나만 쓴다.
- `--exclude-recommended`: 추천으로 **노출된** 책을 정답에서 뺀다. 추천기가 추천한
  책을 추천기가 맞혔다고 세는 순환(노출 편향)을 막기 위한 장치다.
- 기본값은 카탈로그(chroma_db)에 있는 ISBN만 남긴다. 카탈로그에 없는 정답은 어떤
  추천기도 고를 수 없어 Recall 분보를 왜곡한다. `--no-catalog-filter`로 끌 수 있다.

종료 코드: 0 성공, 2 카탈로그/스키마 문제, 3 DB 접속·쿼리 실패.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.errors import AppError  # noqa: E402

DEFAULT_SCHEMA = REPO_ROOT / "eval" / "labels.schema.json"
DEFAULT_OUT = REPO_ROOT / "eval" / "labels.json"

#: SQL에 넣어도 되는 식별자만 허용한다 (테이블/컬럼 이름은 바인딩할 수 없다).
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: 선호 서브카테고리. 운영 추천이 읽는 것과 같은 테이블이다.
PREFERENCE_SQL = """
SELECT ms.member_id AS member_id, sc.sub_category_name AS sub_category_name
FROM member_sub_category_selections ms
JOIN sub_categories sc ON ms.sub_category_id = sc.sub_category_id
"""

#: 추천 노출 이력. 노출 편향 제거에 쓴다.
EXPOSURE_SQL = """
SELECT DISTINCT r.member_id AS member_id, b.isbn AS isbn
FROM daily_book_id_recommendations r
JOIN books b ON b.book_id = r.book_id
WHERE r.recommendation_date >= %s AND r.recommendation_date < %s
"""


class LabelExportError(RuntimeError):
    """라벨을 만들 수 없을 때(설정 오류 포함)."""


# ---------------------------------------------------------------- 행동 정의
@dataclass(frozen=True)
class ActionSource:
    """양성 행동 하나(테이블 + 컬럼 + 가중치)."""

    name: str
    table: str
    member_column: str
    time_column: str
    weight: int = 1
    book_column: str | None = None
    isbn_column: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("table", self.table),
            ("member", self.member_column),
            ("time", self.time_column),
        ):
            if not _IDENTIFIER.match(value or ""):
                raise LabelExportError(f"{label} 이름이 올바르지 않습니다: {value!r}")
        if (self.book_column is None) == (self.isbn_column is None):
            raise LabelExportError(
                f"[{self.name}] book= 또는 isbn= 중 정확히 하나를 지정해야 합니다."
            )
        for value in (self.book_column, self.isbn_column):
            if value is not None and not _IDENTIFIER.match(value):
                raise LabelExportError(f"컬럼 이름이 올바르지 않습니다: {value!r}")
        if self.weight < 1:
            raise LabelExportError(f"[{self.name}] weight는 1 이상이어야 합니다.")

    def sql(self) -> str:
        params = f"a.{self.member_column} AS member_id, a.{self.time_column} AS acted_at"
        where = f"WHERE a.{self.time_column} >= %s AND a.{self.time_column} < %s"
        if self.book_column:
            return (
                f"SELECT {params}, b.isbn AS isbn\n"
                f"FROM {self.table} a\n"
                f"JOIN books b ON b.book_id = a.{self.book_column}\n{where}"
            )
        return f"SELECT {params}, a.{self.isbn_column} AS isbn\nFROM {self.table} a\n{where}"


def parse_action_spec(spec: str) -> ActionSource:
    """`table=..,member=..,book=..,time=..,weight=..` 문자열을 파싱한다."""
    parts: dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise LabelExportError(f"--action 항목이 key=value 형식이 아닙니다: {chunk!r}")
        key, value = chunk.split("=", 1)
        parts[key.strip().lower()] = value.strip()

    allowed = {"table", "member", "book", "isbn", "time", "weight", "name"}
    unknown = set(parts) - allowed
    if unknown:
        raise LabelExportError(f"--action에 알 수 없는 키가 있습니다: {', '.join(sorted(unknown))}")

    for required in ("table", "member", "time"):
        if not parts.get(required):
            raise LabelExportError(f"--action에 {required}= 값이 필요합니다: {spec!r}")

    table = parts["table"]
    name = parts.get("name") or table
    try:
        weight = int(parts.get("weight", "1"))
    except ValueError as exc:
        raise LabelExportError(f"weight는 정수여야 합니다: {parts.get('weight')!r}") from exc

    return ActionSource(
        name=name,
        table=table,
        member_column=parts["member"],
        time_column=parts["time"],
        weight=weight,
        book_column=parts.get("book"),
        isbn_column=parts.get("isbn"),
    )


# ---------------------------------------------------------------- ISBN 정리
def normalize_isbn(value: Any) -> str | None:
    """하이픈/공백을 떼고 ISBN13만 남긴다. ISBN10·빈 값은 버린다."""
    if value is None:
        return None
    digits = re.sub(r"[\s-]", "", str(value))
    if len(digits) != 13 or not digits.isdigit():
        return None
    return digits


# ---------------------------------------------------------------- 라벨 생성
@dataclass
class LabelBuildResult:
    payload: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)


def build_labels(
    actions: Sequence[Mapping[str, Any]],
    preferences: Mapping[Any, Sequence[str]],
    *,
    since: str,
    until: str,
    label_source: str,
    description: str | None = None,
    catalog_isbns: Iterable[str] | None = None,
    excluded: Iterable[tuple[Any, str]] = (),
    min_relevant: int = 1,
) -> LabelBuildResult:
    """행동 행/선호 행 → 라벨 payload.

    `actions`의 각 행은 `member_id`, `isbn`, `weight`, `action`, `acted_at`을 가진다
    (여러 `--action` 결과를 이어 붙인 것).
    """
    catalog = {normalize_isbn(isbn) for isbn in (catalog_isbns or [])}
    catalog.discard(None)
    filtering = catalog_isbns is not None
    excluded_pairs = {(str(member), str(isbn)) for member, isbn in excluded}

    stats: dict[str, Any] = {
        "action_rows": 0,
        "invalid_isbn_rows": 0,
        "rows_outside_catalog": 0,
        "rows_excluded_as_exposed": 0,
        "users_without_preferences": 0,
        "users_without_relevant": 0,
        "duplicate_preference_labels": 0,
        "unknown_preference_labels": {},
        "actions": {},
    }

    # 사용자별 ISBN → 가중치 최댓값
    relevance: dict[str, dict[str, int]] = {}
    for row in actions:
        stats["action_rows"] += 1
        member = str(row.get("member_id"))
        isbn = normalize_isbn(row.get("isbn"))
        action_name = str(row.get("action") or "action")
        stats["actions"][action_name] = stats["actions"].get(action_name, 0) + 1
        if not isbn:
            stats["invalid_isbn_rows"] += 1
            continue
        if (member, isbn) in excluded_pairs:
            stats["rows_excluded_as_exposed"] += 1
            continue
        if filtering and isbn not in catalog:
            stats["rows_outside_catalog"] += 1
            continue
        weight = max(1, int(row.get("weight") or 1))
        current = relevance.setdefault(member, {})
        current[isbn] = max(current.get(isbn, 0), weight)

    known_preferences: set[str] = set()
    users: list[dict[str, Any]] = []
    members = {str(key) for key in preferences} | set(relevance)
    for member in sorted(members, key=_member_sort_key):
        prefs_raw = preferences.get(member) or preferences.get(_int_if_digits(member)) or []
        prefs: list[str] = []
        for value in prefs_raw:
            text = str(value).strip()
            if text and text not in prefs:
                prefs.append(text)
        if not prefs:
            stats["users_without_preferences"] += 1
            continue
        known_preferences.update(prefs)
        graded = relevance.get(member, {})
        if len(graded) < min_relevant:
            stats["users_without_relevant"] += 1
            continue
        ordered = sorted(graded.items(), key=lambda item: (-item[1], item[0]))
        user: dict[str, Any] = {
            "member_id": _int_if_digits(member),
            "preferences": prefs,
            "relevant_isbns": [isbn for isbn, _ in ordered],
        }
        if any(weight > 1 for _, weight in ordered):
            user["graded_relevance"] = {isbn: min(int(weight), 2) for isbn, weight in ordered}
        users.append(user)

    if not users:
        raise LabelExportError(
            "조건에 맞는 사용자가 없습니다. 기간·행동 테이블·카탈로그 필터를 확인하세요."
        )

    payload: dict[str, Any] = {
        "meta": {
            "format_version": 1,
            "label_source": label_source,
            "description": description
            or "MySQL 양성 행동에서 추출한 라벨. 노출 편향 제거 여부는 extraction을 참고.",
            "catalog": f"{len(catalog)} ISBN" if filtering else "unfiltered",
            "extraction": {
                "since": since,
                "until": until,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "actions": sorted({str(row.get("action")) for row in actions if row.get("action")}),
                "excluded_recommended": bool(excluded_pairs),
                "n_action_rows": stats["action_rows"],
            },
        },
        "users": users,
    }
    stats["n_users"] = len(users)
    stats["n_relevant_total"] = sum(len(u["relevant_isbns"]) for u in users)
    stats["n_relevant_mean"] = stats["n_relevant_total"] / len(users)
    stats["n_distinct_preferences"] = len(known_preferences)
    return LabelBuildResult(payload=payload, stats=stats)


def _int_if_digits(value: str) -> int | str:
    return int(value) if str(value).isdigit() else value


def _member_sort_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


# ---------------------------------------------------------- 스키마 검증 (subset)
_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
}


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    python_type = _TYPE_MAP.get(expected)
    if python_type is None:
        return True  # 지원하지 않는 타입은 통과시킨다(오탐 방지)
    if python_type is bool:
        return isinstance(value, bool)
    return isinstance(value, python_type) and not isinstance(value, bool)


def validate_payload(payload: Any, schema: Mapping[str, Any]) -> list[str]:
    """`eval/labels.schema.json`이 쓰는 JSON Schema 부분집합만 검사한다.

    `jsonschema` 의존성을 추가하지 않으려고 필요한 키워드(const, type, required,
    properties, additionalProperties, items, minItems, enum)만 구현했다.
    """
    errors: list[str] = []

    def walk(value: Any, node: Mapping[str, Any], path: str) -> None:
        if "const" in node and value != node["const"]:
            errors.append(f"{path}: {node['const']!r}여야 합니다 (현재 {value!r})")
        if "enum" in node and value not in node["enum"]:
            errors.append(f"{path}: {node['enum']} 중 하나여야 합니다")

        expected = node.get("type")
        if expected is not None:
            types = expected if isinstance(expected, list) else [expected]
            if not any(_type_matches(value, t) for t in types):
                errors.append(f"{path}: 타입이 {expected}여야 합니다 (현재 {type(value).__name__})")
                return

        if isinstance(value, dict):
            for key in node.get("required", []):
                if key not in value:
                    errors.append(f"{path}: 필수 키 '{key}'가 없습니다")
            properties = node.get("properties", {})
            for key, child in properties.items():
                if key in value:
                    walk(value[key], child, f"{path}.{key}")
            if node.get("additionalProperties") is False:
                extra = sorted(set(value) - set(properties))
                for key in extra:
                    errors.append(f"{path}: 허용되지 않는 키 '{key}'")
        elif isinstance(value, list):
            items = node.get("items")
            if items:
                for index, item in enumerate(value):
                    walk(item, items, f"{path}[{index}]")
            if "minItems" in node and len(value) < node["minItems"]:
                errors.append(f"{path}: 최소 {node['minItems']}개가 필요합니다 (현재 {len(value)}개)")

    walk(payload, schema, "$")
    return errors


def load_schema(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise LabelExportError(f"스키마 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ DB 조회
def fetch_rows(repository: Any, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    with repository.connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())


def load_preferences(repository: Any) -> dict[Any, list[str]]:
    grouped: dict[Any, list[str]] = {}
    for row in fetch_rows(repository, PREFERENCE_SQL):
        member = row["member_id"]
        grouped.setdefault(member, []).append(row["sub_category_name"])
    return grouped


def load_exposed(repository: Any, since: str, until: str) -> list[tuple[Any, str]]:
    pairs: list[tuple[Any, str]] = []
    for row in fetch_rows(repository, EXPOSURE_SQL, (since, until)):
        isbn = normalize_isbn(row["isbn"])
        if isbn:
            pairs.append((row["member_id"], isbn))
    return pairs


def load_actions(
    repository: Any, sources: Sequence[ActionSource], since: str, until: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for row in fetch_rows(repository, source.sql(), (since, until)):
            rows.append({**row, "action": source.name, "weight": source.weight})
    return rows


def load_catalog_isbns(catalog_path: str | None) -> set[str] | None:
    """카탈로그에 있는 ISBN 집합. None이면 필터하지 않는다."""
    if not catalog_path:
        return None
    from eval.catalog import CatalogError, load_catalog_from_chroma, load_catalog_from_json

    path = Path(catalog_path)
    try:
        catalog = (
            load_catalog_from_json(path)
            if path.is_file()
            else load_catalog_from_chroma(path)
        )
    except (CatalogError, OSError) as exc:
        raise LabelExportError(
            f"카탈로그를 읽지 못했습니다: {exc}\n"
            "카탈로그 필터 없이 만들려면 --no-catalog-filter를 붙이세요."
        ) from exc
    return {item.isbn for item in catalog.items}


# ---------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MySQL 행동 로그에서 오프라인 평가 라벨(eval/labels.json)을 만든다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            '  python scripts/export_labels.py --since 2026-09-01 --until 2026-12-01 \\\n'
            '      --action "table=rentals,member=member_id,book=book_id,time=rented_at,weight=2" \\\n'
            '      --exclude-recommended --out eval/labels.json\n'
        ),
    )
    parser.add_argument("--since", required=True, help="행동 수집 시작 (YYYY-MM-DD, 포함)")
    parser.add_argument("--until", required=True, help="행동 수집 끝 (YYYY-MM-DD, 미포함)")
    parser.add_argument(
        "--action",
        action="append",
        default=[],
        metavar="SPEC",
        help="양성 행동 정의 (반복 가능): table=..,member=..,book=|isbn=..,time=..[,weight=][,name=]",
    )
    parser.add_argument(
        "--exclude-recommended",
        action="store_true",
        help="추천으로 노출된 (회원, ISBN)을 정답에서 제외 (노출 편향 완화)",
    )
    parser.add_argument("--catalog", default="chroma_db", help="카탈로그(chroma_db 또는 JSON). 기본 chroma_db")
    parser.add_argument(
        "--no-catalog-filter",
        action="store_true",
        help="카탈로그에 없는 ISBN도 정답으로 남긴다(기본은 제거)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="출력 경로 (기본 eval/labels.json)")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="검증에 쓸 라벨 스키마")
    parser.add_argument("--label-source", default=None, help="meta.label_source 값")
    parser.add_argument("--description", default=None, help="meta.description 값")
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=1,
        help="사용자를 남길 최소 정답 수 (기본 1). 0이면 정답 없는 사용자도 남긴다",
    )
    parser.add_argument("--print", dest="print_only", action="store_true", help="파일을 쓰지 않고 출력만")
    return parser.parse_args(argv)


def _validate_dates(args: argparse.Namespace) -> None:
    try:
        since = date.fromisoformat(args.since)
        until = date.fromisoformat(args.until)
    except ValueError as exc:
        raise LabelExportError(f"날짜는 YYYY-MM-DD 형식이어야 합니다: {exc}") from exc
    if since >= until:
        raise LabelExportError(f"--since({since})가 --until({until})보다 앞서야 합니다.")


def run(args: argparse.Namespace) -> int:
    _validate_dates(args)
    if not args.action:
        raise LabelExportError(
            "--action을 하나 이상 지정해야 합니다. 예: "
            '--action "table=rentals,member=member_id,book=book_id,time=rented_at,weight=2"'
        )
    sources = [parse_action_spec(spec) for spec in args.action]
    catalog_isbns = None if args.no_catalog_filter else load_catalog_isbns(args.catalog)

    from app.core.config import get_settings
    from app.services.database import BookRepository

    repository = BookRepository(get_settings().database)
    try:
        preferences = load_preferences(repository)
        actions = load_actions(repository, sources, args.since, args.until)
        excluded = (
            load_exposed(repository, args.since, args.until)
            if args.exclude_recommended
            else []
        )
    except AppError as exc:
        print(f"DB 조회에 실패했습니다: {exc.message}", file=sys.stderr)
        return 3

    label_source = args.label_source or f"mysql-actions-{args.since}..{args.until}"
    result = build_labels(
        actions,
        preferences,
        since=args.since,
        until=args.until,
        label_source=label_source,
        description=args.description,
        catalog_isbns=catalog_isbns,
        excluded=excluded,
        min_relevant=args.min_relevant,
    )

    schema_errors = validate_payload(result.payload, load_schema(args.schema))
    if schema_errors:
        print("라벨 스키마 검증 실패:", file=sys.stderr)
        for error in schema_errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    text = json.dumps(result.payload, ensure_ascii=False, indent=2)
    if args.print_only:
        print(text)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"라벨 저장: {out}")

    print(_format_summary(result.stats, label_source))
    print("스키마 검증: 통과")
    return 0


def _format_summary(stats: Mapping[str, Any], label_source: str) -> str:
    action_text = ", ".join(f"{k} {v}" for k, v in sorted(stats["actions"].items()))
    lines = [
        "",
        f"- label_source: {label_source}",
        f"- 사용자 {stats['n_users']}명, 정답 {stats['n_relevant_total']}건 "
        f"(사용자당 평균 {stats['n_relevant_mean']:.1f})",
        f"- 행동 행 {stats['action_rows']}건" + (f" ({action_text})" if action_text else ""),
        f"- 카탈로그 밖 ISBN 제거: {stats['rows_outside_catalog']}건 / "
        f"ISBN 형식 오류: {stats['invalid_isbn_rows']}건 / "
        f"추천 노출로 제외: {stats['rows_excluded_as_exposed']}건",
        f"- 선호 없는 사용자 제외: {stats['users_without_preferences']}명 / "
        f"정답 부족으로 제외: {stats['users_without_relevant']}명",
        "",
        "주의: 이 라벨로 측정한 수치는 라벨링 기준(어떤 행동을 정답으로 봤는지)에 따라 달라집니다. "
        "label_source와 extraction을 리포트에 함께 남기세요.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except LabelExportError as exc:
        print(f"라벨을 만들지 못했습니다: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
