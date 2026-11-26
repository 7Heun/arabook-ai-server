#!/usr/bin/env python3
"""카탈로그 메타데이터 점검·정규화 (기본 dry-run, 원본은 건드리지 않는다).

왜 필요한가
-----------
실제 `chroma_db`를 점검했을 때 다음 결함이 있었다(`scripts/catalog_stats.py`).
- 대분류(`mainCategory`)에 값이 2개 들어간 책 3권: `"소설, 역사"`, `"자기계발, 사회"`
  → 대분류는 하나여야 하고, 중복 라벨은 카탈로그 통계와 필터를 흐린다.
- 쉼표 뒤 빈 항목, 중복 서브카테고리, 앞뒤 공백 같은 표기 문제.

무엇을 하는가
-------------
1. 각 책의 `mainCategory` / `subCategory` / `hashtags`를 정규화한 **결과를 미리 계산**해
   무엇이 어떻게 바뀌는지 보여준다(기본 dry-run).
2. `--apply`를 주면 그 결과를 **사본**에 써 넣는다.

안전 원칙
---------
- 로컬의 `chroma_db/` 원본은 절대 수정하지 않는다. 읽을 때도 임시 폴더로 복사해서 연다
  (Chroma는 폴더를 열면 마이그레이션/락 파일을 쓸 수 있다).
- `--apply`는 `--out-db`(사본 경로)를 **필수**로 받고, 원본과 같은 경로면 거부한다.
  적용 후 원본을 바꾸려면 사람이 직접 확인하고 교체한다.

사용:
    python scripts/clean_catalog.py --db chroma_db                     # 점검 리포트
    python scripts/clean_catalog.py --db chroma_db --out report.md
    python scripts/clean_catalog.py --db chroma_db --apply --out-db chroma_db_clean
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

MAIN_CATEGORY = "mainCategory"
SUB_CATEGORY = "subCategory"
HASHTAGS = "hashtags"


class CatalogCleanError(RuntimeError):
    """점검/적용을 진행할 수 없을 때."""


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def split_values(value: Any) -> list[str]:
    """쉼표 구분 값을 정규화한다: 공백 제거, 빈 항목 제거, 중복 제거(순서 유지)."""
    if value is None:
        return []
    raw_items: Iterable[str]
    if isinstance(value, (list, tuple, set, frozenset)):
        raw_items = (str(item) for item in value)
    else:
        raw_items = str(value).split(",")
    result: list[str] = []
    for item in raw_items:
        text = _normalize_text(item)
        if text and text not in result:
            result.append(text)
    return result


def normalize_main_category(value: Any) -> str:
    """대분류는 하나만 남긴다(첫 값). `"소설, 역사"` → `"소설"`."""
    values = split_values(value)
    return values[0] if values else ""


# ------------------------------------------------------------------ 데이터 로딩
@dataclass(frozen=True)
class CatalogEntry:
    id: str
    metadata: dict[str, Any]
    document: str = ""

    @property
    def isbn(self) -> str:
        return str(self.metadata.get("isbn") or self.id)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or "")


def load_entries(
    chroma_path: str | Path,
    collection_name: str = "books",
    copy_before_open: bool = True,
) -> list[CatalogEntry]:
    """Chroma 컬렉션을 읽는다. 기본값은 임시 사본을 열어 원본에 아무것도 쓰지 않는다."""
    chroma_path = Path(chroma_path)
    if not chroma_path.exists():
        raise CatalogCleanError(f"Chroma 폴더가 없습니다: {chroma_path}")
    if not (chroma_path / "chroma.sqlite3").exists():
        raise CatalogCleanError(f"Chroma 폴더에 chroma.sqlite3가 없습니다: {chroma_path}")

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    # chroma 0.5.x + 최신 posthog 조합에서 텔레메트리 로그가 stderr를 어지럽힌다.
    # (전송 자체가 실패하는 경고라 CLI 출력만 깨끗하게 만든다)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    temp_dir: str | None = None
    target = chroma_path
    if copy_before_open:
        temp_dir = tempfile.mkdtemp(prefix="arabook-clean-")
        target = Path(temp_dir) / "chroma_db"
        shutil.copytree(chroma_path, target)

    client = chromadb.PersistentClient(
        path=str(target), settings=ChromaSettings(anonymized_telemetry=False)
    )
    try:
        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:  # chroma는 컬렉션이 없으면 예외를 던진다
            raise CatalogCleanError(
                f"컬렉션 '{collection_name}'을(를) 찾을 수 없습니다: {exc}"
            ) from exc
        payload = collection.get(include=["metadatas", "documents"])
    finally:
        del client
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    ids: Sequence[str] = list(payload.get("ids") or [])
    metadatas = list(payload.get("metadatas") or [])
    documents = list(payload.get("documents") or [])
    entries = [
        CatalogEntry(
            id=str(raw_id),
            metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
            document=str(documents[index] or "") if index < len(documents) else "",
        )
        for index, raw_id in enumerate(ids)
    ]
    if not entries:
        raise CatalogCleanError(f"컬렉션 '{collection_name}'이(가) 비어 있습니다: {chroma_path}")
    return entries


# ------------------------------------------------------------------ 점검/계획
@dataclass(frozen=True)
class ItemFix:
    id: str
    isbn: str
    title: str
    before: dict[str, Any]
    after: dict[str, Any]
    changes: tuple[str, ...]

    def as_row(self) -> dict[str, Any]:
        return {
            "isbn": self.isbn,
            "title": self.title,
            "changes": list(self.changes),
            "before": self.before,
            "after": self.after,
        }


@dataclass
class CleanPlan:
    source: str
    n_items: int
    fixes: list[ItemFix] = field(default_factory=list)
    #: 수정하지 않지만 알아야 하는 누락/중복
    issues: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_changed(self) -> int:
        return len(self.fixes)

    def counts(self) -> dict[str, int]:
        main_changed = 0
        list_changed = 0
        for fix in self.fixes:
            if any(change.startswith(f"{MAIN_CATEGORY}:") for change in fix.changes):
                main_changed += 1
            if any(
                change.startswith((f"{SUB_CATEGORY}:", f"{HASHTAGS}:"))
                for change in fix.changes
            ):
                list_changed += 1
        return {"multi_valued_main_category": main_changed, "normalized_list": list_changed}

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_items": self.n_items,
            "n_changed": self.n_changed,
            "counts": self.counts(),
            "issues": self.issues,
            "fixes": [fix.as_row() for fix in self.fixes],
            "applied": False,
        }


def plan_cleanup(entries: Sequence[CatalogEntry], source: str = "") -> CleanPlan:
    plan = CleanPlan(source=source, n_items=len(entries))
    issues: dict[str, list[str]] = {
        "empty_main_category": [],
        "empty_sub_category": [],
        "empty_hashtags": [],
        "empty_description": [],
        "duplicate_isbn": [],
    }

    seen_isbn: dict[str, int] = {}
    for entry in entries:
        seen_isbn[entry.isbn] = seen_isbn.get(entry.isbn, 0) + 1

        main_before = entry.metadata.get(MAIN_CATEGORY)
        main_after = normalize_main_category(main_before)
        subs_before = entry.metadata.get(SUB_CATEGORY)
        subs_after = split_values(subs_before)
        tags_before = entry.metadata.get(HASHTAGS)
        tags_after = split_values(tags_before)

        changes: list[str] = []
        if main_after != _normalize_text(main_before):
            changes.append(f"{MAIN_CATEGORY}: {main_before!r} → {main_after!r}")
        if ", ".join(subs_after) != _normalize_text(subs_before) or (
            subs_before is not None and not isinstance(subs_before, str)
        ):
            changes.append(f"{SUB_CATEGORY}: {subs_before!r} → {', '.join(subs_after)!r}")
        if ", ".join(tags_after) != _normalize_text(tags_before) or (
            tags_before is not None and not isinstance(tags_before, str)
        ):
            changes.append(f"{HASHTAGS}: {tags_before!r} → {', '.join(tags_after)!r}")

        if not main_after:
            issues["empty_main_category"].append(entry.isbn)
        if not subs_after:
            issues["empty_sub_category"].append(entry.isbn)
        if not tags_after:
            issues["empty_hashtags"].append(entry.isbn)
        if not _normalize_text(entry.document):
            issues["empty_description"].append(entry.isbn)

        if changes:
            after = dict(entry.metadata)
            after[MAIN_CATEGORY] = main_after
            after[SUB_CATEGORY] = ", ".join(subs_after)
            after[HASHTAGS] = ", ".join(tags_after)
            plan.fixes.append(
                ItemFix(
                    id=entry.id,
                    isbn=entry.isbn,
                    title=entry.title,
                    before={
                        MAIN_CATEGORY: main_before,
                        SUB_CATEGORY: subs_before,
                        HASHTAGS: tags_before,
                    },
                    after={
                        MAIN_CATEGORY: main_after,
                        SUB_CATEGORY: ", ".join(subs_after),
                        HASHTAGS: ", ".join(tags_after),
                    },
                    changes=tuple(changes),
                )
            )

    issues["duplicate_isbn"] = [
        isbn for isbn, count in seen_isbn.items() if count > 1
    ]
    plan.issues = {key: value for key, value in issues.items() if value}
    return plan


# ---------------------------------------------------------------------- 적용
def copy_database(source: str | Path, target: str | Path, force: bool = False) -> Path:
    """원본 chroma 폴더를 수정 대상 사본으로 복사한다. 원본 경로면 거부한다."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source == target:
        raise CatalogCleanError(
            "원본과 같은 경로에는 적용할 수 없습니다. --out-db에 다른 경로를 주세요."
        )
    if not (source / "chroma.sqlite3").exists():
        raise CatalogCleanError(f"원본 Chroma 폴더가 아닙니다: {source}")
    if target.exists():
        if not force:
            raise CatalogCleanError(
                f"대상 경로가 이미 있습니다: {target} (덮어쓰려면 --force)"
            )
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def apply_plan(
    plan: CleanPlan,
    target: str | Path,
    collection_name: str = "books",
    collection: Any | None = None,
) -> int:
    """정규화된 메타데이터를 사본(`target`)에 쓴다. 쓴 건수를 돌려준다."""
    if not plan.fixes:
        return 0

    if collection is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=str(target), settings=ChromaSettings(anonymized_telemetry=False)
        )
        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:
            raise CatalogCleanError(
                f"사본에서 컬렉션 '{collection_name}'을(를) 찾을 수 없습니다: {exc}"
            ) from exc

    collection.update(
        ids=[fix.id for fix in plan.fixes],
        metadatas=[fix.after for fix in plan.fixes],
    )
    return len(plan.fixes)


# ---------------------------------------------------------------------- 리포트
def format_markdown(plan: CleanPlan, applied_to: str | None = None) -> str:
    counts = plan.counts()
    lines = [
        "# 카탈로그 메타데이터 정제 리포트",
        "",
        f"- 출처: `{plan.source}` (읽기 전용으로 열었습니다)",
        f"- 점검 도서: {plan.n_items}권",
        f"- 수정 대상: **{plan.n_changed}권**",
        f"- 대분류 값이 2개 이상인 책: {counts['multi_valued_main_category']}권",
        f"- 목록 표기(공백·빈 항목·중복)를 고칠 책: {counts['normalized_list']}권",
    ]
    if applied_to:
        lines.append(f"- 적용한 사본: `{applied_to}`")
    else:
        lines.append("- 모드: **dry-run** (아무것도 쓰지 않았습니다)")
    lines.append("")

    lines.append("## 수정 내역")
    lines.append("")
    if plan.fixes:
        lines.append("| ISBN | 제목 | 필드 | 이전 | 이후 |")
        lines.append("|---|---|---|---|---|")
        for fix in plan.fixes:
            for change in fix.changes:
                field_name, _, rest = change.partition(": ")
                before, _, after = rest.partition(" → ")
                lines.append(
                    f"| {fix.isbn} | {fix.title[:24]} | {field_name} | {before} | {after} |"
                )
    else:
        lines.append("없음 (이미 정규화되어 있습니다)")
    lines.append("")

    lines.append("## 누락·중복 점검 (수정하지 않음)")
    lines.append("")
    if plan.issues:
        lines.append("| 항목 | 권수 | ISBN |")
        lines.append("|---|---|---|")
        labels = {
            "empty_main_category": "대분류 없음",
            "empty_sub_category": "서브카테고리 없음",
            "empty_hashtags": "해시태그 없음",
            "empty_description": "설명 없음",
            "duplicate_isbn": "ISBN 중복",
        }
        for key, isbns in plan.issues.items():
            lines.append(
                f"| {labels.get(key, key)} | {len(isbns)} | {', '.join(isbns[:10])}"
                f"{' …' if len(isbns) > 10 else ''} |"
            )
    else:
        lines.append("없음")
    lines.append("")
    lines.append("## 정규화 규칙")
    lines.append("")
    lines.append("- `mainCategory`: 쉼표로 여러 값이 오면 **첫 값만** 남긴다(대분류는 하나).")
    lines.append("- 목록 필드(`subCategory`, `hashtags`): 공백 정리, 빈 항목 제거, 중복 제거(순서 유지).")
    lines.append("- 원본 파일은 수정하지 않는다. 반영하려면 `--apply --out-db <사본 경로>`로 쓴 뒤 사람이 교체한다.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="카탈로그 메타데이터 점검·정규화 (기본 dry-run)")
    parser.add_argument("--db", default="chroma_db", help="원본 Chroma 폴더 (읽기 전용)")
    parser.add_argument("--collection", default="books")
    parser.add_argument("--apply", action="store_true", help="정규화 결과를 --out-db 사본에 쓴다")
    parser.add_argument("--out-db", default=None, help="적용 대상 사본 경로 (--apply 시 필수)")
    parser.add_argument("--force", action="store_true", help="--out-db 경로가 있으면 지우고 다시 만든다")
    parser.add_argument("--out", default=None, help="리포트 저장 경로 (.md)")
    parser.add_argument("--json", action="store_true", help="리포트를 JSON으로 출력")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    entries = load_entries(args.db, collection_name=args.collection)
    plan = plan_cleanup(entries, source=str(args.db))

    applied_to: str | None = None
    if args.apply:
        if not args.out_db:
            raise CatalogCleanError(
                "--apply에는 --out-db가 필요합니다. 원본은 수정하지 않습니다."
            )
        applied_to = str(copy_database(args.db, args.out_db, force=args.force))
        written = apply_plan(plan, applied_to, collection_name=args.collection)
        print(f"사본에 적용: {applied_to} ({written}권)")

    if args.json:
        payload = plan.to_json()
        payload["applied_to"] = applied_to
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = format_markdown(plan, applied_to=applied_to)
    print(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"리포트 저장: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except CatalogCleanError as exc:
        print(f"카탈로그 정제를 진행하지 못했습니다: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
