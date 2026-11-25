#!/usr/bin/env python3
"""Chroma 스냅샷에서 카탈로그 구성 통계를 계산한다.

chromadb 설치 없이 `chroma.sqlite3`를 **읽기 전용**으로 열어 메타데이터만
집계한다. 커밋된 `chroma_db/`를 그대로 두고 실행해도 파일이 변하지 않는다.

사용:
    python scripts/catalog_stats.py --db chroma_db
    python scripts/catalog_stats.py --db chroma_db --out eval/catalog_stats.md
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path


def connect_read_only(db_path: Path) -> sqlite3.Connection:
    sqlite_file = db_path / "chroma.sqlite3" if db_path.is_dir() else db_path
    if not sqlite_file.exists():
        raise SystemExit(f"chroma.sqlite3를 찾을 수 없습니다: {sqlite_file}")
    return sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)


def load_items(connection: sqlite3.Connection) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        SELECT e.id, m.key, m.string_value
        FROM embeddings e
        JOIN embedding_metadata m ON m.id = e.id
        """
    ).fetchall()
    grouped: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for embedding_id, key, value in rows:
        if value is not None:
            grouped[embedding_id][key] = value
    return list(grouped.values())


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def statistics(items: list[dict[str, str]]) -> dict:
    main = collections.Counter(item.get("mainCategory", "") for item in items)
    subs: collections.Counter[str] = collections.Counter()
    tags: collections.Counter[str] = collections.Counter()
    for item in items:
        subs.update(_split(item.get("subCategory")))
        tags.update(_split(item.get("hashtags")))
    multi_main = [item.get("isbn", "") for item in items if "," in item.get("mainCategory", "")]
    no_desc = [item.get("isbn", "") for item in items if not (item.get("chroma:document") or "").strip()]
    return {
        "n_items": len(items),
        "main_categories": main.most_common(),
        "n_main_labels": len(main),
        "multi_valued_main": multi_main,
        "n_sub_labels": len(subs),
        "top_sub_categories": subs.most_common(10),
        "n_hashtags": len(tags),
        "top_hashtags": tags.most_common(10),
        "no_description": no_desc,
        "avg_sub_per_book": (
            sum(len(_split(item.get("subCategory"))) for item in items) / len(items) if items else 0.0
        ),
    }


def format_markdown(stats: dict, source: str) -> str:
    lines = ["# 카탈로그 구성 통계", "", f"- 출처: `{source}`", f"- 도서 수: **{stats['n_items']}권**", ""]
    lines.append("| 대분류 라벨 | 권수 |")
    lines.append("|---|---|")
    for label, count in stats["main_categories"]:
        lines.append(f"| {label or '(없음)'} | {count} |")
    lines.append("")
    lines.append(f"- 대분류 라벨 종류: {stats['n_main_labels']}종")
    lines.append(f"- 서브카테고리 라벨 종류: {stats['n_sub_labels']}종")
    lines.append(f"- 해시태그 종류: {stats['n_hashtags']}종")
    lines.append(f"- 책당 평균 서브카테고리 수: {stats['avg_sub_per_book']:.1f}")
    lines.append("")
    lines.append("## 데이터 품질 점검")
    lines.append("")
    lines.append(
        f"- 대분류 필드에 값이 2개 이상 들어간 책: {len(stats['multi_valued_main'])}권 "
        "(대분류는 하나만 가져야 한다는 규칙 위반)"
    )
    if stats["multi_valued_main"]:
        lines.append("  - ISBN: " + ", ".join(stats["multi_valued_main"]))
    lines.append(f"- 설명이 비어 있는 책: {len(stats['no_description'])}권")
    lines.append("")
    lines.append("## 상위 서브카테고리")
    lines.append("")
    lines.append("| 서브카테고리 | 권수 |")
    lines.append("|---|---|")
    for label, count in stats["top_sub_categories"]:
        lines.append(f"| {label} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chroma 카탈로그 통계 (읽기 전용)")
    parser.add_argument("--db", default="chroma_db")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    connection = connect_read_only(Path(args.db))
    try:
        items = load_items(connection)
    finally:
        connection.close()

    if not items:
        print("메타데이터가 없습니다. 컬렉션이 비어 있을 수 있습니다.", file=sys.stderr)
        return 2

    stats = statistics(items)
    if args.json:
        text = json.dumps(stats, ensure_ascii=False, indent=2)
    else:
        text = format_markdown(stats, str(args.db))
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
