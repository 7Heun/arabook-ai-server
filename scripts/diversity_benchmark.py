#!/usr/bin/env python3
"""Top-5 추천의 **카테고리 다양성·카탈로그 커버리지** 실측 (정확도 아님).

무엇을 하는가
-------------
저장소에 있는 `chroma_db`의 **실제 임베딩 30권**만 읽어(외부 API 호출 없음),
`baseline_vector`(이전 운영: L2 최근접)와 `production_hybrid`(현재 운영:
코사인 + 카테고리 가중 + MMR)가 Top-5에서 내는

* 고유 서브카테고리 수
* 서브카테고리 분포의 정규화 엔트로피
* 카탈로그 커버리지 (모든 질의의 top-k 합집합 / 30권)

를 비교한다. 두 변형의 순위가 실제로 얼마나 다른지(top-1 일치율, top-k 겹침)도 함께 낸다.

**이 수치는 추천 정확도가 아니다.**
- 정답 라벨(사람이 붙인 "관련 있음")이 없으므로 Recall·Precision·MRR은 계산하지 않는다.
- 여기서 재는 것은 "30권 카탈로그 안에서 추천 목록이 얼마나 퍼지는가"뿐이다.
  다양성이 높다고 좋은 추천이 아니며, 관련도를 희생해 다양성을 얻는 trade-off다.
  어느 쪽이 나은지는 라벨이 있어야 판단할 수 있다(`eval/README.md`).

질의를 어떻게 만드는가 (외부 임베딩 API를 쓰지 않기 위한 대리 질의)
----------------------------------------------------------------
운영 질의는 "선호 서브카테고리 이름"을 임베딩한 짧은 문자열이다. 그 임베딩을
만들려면 OpenAI API를 불러야 하므로, 여기서는 **이미 저장된 책 임베딩만으로**
두 종류의 대리 질의를 만든다.

1. `book_holdout` — 책 1권의 임베딩을 질의로 쓰고, 그 책은 후보에서 뺀다.
   "이 책과 비슷한 취향"을 흉내 낸 질의 30건.
2. `category_centroid` — 서브카테고리 하나에 속한 책들의 임베딩 평균(단위 벡터로 정규화)을
   질의로 쓰고, 선호는 그 서브카테고리로 둔다. "이 카테고리를 고른 사용자"를 흉내 낸 질의.

두 방식 모두 실제 질의 텍스트 임베딩과 같지 않다. 따라서 변형 **사이의 상대 비교**로만
읽어야 하고, 절대 수치를 운영 성능으로 인용하면 안 된다.

원본 보호
---------
`chroma_db`는 항상 임시 폴더로 복사한 사본을 연다(`eval.catalog`). 실행 전후 원본
파일 md5를 찍어 바뀌지 않았음을 리포트에 남긴다.

사용:
    python scripts/diversity_benchmark.py
    python scripts/diversity_benchmark.py --query-mode both --ks 1,3,5
    python scripts/diversity_benchmark.py --out eval/diversity_benchmark.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.catalog import Catalog, CatalogItem, load_catalog_from_chroma  # noqa: E402
from eval.metrics import category_entropy, coverage, distinct_category_count  # noqa: E402
from eval.recommenders import (  # noqa: E402
    BaselineVectorRecommender,
    ProductionHybridRecommender,
    QueryContext,
    VectorIndex,
)

#: 비교하는 두 변형 (이전 운영 vs 현재 운영)
VARIANT_KEYS = ("baseline_vector", "production_hybrid")
VARIANT_LABELS = {
    "baseline_vector": "baseline_vector (이전 운영: L2 최근접)",
    "production_hybrid": "production_hybrid (현재 운영: 코사인 + 카테고리 가중 + MMR)",
}
QUERY_MODES = ("book", "category")


@dataclass(frozen=True)
class QueryCase:
    """대리 질의 1건."""

    name: str
    preferences: tuple[str, ...]
    embedding: tuple[float, ...]
    #: 후보로 쓸 ISBN (None이면 카탈로그 전체)
    allowed_isbns: frozenset[str] | None = None


@dataclass
class VariantOutcome:
    key: str
    label: str
    rankings: dict[str, list[str]]
    distinct: dict[int, float]
    entropy: dict[int, float]
    coverage: dict[int, float]


def file_md5(directory: Path) -> dict[str, str]:
    """폴더 안 모든 파일의 (상대경로 → md5). 원본 불변 확인용."""
    checksums: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            checksums[str(path.relative_to(directory))] = digest
    return checksums


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 0:
        return tuple(float(value) for value in vector)
    return tuple(float(value) / norm for value in vector)


def build_queries(
    catalog: Catalog, mode: str, min_books: int = 2
) -> list[QueryCase]:
    """질의 모드별 대리 질의 목록."""
    if not catalog.has_embeddings:
        raise SystemExit("카탈로그에 임베딩이 없습니다. chroma_db에서 읽어 주세요.")
    queries: list[QueryCase] = []

    if mode == "book":
        for item in catalog.items:
            assert item.embedding is not None
            queries.append(
                QueryCase(
                    name=f"book:{item.isbn}",
                    preferences=item.sub_categories,
                    embedding=tuple(item.embedding),
                    allowed_isbns=frozenset(
                        other.isbn for other in catalog.items if other.isbn != item.isbn
                    ),
                )
            )
    elif mode == "category":
        members: dict[str, list[CatalogItem]] = defaultdict(list)
        for item in catalog.items:
            for sub in item.sub_categories:
                members[sub].append(item)
        for sub, items in sorted(members.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            if len(items) < min_books:
                continue
            vectors = [item.embedding for item in items if item.embedding is not None]
            if not vectors:
                continue
            dimension = len(vectors[0])
            mean = [
                sum(vector[i] for vector in vectors) / len(vectors) for i in range(dimension)
            ]
            queries.append(
                QueryCase(
                    name=f"category:{sub}",
                    preferences=(sub,),
                    embedding=_unit(mean),
                )
            )
    else:
        raise SystemExit(f"알 수 없는 질의 모드: {mode} (가능: {', '.join(QUERY_MODES)})")
    return queries


def _subset(catalog: Catalog, allowed: frozenset[str] | None) -> Catalog:
    if allowed is None:
        return catalog
    items = [item for item in catalog.items if item.isbn in allowed]
    return Catalog(
        items=items,
        source=catalog.source,
        collection=catalog.collection,
        dimension=catalog.dimension,
    )


def run_variant(
    catalog: Catalog,
    queries: Sequence[QueryCase],
    variant: str,
    ks: Sequence[int],
) -> VariantOutcome:
    k_max = max(ks)
    categories = catalog.categories_by_isbn()
    rankings: dict[str, list[str]] = {}
    distinct: dict[int, list[int]] = {k: [] for k in ks}
    entropy: dict[int, list[float]] = {k: [] for k in ks}

    for query in queries:
        pool = _subset(catalog, query.allowed_isbns)
        index = VectorIndex(pool)
        recommender = (
            BaselineVectorRecommender(index)
            if variant == "baseline_vector"
            else ProductionHybridRecommender(index)
        )
        context = QueryContext(
            member_id=query.name,
            preferences=query.preferences,
            query_text=" ".join(query.preferences),
            query_embedding=list(query.embedding),
        )
        ranked = recommender.rank(context, k_max)
        rankings[query.name] = ranked
        for k in ks:
            distinct[k].append(distinct_category_count(ranked[:k], categories))
            entropy[k].append(category_entropy(ranked, categories, k))

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return VariantOutcome(
        key=variant,
        label=VARIANT_LABELS[variant],
        rankings=rankings,
        distinct={k: mean(values) for k, values in distinct.items()},
        entropy={k: mean(values) for k, values in entropy.items()},
        coverage={
            k: coverage([ranking[:k] for ranking in rankings.values()], catalog.isbns)
            for k in ks
        },
    )


def compare_variants(
    queries: Sequence[QueryCase], outcomes: dict[str, VariantOutcome], k: int
) -> dict[str, float]:
    """두 변형의 순위가 실제로 얼마나 다른지."""
    baseline = outcomes["baseline_vector"].rankings
    hybrid = outcomes["production_hybrid"].rankings
    same_top1 = 0
    overlaps: list[float] = []
    for query in queries:
        left = set(baseline[query.name][:k])
        right = set(hybrid[query.name][:k])
        if baseline[query.name][:1] == hybrid[query.name][:1]:
            same_top1 += 1
        union = left | right
        overlaps.append(len(left & right) / len(union) if union else 0.0)
    return {
        "same_top1_share": same_top1 / len(queries) if queries else 0.0,
        "mean_topk_jaccard": sum(overlaps) / len(overlaps) if overlaps else 0.0,
    }


def _summary_row(outcome: VariantOutcome, k: int) -> str:
    return (
        f"| `{outcome.key}` | {outcome.distinct.get(k, 0.0):.2f} | "
        f"{outcome.entropy.get(k, 0.0):.3f} | {outcome.coverage.get(k, 0.0) * 100:.1f}% |"
    )


def _short(title: str, limit: int = 18) -> str:
    """표 안에서 읽히도록 제목을 줄인다(자르면 말줄임표)."""
    return title if len(title) <= limit else title[: limit - 1] + "…"


def _render_examples(
    catalog: Catalog,
    queries: Sequence[QueryCase],
    outcomes: dict[str, VariantOutcome],
    k: int,
) -> list[str]:
    """두 변형의 top-k가 가장 많이 갈린 질의 3건 (근거 확인용)."""
    titles = {item.isbn: item.title for item in catalog.items}
    baseline = outcomes["baseline_vector"].rankings
    hybrid = outcomes["production_hybrid"].rankings
    rows = []
    for query in queries:
        left = baseline[query.name][:k]
        right = hybrid[query.name][:k]
        union = set(left) | set(right)
        rows.append((len(set(left) & set(right)) / len(union) if union else 1.0, query.name, left, right))
    rows.sort(key=lambda row: (row[0], row[1]))

    names = {query.name: query.preferences for query in queries}
    by_isbn = {item.isbn: item for item in catalog.items}

    lines = [f"### 예시 — top-{k}가 가장 갈린 질의 3건", ""]
    lines.append("| 질의 | baseline_vector | production_hybrid |")
    lines.append("|---|---|---|")
    for _, name, left, right in rows[:3]:
        if name.startswith("category:"):
            label = f"카테고리 `{names[name][0]}`"
        else:
            isbn = name.split(":", 1)[1]
            item = by_isbn.get(isbn)
            label = f"책 `{item.title}` ({', '.join(item.sub_categories)})" if item else f"책 `{isbn}`"
        left_text = " · ".join(_short(titles.get(isbn, isbn)) for isbn in left)
        right_text = " · ".join(_short(titles.get(isbn, isbn)) for isbn in right)
        lines.append(f"| {label} | {left_text} | {right_text} |")
    lines.append("")
    return lines


def render_markdown(
    *,
    catalog: Catalog,
    queries_by_mode: dict[str, list[QueryCase]],
    results: dict[str, dict[str, VariantOutcome]],
    ks: Sequence[int],
    command: str,
    md5_before: dict[str, str],
    md5_after: dict[str, str],
    min_books: int,
) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    focus = 5 if 5 in ks else max(ks)
    unchanged = md5_before == md5_after
    lines: list[str] = []

    lines.append("# Top-5 다양성·커버리지 실측 (정확도 아님)")
    lines.append("")
    lines.append(
        "> **이 문서의 수치는 추천 정확도가 아닙니다.** 정답 라벨 없이, 저장된 30권 "
        "카탈로그 안에서 두 순위 규칙이 만든 목록이 얼마나 퍼지는지만 잰 값입니다. "
        "다양성이 높다고 좋은 추천이 아니며, 어느 규칙이 나은지는 라벨이 있어야 판단할 수 있습니다."
    )
    lines.append("")
    lines.append(f"- 생성 시각: {generated}")
    lines.append(f"- 실행 명령: `{command}`")
    lines.append(f"- 카탈로그: `{catalog.source}` — {len(catalog.items)}권, 임베딩 {catalog.dimension}차원")
    lines.append("- 외부 API 호출: 없음 (저장된 임베딩만 사용)")
    lines.append(
        f"- 원본 `chroma_db` md5 불변: {'예 (파일 %d개 동일)' % len(md5_before) if unchanged else '**아니오 — 확인 필요**'}"
    )
    for mode, queries in queries_by_mode.items():
        lines.append(f"- 질의 `{mode}`: {len(queries)}건")
    lines.append("")

    lines.append("## 1. 질의 방식 (대리 질의)")
    lines.append("")
    lines.append("| 모드 | 질의 벡터 | 선호 카테고리 | 후보 풀 | 건수 |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| `book` | 책 1권의 저장된 임베딩 | 그 책의 서브카테고리 | 전체 30권에서 그 책만 제외 | "
        f"{len(queries_by_mode.get('book', []))} |"
    )
    lines.append(
        "| `category` | 해당 서브카테고리 책들의 평균(단위 벡터) | 그 서브카테고리 하나 | 전체 30권 | "
        f"{len(queries_by_mode.get('category', []))} |"
    )
    lines.append("")
    lines.append(
        f"`category` 모드는 책이 {min_books}권 이상인 서브카테고리만 씁니다. 운영 질의는 "
        "서브카테고리 **이름 텍스트**를 임베딩한 것이라, 여기 질의(책 임베딩·평균)와 조건이 다릅니다. "
        "그래서 절대값이 아니라 **두 변형 사이의 상대 비교**로만 읽어야 합니다."
    )
    lines.append("")

    section = 2
    for mode in QUERY_MODES:
        if mode not in results:
            continue
        mode_outcomes = results[mode]
        queries = queries_by_mode[mode]
        lines.append(f"## {section}. 결과 — `{mode}` ({len(queries)}건, k={focus})")
        lines.append("")
        lines.append(f"| variant | 고유 서브카테고리@{focus} | 정규화 엔트로피@{focus} | 커버리지@{focus} |")
        lines.append("|---|---|---|---|")
        for key in VARIANT_KEYS:
            lines.append(_summary_row(mode_outcomes[key], focus))
        lines.append("")
        comparison = compare_variants(queries, mode_outcomes, focus)
        lines.append(
            f"두 변형의 top-{focus}가 같은 책으로 겹치는 정도: 평균 Jaccard "
            f"**{comparison['mean_topk_jaccard']:.3f}**, top-1 일치 "
            f"**{comparison['same_top1_share'] * 100:.1f}%** "
            f"({int(round(comparison['same_top1_share'] * len(queries)))}/{len(queries)}건)."
        )
        lines.append("")
        lines.append("### k별 값")
        lines.append("")
        header = "| variant | 지표 | " + " | ".join(f"k={k}" for k in ks) + " |"
        lines.append(header)
        lines.append("|---" * (len(ks) + 2) + "|")
        for key in VARIANT_KEYS:
            outcome = mode_outcomes[key]
            for metric, values, fmt in (
                ("고유 서브카테고리", outcome.distinct, lambda v: f"{v:.2f}"),
                (
                    "정규화 엔트로피",
                    outcome.entropy,
                    lambda v: f"{v:.3f}",
                ),
                ("커버리지", outcome.coverage, lambda v: f"{v * 100:.1f}%"),
            ):
                cells = " | ".join(
                    ("–" if metric == "정규화 엔트로피" and k <= 1 else fmt(values.get(k, 0.0)))
                    for k in ks
                )
                lines.append(f"| `{key}` | {metric} | {cells} |")
        lines.append("")
        lines.append(
            "k=1의 엔트로피는 정의상 0이라 값 대신 `–`로 두었습니다. "
            "k=1 행의 고유 서브카테고리 수는 '고른 책 1권이 가진 서브카테고리 수'의 평균입니다."
        )
        lines.append("")
        lines.extend(_render_examples(catalog, queries, mode_outcomes, focus))
        section += 1

    lines.append(f"## {section}. 읽는 법과 한계")
    lines.append("")
    lines.append(
        "- **정확도 아님.** 라벨이 없어 Recall·Precision·MRR·nDCG는 계산하지 않았습니다. "
        "관련도를 희생하고 다양성을 얻었을 수도 있으므로, 이 표만으로 \"개선\"이라고 말할 수 없습니다."
    )
    lines.append(
        "- **30권 카탈로그.** 커버리지는 분모가 30이라 작은 차이도 크게 보입니다. "
        "k=5면 질의 6건만으로 카탈로그 전체가 노출될 수 있습니다."
    )
    lines.append(
        "- **대리 질의.** 실제 사용자 선호는 서브카테고리 이름 텍스트의 임베딩입니다. "
        "여기서는 API 호출 없이 책 임베딩(과 그 평균)으로 대신했습니다."
    )
    lines.append(
        "- **제외 없음.** 이 실측은 중복 제외(`exclude_isbns`)를 켜지 않았습니다. "
        "제외는 후보 풀을 줄여 다양성에 영향을 주므로, 운영 수치와 같지 않습니다."
    )
    lines.append(
        "- 정의: 고유 서브카테고리 수와 엔트로피는 `eval/metrics.py`의 "
        "`distinct_category_count`/`category_entropy`(top-k 안 분포의 정규화 엔트로피), "
        "커버리지는 `coverage`(모든 질의의 top-k 합집합 / 카탈로그)를 그대로 썼습니다."
    )
    lines.append("")
    return "\n".join(lines)


def write_csv(
    path: str | Path, results: dict[str, dict[str, VariantOutcome]], ks: Sequence[int]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_mode", "variant", "k", "distinct_sub_categories", "entropy", "coverage"])
        for mode, outcomes in results.items():
            for variant, outcome in outcomes.items():
                for k in ks:
                    writer.writerow(
                        [
                            mode,
                            variant,
                            k,
                            f"{outcome.distinct.get(k, 0.0):.4f}",
                            f"{outcome.entropy.get(k, 0.0):.4f}",
                            f"{outcome.coverage.get(k, 0.0):.4f}",
                        ]
                    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Top-k 다양성·커버리지 실측 (30권 카탈로그, 정확도 아님)"
    )
    parser.add_argument("--db", default="chroma_db", help="Chroma 폴더 (복사본을 열어 원본 불변)")
    parser.add_argument("--collection", default="books")
    parser.add_argument(
        "--query-mode",
        default="both",
        choices=["both", *QUERY_MODES],
        help="both=book+category",
    )
    parser.add_argument("--min-books", type=int, default=2, help="category 질의 최소 책 수")
    parser.add_argument("--ks", default="1,3,5")
    parser.add_argument("--out", default="eval/diversity_benchmark.md")
    parser.add_argument("--csv", default="eval/diversity_benchmark.csv")
    args = parser.parse_args(argv)

    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    modes = list(QUERY_MODES) if args.query_mode == "both" else [args.query_mode]

    db_path = Path(args.db)
    md5_before = file_md5(db_path) if db_path.exists() else {}
    catalog = load_catalog_from_chroma(db_path, collection_name=args.collection)
    md5_after = file_md5(db_path) if db_path.exists() else {}

    queries_by_mode = {
        mode: build_queries(catalog, mode, min_books=args.min_books) for mode in modes
    }
    for mode, queries in queries_by_mode.items():
        if not queries:
            print(f"질의가 없습니다: {mode} (--min-books={args.min_books})", file=sys.stderr)
            return 1

    results = {
        mode: {
            key: run_variant(catalog, queries, key, ks) for key in VARIANT_KEYS
        }
        for mode, queries in queries_by_mode.items()
    }

    argv_for_display = sys.argv[1:] if argv is None else argv
    command = " ".join(["python", "scripts/diversity_benchmark.py", *argv_for_display])
    text = render_markdown(
        catalog=catalog,
        queries_by_mode=queries_by_mode,
        results=results,
        ks=ks,
        command=command,
        md5_before=md5_before,
        md5_after=md5_after,
        min_books=args.min_books,
    )
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"리포트 저장: {args.out}")
    if args.csv:
        write_csv(args.csv, results, ks)
        print(f"CSV 저장: {args.csv}")
    if md5_before != md5_after:
        print("경고: 원본 chroma_db 파일이 바뀌었습니다.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
