#!/usr/bin/env python3
"""추천 품질 오프라인 평가 CLI.

예시
----
형식만 점검 (API 키/라벨 없이):
    python scripts/eval_recommend.py --dry-run

실제 측정 (라벨 + OpenAI 키 필요):
    python scripts/eval_recommend.py --labels eval/labels.json

스모크 테스트 (파이프라인 확인용, 수치 해석 불가):
    python scripts/eval_recommend.py --labels eval/labels.example.json \
        --embedder fake --allow-synthetic

카탈로그 스냅샷 내보내기 (임베딩 제외):
    python scripts/eval_recommend.py --export-catalog eval/catalog_snapshot.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.catalog import (  # noqa: E402
    Catalog,
    CatalogError,
    catalog_statistics,
    load_catalog_from_chroma,
    load_catalog_from_json,
)
from eval.embedders import EmbedderError, build_embedder  # noqa: E402
from eval.harness import (  # noqa: E402
    LabelError,
    evaluate,
    load_labels,
    validate_labels,
)
from eval.report import RunContext, render_markdown, write_csv, write_markdown  # noqa: E402

EXIT_OK = 0
EXIT_NOT_MEASURED = 2

VARIANT_CHOICES = [
    "baseline_vector",
    "category_filter",
    "hybrid_category_boost",
    "production_hybrid",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AraBook 추천 품질 오프라인 평가")
    parser.add_argument("--labels", default="eval/labels.json", help="라벨 JSON 경로")
    parser.add_argument("--chroma-path", default="chroma_db", help="Chroma 폴더 (복사 후 읽기 전용으로 열림)")
    parser.add_argument("--collection", default="books", help="Chroma 컬렉션 이름")
    parser.add_argument("--catalog-json", default=None, help="카탈로그 JSON을 사용할 경우 경로")
    parser.add_argument("--embedder", choices=["openai", "fake"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-ada-002")
    parser.add_argument("--variants", default=",".join(VARIANT_CHOICES))
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--alpha", type=float, default=0.15, help="hybrid 변형의 카테고리 가중치")
    parser.add_argument("--out", default="eval/results.md")
    parser.add_argument("--csv", default="eval/results.csv")
    parser.add_argument("--cache", default="eval/cache/query_embeddings.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="라벨/카탈로그 검증만 하고 측정하지 않음")
    parser.add_argument("--export-catalog", default=None, help="카탈로그 메타데이터 스냅샷 저장 경로")
    parser.add_argument("--include-embeddings", action="store_true", help="스냅샷에 임베딩 포함(파일이 매우 커짐)")
    parser.add_argument("--allow-synthetic", action="store_true", help="예시/합성 라벨로도 계산(수치 해석 불가)")
    parser.add_argument("--print-stats", action="store_true", help="카탈로그 구성 통계 출력")
    return parser.parse_args(argv)


def load_catalog(args: argparse.Namespace) -> Catalog:
    if args.catalog_json:
        return load_catalog_from_json(args.catalog_json)
    return load_catalog_from_chroma(args.chroma_path, collection_name=args.collection)


def command_line(argv: list[str] | None) -> str:
    argv = sys.argv[1:] if argv is None else argv
    return "python scripts/eval_recommend.py " + " ".join(argv)


def print_catalog_stats(catalog: Catalog) -> None:
    stats = catalog_statistics(catalog)
    print(f"카탈로그: {catalog.source} — {stats['n_items']}권, 임베딩 차원 {stats['dimension']}")
    print(f"  대분류 라벨 {stats['n_main_category_labels']}종: {stats['main_categories']}")
    print(f"  서브카테고리 라벨 {stats['n_sub_category_labels']}종")
    print(f"  해시태그 {stats['n_hashtags']}종")
    if stats["multi_valued_main_category"]:
        print(
            "  ⚠️ 대분류 필드에 값이 2개 이상 들어간 책: "
            f"{len(stats['multi_valued_main_category'])}권"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    variant_keys = [v.strip() for v in args.variants.split(",") if v.strip()]

    try:
        catalog = load_catalog(args)
    except CatalogError as exc:
        print(f"카탈로그를 읽지 못했습니다: {exc}", file=sys.stderr)
        return EXIT_NOT_MEASURED

    if args.print_stats:
        print_catalog_stats(catalog)

    if args.export_catalog:
        _export_catalog(catalog, args.export_catalog, include_embeddings=args.include_embeddings)

    labels_path = Path(args.labels)
    try:
        labels = load_labels(labels_path)
    except LabelError as exc:
        context = RunContext(
            labels_path=str(labels_path),
            label_source="missing",
            n_users=0,
            catalog_source=catalog.source,
            n_items=len(catalog.items),
            embedder=args.embedder,
            ks=ks,
            alpha=args.alpha,
            command=command_line(argv),
            not_run_reason=str(exc),
        )
        return _finish_not_measured(context, args)

    validation = validate_labels(labels, catalog)
    context = RunContext(
        labels_path=str(labels_path),
        label_source=labels.label_source,
        n_users=len(labels.users),
        catalog_source=catalog.source,
        n_items=len(catalog.items),
        embedder=args.embedder,
        ks=ks,
        alpha=args.alpha,
        command=command_line(argv),
        validation=validation,
    )
    if args.embedder == "fake":
        context.warnings.append(
            "fake 임베더는 의미 기반 유사도가 아닙니다. 이 리포트의 수치는 파이프라인 "
            "동작 확인용이며 품질 지표로 인용하면 안 됩니다."
        )
    if labels.is_synthetic:
        context.warnings.append(
            f"라벨 출처가 '{labels.label_source}'입니다. 실제 사용자 라벨이 아니므로 수치는 의미가 없습니다."
        )

    if args.dry_run:
        context.not_run_reason = "--dry-run: 라벨/카탈로그 검증만 수행했습니다."
        return _finish_not_measured(context, args)

    if labels.is_synthetic and not args.allow_synthetic:
        context.not_run_reason = (
            f"라벨 출처가 '{labels.label_source}'인 예시 데이터입니다. 실제 측정에는 "
            "실제 사용자 라벨(eval/labels.json)이 필요합니다. 파이프라인 점검이면 "
            "--allow-synthetic을 붙이세요(그 수치는 인용 불가)."
        )
        return _finish_not_measured(context, args)

    try:
        embedder = build_embedder(
            args.embedder,
            model=args.embedding_model,
            cache_path=None if args.no_cache else args.cache,
        )
    except EmbedderError as exc:
        context.not_run_reason = str(exc)
        return _finish_not_measured(context, args)

    try:
        results = evaluate(
            labels=labels,
            catalog=catalog,
            embedder=embedder,
            variant_keys=variant_keys,
            ks=ks,
            alpha=args.alpha,
        )
    except CatalogError as exc:
        context.not_run_reason = str(exc)
        return _finish_not_measured(context, args)

    text = render_markdown(context, results)
    write_markdown(text, args.out)
    write_csv(results, args.csv, ks)
    print(text)
    print(f"리포트 저장: {args.out}, {args.csv}")
    return EXIT_OK


def _export_catalog(catalog: Catalog, path: str, include_embeddings: bool) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(catalog.to_json(include_embeddings=include_embeddings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"카탈로그 스냅샷 저장: {target} ({len(catalog.items)}권)")


def _finish_not_measured(context: RunContext, args: argparse.Namespace) -> int:
    text = render_markdown(context, None)
    write_markdown(text, args.out)
    print(text)
    print(f"리포트 저장: {args.out}")
    return EXIT_NOT_MEASURED


if __name__ == "__main__":
    raise SystemExit(main())
