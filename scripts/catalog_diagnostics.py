#!/usr/bin/env python3
"""카탈로그 임베딩 공간 진단 (라벨 불필요).

**이것은 추천 정확도가 아니다.** 사람이 붙인 정답 라벨 없이 지금 저장된 벡터만으로
"임베딩 공간에서 카테고리가 어느 정도 뭉쳐 있는가"를 확인하는 자기 점검이다.

무엇을 재는가
- 각 책의 최근접 이웃 1권(자기 자신 제외)이 서브카테고리를 공유하는 비율
- 상위 5권 중 하나라도 서브카테고리를 공유하는 비율
- 위 값의 **우연 기대값**(카테고리 빈도만으로 계산)과의 비율(lift)

왜 의미가 제한적인가
- 책 벡터는 제목·저자·설명·카테고리·해시태그를 모두 이어 붙여 만든다. 즉 카테고리
  정보가 벡터에 이미 들어 있어, 이웃이 같은 카테고리인 것은 어느 정도 당연하다.
- 실제 추천 질의는 "서브카테고리 이름"만 임베딩한 짧은 문자열이다. 따라서 이 수치는
  **추천 정확도의 추정치가 아니며**, 카탈로그가 커졌을 때 카테고리가 뒤섞이지 않는지
  보는 진단으로만 쓴다.

사용:
    python scripts/catalog_diagnostics.py --db chroma_db
    python scripts/catalog_diagnostics.py --db chroma_db --out eval/catalog_diagnostics.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from eval.catalog import Catalog, load_catalog_from_chroma, load_catalog_from_json  # noqa: E402


def diagnose(catalog: Catalog, top_k: int = 5) -> dict:
    if not catalog.has_embeddings:
        raise SystemExit("카탈로그에 임베딩이 없습니다. chroma_db에서 읽어 주세요.")

    matrix = catalog.embedding_matrix().astype("float64")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-12)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)

    items = catalog.items
    n = len(items)
    cat_sets = [set(item.sub_categories) for item in items]

    def shares(i: int, j: int) -> bool:
        return bool(cat_sets[i] & cat_sets[j])

    order = np.argsort(-similarity, axis=1, kind="stable")
    hit1 = sum(1 for i in range(n) if shares(i, int(order[i, 0])))
    hitk = sum(1 for i in range(n) if any(shares(i, int(j)) for j in order[i, :top_k]))

    # 우연 기대값: 카테고리 구성을 유지한 채 무작위로 이웃을 뽑았을 때의 비율
    expected = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if not others:
            continue
        expected.append(sum(1 for j in others if shares(i, j)) / len(others))
    expected_mean = float(np.mean(expected)) if expected else 0.0

    nearest_index = order[:, 0]
    hub_counts = np.bincount(nearest_index, minlength=n)

    worst = [
        {
            "isbn": items[i].isbn,
            "title": items[i].title,
            "sub_categories": list(items[i].sub_categories),
            "neighbor_title": items[int(nearest_index[i])].title,
            "neighbor_sub_categories": list(items[int(nearest_index[i])].sub_categories),
            "similarity": float(similarity[i, int(nearest_index[i])]),
        }
        for i in range(n)
        if not shares(i, int(nearest_index[i]))
    ]
    worst.sort(key=lambda row: -row["similarity"])

    nearest_sims = [float(similarity[i, int(nearest_index[i])]) for i in range(n)]
    return {
        "n_items": n,
        "dimension": catalog.dimension,
        "top_k": top_k,
        "hit_at_1": hit1,
        "hit_at_k": hitk,
        "expected_at_1": expected_mean,
        "lift_at_1": (hit1 / n) / expected_mean if expected_mean else None,
        "share_at_1": hit1 / n,
        "share_at_k": hitk / n,
        "nearest_similarity_min": float(np.min(nearest_sims)),
        "nearest_similarity_median": float(np.median(nearest_sims)),
        "nearest_similarity_max": float(np.max(nearest_sims)),
        "hub": [
            {"isbn": items[i].isbn, "title": items[i].title, "count": int(hub_counts[i])}
            for i in np.argsort(-hub_counts)[:5]
            if hub_counts[i] > 0
        ],
        "no_shared_category": worst,
    }


def format_markdown(result: dict, source: str) -> str:
    share1 = result["share_at_1"] * 100
    sharek = result["share_at_k"] * 100
    expected = result["expected_at_1"] * 100
    lift = result["lift_at_1"]
    lines = [
        "# 카탈로그 임베딩 진단 (정확도 아님)",
        "",
        f"- 출처: `{source}`",
        f"- 도서 {result['n_items']}권, 임베딩 {result['dimension']}차원, 코사인 유사도 기준",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 최근접 1권이 서브카테고리를 공유 | {result['hit_at_1']}/{result['n_items']} ({share1:.1f}%) |",
        f"| 우연 기대값(카테고리 빈도 기준) | {expected:.1f}% |",
        f"| lift (관측 / 우연) | {lift:.2f}× |" if lift else "| lift | – |",
        f"| 상위 {result['top_k']}권 중 하나라도 공유 | {result['hit_at_k']}/{result['n_items']} ({sharek:.1f}%) |",
        f"| 최근접 이웃 유사도 (최소/중앙/최대) | {result['nearest_similarity_min']:.3f} / {result['nearest_similarity_median']:.3f} / {result['nearest_similarity_max']:.3f} |",
        "",
        "## 최근접 이웃이 가장 많이 등장한 책 (hub)",
        "",
        "| 제목 | 최근접 이웃으로 선택된 횟수 |",
        "|---|---|",
    ]
    for row in result["hub"]:
        lines.append(f"| {row['title'][:40]} | {row['count']} |")
    lines.append("")
    lines.append(f"## 최근접 이웃과 서브카테고리가 전혀 겹치지 않는 책 ({len(result['no_shared_category'])}권)")
    lines.append("")
    if result["no_shared_category"]:
        lines.append("| 제목 | 서브카테고리 | 최근접 이웃 | 유사도 |")
        lines.append("|---|---|---|---|")
        for row in result["no_shared_category"]:
            lines.append(
                f"| {row['title'][:30]} | {', '.join(row['sub_categories'])} | "
                f"{row['neighbor_title'][:30]} ({', '.join(row['neighbor_sub_categories'])}) | "
                f"{row['similarity']:.3f} |"
            )
    else:
        lines.append("없음")
    lines.append("")
    lines.append("## 해석 주의")
    lines.append("")
    lines.append(
        "- 책 벡터에는 카테고리·해시태그가 포함되어 있어, 이웃이 같은 카테고리인 것은 "
        "부분적으로 당연한 결과입니다. **추천 정확도로 인용하면 안 됩니다.**"
    )
    lines.append(
        "- 실제 질의는 서브카테고리 이름만 임베딩한 짧은 문자열이라 이 진단과 조건이 다릅니다."
    )
    lines.append(
        "- 사람이 붙인 정답 라벨이 없으므로, 추천 품질은 `eval/README.md`의 평가 하네스로 "
        "따로 측정해야 합니다(현재는 라벨 부재로 수치 미산출)."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="카탈로그 임베딩 공간 진단 (읽기 전용)")
    parser.add_argument("--db", default="chroma_db", help="Chroma 폴더")
    parser.add_argument("--catalog-json", default=None, help="임베딩이 포함된 카탈로그 JSON")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    catalog = (
        load_catalog_from_json(args.catalog_json)
        if args.catalog_json
        else load_catalog_from_chroma(args.db)
    )
    result = diagnose(catalog, top_k=args.top_k)
    text = format_markdown(result, catalog.source)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
