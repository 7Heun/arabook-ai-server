"""평가 결과 리포트 생성 (Markdown + CSV).

리포트는 항상 같은 형식으로 생성된다. 측정을 못 했을 때도 그 사실과 사유를
남기므로, 나중에 누가 봐도 "빈칸인지 0인지" 헷갈리지 않는다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from eval.harness import ValidationReport, VariantResult
from eval.metrics import format_metric


@dataclass
class RunContext:
    labels_path: str
    label_source: str
    n_users: int
    catalog_source: str
    n_items: int
    embedder: str
    ks: Sequence[int]
    alpha: float
    command: str
    warnings: list[str] = field(default_factory=list)
    validation: ValidationReport | None = None
    not_run_reason: str | None = None


def _metric_header(ks: Sequence[int]) -> list[str]:
    headers: list[str] = []
    headers += [f"Recall@{k}" for k in ks]
    headers += [f"Precision@{k}" for k in ks]
    headers.append("MRR")
    headers += [f"nDCG@{k}" for k in ks]
    return headers


def render_markdown(
    context: RunContext,
    results: dict[str, VariantResult] | None,
) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    ks = list(context.ks)
    lines: list[str] = []
    lines.append("# 추천 품질 평가 결과")
    lines.append("")
    lines.append(f"- 생성 시각: {generated}")
    lines.append(f"- 실행 명령: `{context.command}`")
    lines.append(f"- 라벨 파일: `{context.labels_path}` (label_source: `{context.label_source}`)")
    lines.append(f"- 평가 사용자 수: {context.n_users}")
    lines.append(f"- 카탈로그: `{context.catalog_source}` ({context.n_items}권)")
    lines.append(f"- 임베딩: `{context.embedder}`")
    lines.append("")

    if context.warnings:
        lines.append("> ⚠️ 주의")
        for warning in context.warnings:
            lines.append(f"> - {warning}")
        lines.append("")

    if results is None:
        lines.append("## 수치 미산출")
        lines.append("")
        lines.append(context.not_run_reason or "평가를 실행하지 않았습니다.")
        lines.append("")
        lines.append(
            "이 파일은 하네스가 실제로 실행되어 생성되었지만, 위 사유로 지표 값은 "
            "계산되지 않았습니다. 측정하지 않은 수치를 채워 넣지 않습니다."
        )
        lines.append("")
        if context.validation:
            lines.extend(_render_validation(context.validation))
        return "\n".join(lines)

    lines.append("## 요약")
    lines.append("")
    header = ["variant"] + _metric_header(ks)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for result in results.values():
        agg = result.aggregate
        cells = [f"`{result.key}`"]
        cells += [format_metric(agg["recall"][k]["value"]) for k in ks]
        cells += [format_metric(agg["precision"][k]["value"]) for k in ks]
        cells.append(format_metric(agg["mrr"]["value"]))
        cells += [format_metric(agg["ndcg"][k]["value"]) for k in ks]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 커버리지 · 다양성")
    lines.append("")
    header2 = ["variant"] + [f"Coverage@{k}" for k in ks] + [f"서브카테고리 종류@{k}" for k in ks] + [f"정규화 엔트로피@{k}" for k in ks]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "|".join(["---"] * len(header2)) + "|")
    for result in results.values():
        agg = result.aggregate
        cells = [f"`{result.key}`"]
        cells += [f"{result.coverage[k]:.3f}" for k in ks]
        cells += [format_metric(agg["distinct_categories"][k]["value"], 2) for k in ks]
        cells += [format_metric(agg["entropy"][k]["value"]) for k in ks]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "nDCG는 등급 라벨(`graded_relevance`)이 있는 사용자만 평균에 포함됩니다. "
        "서브카테고리 종류/엔트로피는 사용자별 값의 평균입니다."
    )
    lines.append("")

    lines.append("## 변형 설명")
    lines.append("")
    lines.append("| variant | 설명 | 비고 |")
    lines.append("|---|---|---|")
    for result in results.values():
        lines.append(f"| `{result.key}` | {result.label} | {result.notes or '-'} |")
    lines.append("")

    if context.validation:
        lines.extend(_render_validation(context.validation))

    lines.append("## 읽는 방법")
    lines.append("")
    lines.append(
        "- 표본이 작으면(사용자 수십 명 이하) 지표 차이는 노이즈일 수 있습니다. "
        "변형 간 차이를 주장하려면 사용자 단위 부트스트랩 신뢰구간을 함께 보세요."
    )
    lines.append(
        "- Precision@k의 분모는 k입니다. 카탈로그가 k보다 작으면 값이 낮게 나옵니다."
    )
    lines.append(
        "- 이 표는 오프라인 지표입니다. 실제 서비스 품질은 클릭/대여/구매 같은 "
        "온라인 지표로 따로 확인해야 합니다."
    )
    lines.append("")
    return "\n".join(lines)


def _render_validation(validation: ValidationReport) -> list[str]:
    lines = ["## 라벨 검증", ""]
    lines.append(f"- 사용자 수: {validation.n_users}")
    lines.append(f"- 선호 카테고리가 없는 사용자: {len(validation.n_users_without_preferences)}명")
    lines.append(f"- 정답 ISBN이 없는 사용자: {len(validation.n_users_without_relevant)}명")
    if validation.n_relevant_values:
        values = validation.n_relevant_values
        lines.append(
            f"- 사용자당 정답 수: 최소 {min(values)} / 최대 {max(values)} / "
            f"평균 {sum(values) / len(values):.1f}"
        )
    if validation.unknown_relevant_isbns:
        total = sum(len(v) for v in validation.unknown_relevant_isbns.values())
        lines.append(
            f"- 카탈로그에 없는 정답 ISBN: {total}건 "
            f"({len(validation.unknown_relevant_isbns)}명). 해당 항목은 Recall 계산에서 "
            "빠지므로 Recall이 실제보다 낮게 나옵니다."
        )
    if validation.unknown_preference_labels:
        total = sum(len(v) for v in validation.unknown_preference_labels.values())
        lines.append(
            f"- 카탈로그 서브카테고리에 없는 선호 라벨: {total}건 "
            f"({len(validation.unknown_preference_labels)}명)"
        )
    lines.append("")
    return lines


def write_csv(results: dict[str, VariantResult], path: str | Path, ks: Sequence[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "metric", "k", "value", "n"])
        for result in results.values():
            for k in ks:
                writer.writerow([result.key, "recall", k, _csv(result.aggregate["recall"][k]["value"]), result.aggregate["recall"][k]["n"]])
                writer.writerow([result.key, "precision", k, _csv(result.aggregate["precision"][k]["value"]), result.aggregate["precision"][k]["n"]])
                writer.writerow([result.key, "ndcg", k, _csv(result.aggregate["ndcg"][k]["value"]), result.aggregate["ndcg"][k]["n"]])
                writer.writerow([result.key, "distinct_categories", k, _csv(result.aggregate["distinct_categories"][k]["value"]), result.aggregate["distinct_categories"][k]["n"]])
                writer.writerow([result.key, "entropy", k, _csv(result.aggregate["entropy"][k]["value"]), result.aggregate["entropy"][k]["n"]])
                writer.writerow([result.key, "coverage", k, f"{result.coverage[k]:.6f}", len(result.ranked)])
            writer.writerow([result.key, "mrr", "", _csv(result.aggregate["mrr"]["value"]), result.aggregate["mrr"]["n"]])


def _csv(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def write_markdown(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
