# 추천 품질 평가 결과

- 생성 시각: 2026-11-26 04:11:57 KST
- 실행 명령: `python scripts/eval_recommend.py --labels eval/labels.example.json --embedder fake --dry-run`
- 라벨 파일: `eval/labels.example.json` (label_source: `synthetic-format-example`)
- 평가 사용자 수: 3
- 카탈로그: `chroma:chroma_db` (30권)
- 임베딩: `fake`

> ⚠️ 주의
> - fake 임베더는 의미 기반 유사도가 아닙니다. 이 리포트의 수치는 파이프라인 동작 확인용이며 품질 지표로 인용하면 안 됩니다.
> - 라벨 출처가 'synthetic-format-example'입니다. 실제 사용자 라벨이 아니므로 수치는 의미가 없습니다.

## 수치 미산출

--dry-run: 라벨/카탈로그 검증만 수행했습니다.

이 파일은 하네스가 실제로 실행되어 생성되었지만, 위 사유로 지표 값은 계산되지 않았습니다. 측정하지 않은 수치를 채워 넣지 않습니다.

## 라벨 검증

- 사용자 수: 3
- 선호 카테고리가 없는 사용자: 0명
- 정답 ISBN이 없는 사용자: 0명
- 사용자당 정답 수: 최소 2 / 최대 3 / 평균 2.3
