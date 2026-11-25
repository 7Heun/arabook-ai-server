# 추천 품질 평가 (오프라인)

"이 추천이 좋은가?"를 감이 아니라 숫자로 답하기 위한 하네스입니다.
운영 추천 로직과 개선안 후보를 **같은 라벨셋**으로 돌려 Recall·Precision·MRR·
커버리지·다양성을 비교합니다.

> **현재 상태: 수치 미산출.**
> 저장소에는 아직 실제 사용자 라벨이 없습니다. `eval/labels.example.json`은
> 형식 예시(합성)일 뿐이라 하네스가 측정을 거부합니다. 이 문서의 표는
> "측정 방법"이지 "측정 결과"가 아닙니다.

---

## 1. 무엇을 측정하는가

| 항목 | 내용 |
|---|---|
| 대상 | `/recommend`가 고르는 책 1권(과 상위 k권) |
| 입력 | 사용자가 고른 서브카테고리 목록 (`member_sub_category_selections`) |
| 비교 | `baseline_vector`(현재 운영 로직) vs 개선 후보 2개 |
| 단위 | 사용자별 순위 목록 → 지표 평균 |
| 데이터 | 라벨 JSON + 도서 임베딩(Chroma 스냅샷) |

운영 코드의 추천은 "선호 서브카테고리 이름들을 공백으로 이어 붙인 문자열"을
임베딩해 Chroma에서 가장 가까운 책 1권을 가져옵니다
(`ChromaManager.recommend_book`, `n_results=1`). 평가의 baseline은 이 동작을
그대로 재현합니다. 컬렉션의 거리 공간은 Chroma 기본값인 **L2**이고, 저장된
임베딩(ada-002)의 L2 노름은 실측 0.9999999~1.0000001로 단위 벡터이므로
L2 순위와 코사인 순위는 같습니다. 그래서 baseline은 L2로, 가중치를 더하는
변형은 코사인으로 계산합니다.

## 2. 라벨 데이터

### 형식
[`labels.example.json`](labels.example.json) / 스키마 [`labels.schema.json`](labels.schema.json)

```json
{
  "meta": { "format_version": 1, "label_source": "mysql-loans-2026-09..11" },
  "users": [
    {
      "member_id": 1001,
      "preferences": ["추리/스릴러"],
      "relevant_isbns": ["9791131963746", "9791158790745"],
      "graded_relevance": { "9791131963746": 2 }
    }
  ]
}
```

| 필드 | 필수 | 의미 |
|---|---|---|
| `member_id` | O | 사용자 식별자 |
| `preferences` | O | 사용자가 고른 서브카테고리 이름. 카탈로그의 `subCategory` 값과 같아야 함 |
| `relevant_isbns` | O | 관련 있다고 라벨링한 ISBN13. 비어 있으면 Recall 계산에서 제외 |
| `graded_relevance` | X | 2=매우 관련, 1=관련. 있으면 nDCG도 계산 |

### 실제 라벨을 만드는 방법 (계획)
1. MySQL의 **양성 행동**을 모읍니다: 대여, 구매, 좋아요/서재 추가, 리뷰 작성.
2. 사용자별로 `(member_id, isbn, 행동, 시각)`을 뽑고, 평가 시점 이전 행동만 씁니다
   (미래 정보 누설 방지). 이 저장소의 추천은 하루 1권이므로, 최소 1주일 이상의
   행동을 모아 사용자당 정답 5~10권을 확보하는 것이 목표입니다.
3. **노출 편향**을 줄입니다. 추천으로 노출된 책만 정답으로 쓰면 "추천기가 추천한
   책을 추천기가 잘 맞힌다"는 순환 논리가 됩니다. 사용자가 스스로 검색해 읽은
   책을 포함하거나, 노출 로그를 별도로 두고 평가에서 제외/보정합니다.
4. 라벨링 시점의 카탈로그에 없는 ISBN은 제거하거나 `Recall` 분모에서 빠진다는
   사실을 리포트에 남깁니다(하네스가 자동 집계).
5. `meta.label_source`에 수집 방법과 기간을 적습니다. `example`/`synthetic`으로
   시작하는 값이면 하네스가 측정을 거부합니다(실수로 합성 수치가 커밋되는 것 방지).

### 카탈로그 (실측, 2026-11-26)
`python scripts/catalog_stats.py --db chroma_db` 실행 결과:

| 항목 | 값 |
|---|---|
| 도서 수 | 30권 |
| 임베딩 차원 | 1536 (ada-002와 동일) |
| 대분류 라벨 종류 | 8종 (소설 12, 과학 7, 인문 3, 소설·역사 2, 자기계발 2, 시/에세이 2, 자기계발·사회 1, 역사 1) |
| 서브카테고리 라벨 종류 | 32종 |
| 해시태그 종류 | 113종 |
| 책당 평균 서브카테고리 수 | 2.1 |

30권은 평가에 너무 작습니다. `Recall@10`은 카탈로그의 1/3을 반환하는 것이라
지표가 쉽게 포화되고, 카테고리 다양성도 의미를 갖기 어렵습니다. **최소 수백 권**
이상 쌓은 뒤 측정하는 것을 전제로 만들었습니다. 다만 지표 구현 자체는 합성
카탈로그로 단위 테스트해 두었습니다(`tests/test_eval_metrics.py`).

## 3. 지표와 선택 이유

| 지표 | 정의 | 왜 보는가 | 주의 |
|---|---|---|---|
| Recall@k | top-k 중 정답 비율 / 전체 정답 수 | "읽을 만한 책을 놓치지 않는가" | k가 정답 수보다 작으면 Recall 상한이 k ÷ 정답 수 |
| Precision@k | top-k 중 정답 비율 / k | "추천한 것이 얼마나 맞는가" | 카탈로그가 k보다 작거나 반환 수가 적으면 불리 |
| MRR | 첫 정답의 역순위 평균 | 하루 1권을 보여주는 서비스라 **첫 순위**가 중요 | 1위만 맞으면 1.0 |
| nDCG@k | 등급 라벨 기반 순위 품질 | "매우 관련"을 위에 놓았는가 | 등급 라벨 있는 사용자만 평균 |
| Coverage@k | 모든 사용자의 top-k 합집합 / 카탈로그 | 특정 책만 반복 추천되는지 | 사용자 수가 적으면 낮게 나옴 |
| 다양성 | top-k 서브카테고리 분포의 정규화 엔트로피, 서브카테고리 종류 수 | 추천이 한 장르에 몰리는지 | 다중 라벨 책은 1/n로 나눠 계산 |

- **Recall과 Precision을 함께** 봅니다. 하나만 보면 "다 추천하기" 또는 "1권만
  확실히" 쪽으로 치우친 해를 구분할 수 없습니다.
- **MRR을 넣은 이유**는 서비스 형태 때문입니다. `/recommend`는 하루 1권을
  저장하므로, 10위권에 정답이 있는 것은 사용자 경험에 거의 의미가 없습니다.
- 지표 정의·경계값은 `tests/test_eval_metrics.py`에서 손 계산 값으로 검증합니다
  (중복 추천, 정답 없음, k > 목록 길이, 다중 라벨 등).

## 4. 비교 변형

| key | 내용 | 상태 |
|---|---|---|
| `baseline_vector` | 현재 운영: 선호 텍스트 임베딩 → L2 최근접 | 측정 대상(현행) |
| `category_filter` | 선호 서브카테고리와 겹치는 책만 후보 → 벡터 정렬. 후보가 없으면 전체 | 가설 |
| `hybrid_category_boost` | `cosine + alpha × 카테고리 일치율` (alpha=0.15 고정) | 가설 |

`alpha`는 평가셋을 보고 고른 값이 **아닙니다**. 평가셋으로 튜닝하면 그 평가셋에서만
좋은 값이 되어 비교가 무의미해지므로, 사전에 정한 값을 쓰고 튜닝은 별도 dev
split에서 해야 한다는 것을 코드 주석과 리포트에 남겼습니다.

주의: `category_filter`는 후보를 좁히므로 **Recall 상한이 카테고리 일치 책 수**로
제한됩니다. 카테고리 라벨이 거칠면(예: "사랑" 8권) 다양성이 떨어질 수 있습니다.
이 가설이 맞는지는 라벨로만 판단할 수 있습니다.

## 5. 라벨 없이 할 수 있는 점검

`scripts/catalog_diagnostics.py`는 저장된 임베딩만으로 "최근접 이웃이 같은 서브카테고리인가"를
확인합니다(우연 기대값 대비 lift). 카테고리 정보가 벡터에 실려 있는지, 유사도가 지나치게
압축되어 있지는 않은지 보는 **진단**이며, 추천 정확도가 아닙니다. 결과는
[`catalog_diagnostics.md`](catalog_diagnostics.md)에 있습니다(30권, lift 3.85×,
최근접 유사도 0.839~0.931).

## 6. 실행

```bash
# 0) 의존성
pip install -r requirements-dev.txt

# 1) 형식 점검 (API 키·라벨 없이)
python scripts/eval_recommend.py --dry-run --labels eval/labels.example.json --embedder fake

# 2) 실제 측정 (eval/labels.json + OPENAI_API_KEY 필요)
python scripts/eval_recommend.py --labels eval/labels.json

# 3) 파이프라인 스모크 테스트 (수치 인용 불가)
python scripts/eval_recommend.py --labels eval/labels.example.json \
    --embedder fake --allow-synthetic --out /tmp/smoke.md

# 4) 카탈로그 통계 (chromadb 불필요, 읽기 전용)
python scripts/catalog_stats.py --db chroma_db

# 5) 임베딩 공간 진단 (라벨 불필요, 정확도 아님)
python scripts/catalog_diagnostics.py --db chroma_db
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--chroma-path` | `chroma_db` | 임시 복사본을 열어 원본을 건드리지 않음 |
| `--catalog-json` | – | `catalog_snapshot.json` 같은 스냅샷 사용 |
| `--embedder` | `openai` | `fake`는 의미 없는 해시 임베딩(스모크 전용) |
| `--embedding-model` | `text-embedding-ada-002` | 운영과 같은 모델 |
| `--variants` | 3종 전부 | 쉼표 구분 |
| `--ks` | `1,3,5,10` | |
| `--alpha` | `0.15` | hybrid 가중치 |
| `--allow-synthetic` | off | 합성 라벨 계산 허용(경고 기록) |
| `--export-catalog` | – | 메타데이터 스냅샷 저장 |

> 새로 clone한 환경에는 `chroma_db/`가 없습니다(저장소에 포함하지 않음). 이때는
> `--catalog-json eval/catalog_snapshot.json`으로 형식 점검(`--dry-run`)만 가능하고,
> 실제 측정에는 Chroma 데이터가 필요합니다. 스냅샷에는 임베딩이 없습니다.

결과는 `eval/results.md`(사람이 읽는 표)와 `eval/results.csv`(지표별 long format)로
떨어집니다. 측정을 못 하면 `results.md`에 **사유와 함께 "미산출"**로 남고, 0으로
채우지 않습니다.

## 7. 재현성

- `catalog_snapshot.json`: 2026-11-26 기준 카탈로그 30권의 메타데이터 스냅샷
  (임베딩 제외). `chroma_db/`는 더 이상 git에 올리지 않으므로, 카탈로그 구성이
  어떻게 바뀌었는지 비교하는 기준점으로 씁니다.
- 순위는 `numpy.argsort(kind="stable")` + ISBN 순서로 결정적입니다. 같은 입력이면
  같은 결과가 나옵니다.
- 질의 임베딩은 `eval/cache/`에 캐시합니다(캐시 파일은 커밋하지 않음). 같은
  라벨로 다시 돌릴 때 API 호출을 줄입니다.
- `chroma_db/`를 열 때는 항상 임시 폴더로 복사한 사본을 엽니다. 평가가 원본을
  바꾸지 않습니다.

## 8. 한계 (지금 시점)

1. **실제 라벨 없음** → 수치 없음. 이 문서와 `results.md`는 방법론까지만 제공합니다.
2. 카탈로그 30권 → 지표가 포화되고 분산이 큽니다. 사용자 수십 명으로는 변형 간
   차이를 주장할 수 없습니다.
3. 라벨은 "관련 있음/없음" 이진 판단이라 사용자 만족(별점, 완독률)을 담지 못합니다.
4. 오프라인 지표는 온라인 지표(클릭률, 대여 전환)와 다를 수 있습니다.
5. 콜드 스타트 사용자(선호 카테고리 없음)는 이 평가에서 제외됩니다. 운영에서는
   `"No preferences found for the member."`를 반환하므로 별도 과제입니다.
6. 단일 정답 1권을 반환하는 현행 API에서는 Recall@k의 의미가 제한적입니다.
   API를 top-k로 확장할 때 진짜 값이 생깁니다.

## 9. 다음 실험 계획

1. MySQL 행동 로그로 `eval/labels.json` 생성 스크립트(`scripts/build_labels.py`)를
   만들고, 라벨 생성 시점과 카탈로그를 고정한다.
2. 사용자 단위 **dev/test split**(예: 7:3)을 두고, dev에서만 alpha를 튜닝한 뒤
   test에서 한 번 평가한다.
3. 사용자 단위 부트스트랩으로 지표 **신뢰구간**을 계산해, 변형 간 차이가 노이즈인지
   확인한다.
4. 카탈로그가 수백 권 이상이 되면 `category_filter`의 Recall 상한 문제를
   "카테고리 soft filter(가중)"와 비교한다.
5. 오프라인에서 고른 변형을 온라인 A/B(추천 노출 → 대여 전환)로 검증한다.
