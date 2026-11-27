# 추천 품질 평가 (오프라인)

"이 추천이 좋은가?"를 감이 아니라 숫자로 답하기 위한 하네스입니다.
운영 추천 로직과 개선안 후보를 **같은 라벨셋**으로 돌려 Recall·Precision·MRR·
커버리지·다양성을 비교합니다.

> **현재 상태**
> - **정확도 지표: 수치 미산출.** 저장소에는 아직 실제 사용자 라벨이 없습니다.
>   `eval/labels.example.json`은 형식 예시(합성)일 뿐이라 하네스가 측정을 거부합니다.
>   이 문서의 표는 "측정 방법"이지 "측정 결과"가 아닙니다.
> - **다양성·커버리지: 30권 카탈로그에서 실측함.** 라벨 없이도 잴 수 있는 지표만 따로
>   계산했습니다([§5](#5-라벨 없이-할-수-있는-점검), [`diversity_benchmark.md`](diversity_benchmark.md)).
>   이것도 **추천 정확도가 아닙니다.**

---

## 1. 무엇을 측정하는가

| 항목 | 내용 |
|---|---|
| 대상 | `/recommend`가 고르는 책 1권(과 상위 k권) |
| 입력 | 사용자가 고른 서브카테고리 목록 (`member_sub_category_selections`) |
| 비교 | `baseline_vector`(이전 운영) · 개선 후보 2개 · `production_hybrid`(현재 운영) |
| 단위 | 사용자별 순위 목록 → 지표 평균 |
| 데이터 | 라벨 JSON + 도서 임베딩(Chroma 스냅샷) |

운영 코드의 추천은 이제 "후보 20권을 하이브리드(코사인 + 카테고리 가중 + MMR)로
재정렬한 뒤 1권"입니다(`app/services/recommendation.py` → `app/services/reranking.py`).
평가의 `production_hybrid` 변형은 **그 함수를 그대로 import**해 같은 순위를 만듭니다.
`baseline_vector`는 이전 운영(L2 최근접 1권)의 재현으로, 비교 기준으로만 남겨 두었습니다.
컬렉션의 거리 공간은 Chroma 기본값인 **L2**이고, 저장된
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

### 실제 라벨 만드는 법 — `scripts/export_labels.py`

```bash
python scripts/export_labels.py \
  --since 2026-09-01 --until 2026-12-01 \
  --action "table=rentals,member=member_id,book=book_id,time=rented_at,weight=2" \
  --action "table=book_likes,member=member_id,book=book_id,time=created_at,weight=1" \
  --exclude-recommended --out eval/labels.json
```

1. 행동 테이블 이름·컬럼은 이 저장소에 없으므로 인자로 받는다(식별자는 정규식으로
   검증한 뒤 SQL에 넣는다). 틀린 이름은 0건이 아니라 SQL 오류로 바로 드러난다.
2. `weight`가 큰 행동이 그 책의 등급이 된다(2=대여/완독, 1=좋아요). 여러 행동이 있으면
   최댓값을 쓴다.
3. `--exclude-recommended`는 추천 노출 이력(`daily_book_id_recommendations`)에 있는
   (회원, ISBN)을 정답에서 뺀다 — 노출 편향 완화.
4. 기본값으로 카탈로그(chroma_db)에 없는 ISBN을 제거한다(어떤 추천기도 고를 수 없는
   정답은 Recall 분모를 왜곡한다). `--no-catalog-filter`로 끈다.
5. 결과는 `eval/labels.schema.json`으로 검증하고, 통과해야 파일을 쓴다.
   `eval/labels.json`은 실제 사용자 데이터라 커밋하지 않는다(.gitignore).

이전 계획 문서에 적었던 단계(양성 행동 수집 → 사용자별 정답 → 노출 편향 제거 →
카탈로그 필터)를 그대로 구현한 것이며, **실행에는 운영 DB 접속 정보가 필요**하다.

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
| `baseline_vector` | 이전 운영: 선호 텍스트 임베딩 → L2 최근접 | 비교 기준(과거) |
| `category_filter` | 선호 서브카테고리와 겹치는 책만 후보 → 벡터 정렬. 후보가 없으면 전체 | 가설 |
| `hybrid_category_boost` | `cosine + alpha × 카테고리 일치율` (alpha=0.15 고정) | 가설 |
| `production_hybrid` | 현재 운영: `cosine + alpha × 카테고리 일치율` → MMR(lambda=0.85) | 현행 (운영과 같은 함수) |

`production_hybrid`는 `app/services/reranking.py`의 `build_candidates`/`mmr_rank`를
그대로 호출합니다. 운영은 Chroma 상위 `pool_size`(기본 20)권을 후보로 쓰고,
평가에서는 카탈로그 전체를 후보로 씁니다(순위 비교 목적).

`alpha`는 평가셋을 보고 고른 값이 **아닙니다**. 평가셋으로 튜닝하면 그 평가셋에서만
좋은 값이 되어 비교가 무의미해지므로, 사전에 정한 값을 쓰고 튜닝은 별도 dev
split에서 해야 한다는 것을 코드 주석과 리포트에 남겼습니다. `mmr_lambda`도 같습니다.

주의: `category_filter`는 후보를 좁히므로 **Recall 상한이 카테고리 일치 책 수**로
제한됩니다. 카테고리 라벨이 거칠면(예: "사랑" 8권) 다양성이 떨어질 수 있습니다.
이 가설이 맞는지는 라벨로만 판단할 수 있습니다.

## 5. 라벨 없이 할 수 있는 점검

### 5.1 임베딩 공간 진단

`scripts/catalog_diagnostics.py`는 저장된 임베딩만으로 "최근접 이웃이 같은 서브카테고리인가"를
확인합니다(우연 기대값 대비 lift). 카테고리 정보가 벡터에 실려 있는지, 유사도가 지나치게
압축되어 있지는 않은지 보는 **진단**이며, 추천 정확도가 아닙니다. 결과는
[`catalog_diagnostics.md`](catalog_diagnostics.md)에 있습니다(30권, lift 3.85×,
최근접 유사도 0.839~0.931).

### 5.2 Top-5 다양성·커버리지 실측 — `scripts/diversity_benchmark.py`

라벨 없이 잴 수 있는 것 중 하나는 **"순위 규칙이 만드는 목록이 카탈로그 안에서 얼마나
퍼지는가"** 입니다. 이 스크립트는 `chroma_db`의 실제 임베딩 30권만 읽어(외부 API 호출 없음,
원본은 임시 복사본만 열고 md5 불변 확인) `baseline_vector`와 `production_hybrid`의 Top-k를
비교합니다.

질의는 저장된 벡터로 만든 **대리 질의**입니다(운영 질의는 서브카테고리 이름 텍스트의 임베딩).

| 모드 | 질의 벡터 | 선호 | 후보 풀 | 건수 |
|---|---|---|---|---|
| `book` | 책 1권의 임베딩 | 그 책의 서브카테고리 | 30권에서 그 책만 제외 | 30 |
| `category` | 서브카테고리 책들의 평균(단위 벡터) | 그 서브카테고리 | 전체 30권 | 13 |

2026-11-26 실행 결과 ([`diversity_benchmark.md`](diversity_benchmark.md)):

| 질의 | variant | 고유 서브카테고리@5 | 엔트로피@5 | 커버리지@3 | 커버리지@5 |
|---|---|---|---|---|---|
| `book` | `baseline_vector` | 7.03 | 0.937 | 80.0% | 93.3% |
| `book` | `production_hybrid` | 7.83 | 0.954 | 100.0% | 100.0% |
| `category` | `baseline_vector` | 7.15 | 0.911 | 80.0% | 83.3% |
| `category` | `production_hybrid` | 7.69 | 0.917 | 80.0% | 90.0% |

두 변형의 Top-5는 실제로 다릅니다(평균 Jaccard 0.535(`book`)·0.696(`category`),
top-1 일치 46.7%·76.9%).

**해석 주의**

- 정확도가 아닙니다. 라벨이 없어 Recall·Precision·MRR·nDCG는 계산하지 않았습니다.
- MMR은 관련도를 포기하고 다양성을 얻습니다. 하이브리드 목록의 2~5위에는 관련도가 낮은
  다른 장르 책이 섞이는 경우가 있습니다. 어느 쪽이 나은지는 라벨로만 판단할 수 있습니다.
- 30권·대리 질의·중복 제외(`exclude_isbns`) 미적용 조건입니다. 운영 수치가 아닙니다.
- 지표 정의는 `metrics.py`의 `distinct_category_count`/`category_entropy`/`coverage`를
  그대로 썼습니다(별도 정의를 만들지 않았습니다).

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

# 6) Top-5 다양성·커버리지 실측 (라벨·API 키 불필요, 정확도 아님)
python scripts/diversity_benchmark.py
python scripts/diversity_benchmark.py --query-mode book --ks 1,3,5
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

1. **실제 라벨 없음** → 수치 없음. 라벨 추출 스크립트는 준비됐지만(`scripts/export_labels.py`),
   운영 DB 접속이 필요한 단계라 이 작업 환경에서는 실행하지 못했습니다.
2. 카탈로그 30권 → 지표가 포화되고 분산이 큽니다. 사용자 수십 명으로는 변형 간
   차이를 주장할 수 없습니다.
3. 라벨은 "관련 있음/없음" 이진 판단이라 사용자 만족(별점, 완독률)을 담지 못합니다.
4. 오프라인 지표는 온라인 지표(클릭률, 대여 전환)와 다를 수 있습니다.
5. 콜드 스타트 사용자(선호 카테고리 없음)는 이 평가에서 제외됩니다. 운영에서는
   `"No preferences found for the member."`를 반환하므로 별도 과제입니다.
6. API는 `/recommendations`로 top-k를 제공하지만, Spring 서버가 아직 1권(`/recommend`)만
   쓰므로 온라인 지표는 1권 기준입니다.
7. `production_hybrid`의 가중치(alpha/lambda)는 사전 고정값입니다. 어느 값이
   나은지는 라벨이 생긴 뒤 dev split에서 판단해야 합니다.
8. 중복 제외(`exclude_isbns`)는 평가 하네스에 아직 넣지 않았습니다(후보 풀이 달라져
   이전 결과와 직접 비교가 안 되므로). 다양성 실측도 제외 없이 잽니다.

## 9. 다음 실험 계획

1. 운영 DB에서 `scripts/export_labels.py`를 실행해 `eval/labels.json`을 만들고,
   라벨 생성 시점과 카탈로그를 고정한다.
2. 사용자 단위 **dev/test split**(예: 7:3)을 두고, dev에서만 alpha를 튜닝한 뒤
   test에서 한 번 평가한다.
3. 사용자 단위 부트스트랩으로 지표 **신뢰구간**을 계산해, 변형 간 차이가 노이즈인지
   확인한다.
4. 카탈로그가 수백 권 이상이 되면 `category_filter`의 Recall 상한 문제를
   "카테고리 soft filter(가중)"와 비교한다.
5. 오프라인에서 고른 변형을 온라인 A/B(추천 노출 → 대여 전환)로 검증한다.
