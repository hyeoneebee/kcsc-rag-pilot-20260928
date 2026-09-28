# 실제 적용 실험 조건

## 먼저 구분할 것

상속한 `retrieval_config_v1.json`과 실행 manifest의 `source_config`/`effective_config`에는 옛 평가 설정 일부가 남아 있다. 이름이 `effective_config`여도 실행 코드의 상수를 모두 반영한 것은 아니다. 실제 설정은 실행 코드와 `source_runtime_overrides`, v4 `PROTOCOL.md`를 대조했다. 아래 및 [effective_settings.json](../provenance/effective_settings.json)은 그 대조 요약이다.

| 항목 | 실제 v4 값 | 혼동하기 쉬운 과거 값 |
|---|---|---|
| 평가 질문 | 전체 코퍼스 48개 | 46개+대조 2개는 초기 실험 |
| 평가 cutoff | 5, 10, 15 | 상속 설정의 5, 10, 20 |
| 주 지표 | Complete Evidence Success@10 | 이전 설정의 required evidence recall |
| 실행 seed | 20260927 | 상속 설정의 20260909 |
| 재정렬 후보·저장 순위 | 입력 최대 50개, 출력 50개 | 상속 설정의 출력 20개 |
| Dense 질의 batch / 재정렬 batch | 1 / 4 | 초기 자원 probe 값과 구분 |

## 모델과 검색

| 항목 | 조건 |
|---|---|
| Dense 모델 | `BAAI/bge-m3` |
| Dense revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| 임베딩 | dense-only, CLS pooling, L2 정규화, 1,024차원 |
| 검색 | 전체 절 exact inner product, 후보 100개, 근사 검색 아님 |
| 입력 길이 | 질의 128토큰 / 절 1,024토큰, 메타데이터 포함 |
| 정밀도 | 모델 float16, 임베딩 저장·내적 float32, TF32 비활성 |
| 재정렬 모델 | `BAAI/bge-reranker-v2-m3` |
| 재정렬 revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| 재정렬 입력 | 질문+절 최대 1,536토큰, 본문 측만 truncation, batch 4, 최대 50쌍 |
| 재정렬 점수 | float16 추론 raw logit, 내림차순; 동점은 section_id 오름차순 |
| 문자 검색 | SQLite FTS5 trigram BM25, top100 |
| FTS 질의 | NFKC·코드 표현 정규화, 고정 불용어/조사 처리, 3글자 이상 용어와 짧은 인접 용어 결합, OR |
| FTS 열 가중치 | 코드 8, 문서명 5, 제목 3, label 2, 본문 1 |
| 결합 | FTS100 + Dense100 → RRF(k0=60) → 상위 50 재정렬 |

검색은 질문 문자열만 입력받으며 Gold·질문 유형·참조 정답 그래프는 도구 입력이 아니다. FTS에 Gold에서 뽑은 용어나 수동 동의어를 넣지 않는다. 구현은 `scripts/retrieval_config.py`와 `run_baselines.py`, v3 `run_local.py`를 원 해시 그대로 포함했다.

## 여섯 고정 작업

| 이름 | 실제 동작 |
|---|---|
| STOP | Dense100 순서 그대로 |
| RERANK | Dense 상위 50개 재정렬 |
| REFERENCE | Dense 상위 5개 본문의 명시적 코드+절 링크를 1단계 보충. 소스당 최대 3개 코드 일치, 일치당 최대 20절. 기본/추가 후보를 번갈아 병합해 50개로 제한 후 같은 재정렬 |
| SPLIT | 접속표현 규칙으로 최대 2개 추가 질의. 각 Dense100과 원질의 결과를 번갈아 병합해 50개 후 같은 재정렬. 분리되지 않으면 RERANK와 동일 |
| HYBRID_RERANK | 위 FTS+Dense 결합 후보 50개를 같은 모델로 재정렬 |
| KEEP3 | Dense 상위 3개를 앞으로 두고 나머지를 RERANK 순서로 중복 없이 이어붙임. 동일한 50개 후보에서 결정적 후처리 |

모든 질문에 모든 작업을 적용한다. 유형별로 한 작업만 실행한 실험은 아니다. SCOPE와 선택 정책은 제외했다. KEEP3은 별도 학습도 새 모델 추론도 아니다.

## 실행·재사용·환경

- 동일 질의·Gold·코드 해시·코퍼스·모델 revision·설정·주요 실행환경을 확인해 36문항×5작업×2회 = **360개** 기존 추론 기록을 재사용했다.
- 수정 참조 12문항×5작업×2회 = **120개**를 새로 추론했다.
- KEEP3 48문항×2회 = **96개** 후처리 기록을 만들었다. 총 **576행**이다.
- 현재 결과에는 재사용분의 미사용 선택용 특징 계산 시간을 모든 방법에서 공통으로 제외했다.
- GPU: NVIDIA GeForce RTX 3090, driver 595.84. v4 기록: torch `2.13.0+cu130`, numpy `2.5.2`.
- 초기 preflight에는 Python 3.12.3, SQLite 3.45.1, transformers 5.15.1 등이 기록돼 있다. v4가 모든 패키지의 완전한 lock을 저장한 것은 아니다. 현재 환경 관찰과 당시 실행 기록을 혼동하지 않는다.
- 평가 외 질문으로 warmup 후 진행. 새 질문의 순서와 작업 순서를 고정 seed로 섞었다. 질의 길이 최대 65토큰으로 128토큰 한도 미만이었다.
- 기존 source Git HEAD: `e2607b9a256543af4f90cc48fe0bc065af176e12`. 실험 폴더는 당시 미추적 파일이므로 이 commit만으로 실험 코드 전체를 식별할 수 없다. 파일별 SHA-256을 함께 사용한다.

## 지표와 시간 해석

- 완전 근거 확보율 (Complete Evidence Success@k): 모든 필수 근거 그룹에서 하나 이상 확보한 질문 비율.
- 하나 이상 적중 (Hit Rate@k): 필수 근거 중 하나 이상 확보한 질문 비율.
- 근거 확보 비율 (evidence-group Recall@k): 질문별 확보한 그룹 수/전체 그룹 수를 문항 간 평균.
- 회복률: 비교 기준의 실패 중 성공 전환 비율. 악화율: 비교 기준의 성공 중 실패 전환 비율. 분모를 따로 보고한다.
- STOP과 RERANK 모두에 대해 짝지은 비교를 한다. 재정렬 추가 효과와 보충 작업의 추가 효과를 구분하기 위해서다.
- 시간: 질의 임베딩+Dense+해당 작업의 구간 측정 합. 각 질문의 2회 평균을 먼저 만든 뒤 문항 간 평균/P50/P95를 계산한다.
- 모델 로딩·색인 구축·캐시 읽기·채점·질문 작성은 질문당 시간에서 제외. 서로 다른 세션 결과가 섞여 있으며 전체 요청 E2E 측정이 아니다.
- 외부 API 호출, 추가 Dense/FTS 호출, 재정렬 쌍 수는 보고한다. GPU 전용 시간·전력·원화 비용은 측정하지 않았다.

조건·질문을 고정한 기록은 있으나 공인 사전등록 플랫폼의 등록 연구라고 표현하지 않는다. 점수에 맞춰 이번 결과를 숨기거나 다른 cutoff로 교체하지 않는다.
