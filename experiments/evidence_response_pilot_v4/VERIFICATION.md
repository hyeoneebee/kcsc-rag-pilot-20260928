# 검증 결과

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-28
- Verification Status: PASS — 실행·기계적 재현/채점 검증 범위만
- Version Label: evidence_response_pilot_v4

AI의 의미 판단·정답 완전성·전문가 검수까지 검증됐다는 뜻이 아니다.

## 검사

- [x] unique_questions_and_rows
- [x] balanced_four_types
- [x] frozen_input_hashes
- [x] executed_code_hashes
- [x] source_code_hashes
- [x] metric_matrix_complete
- [x] reused_records_exact_unchanged
- [x] all_288_question_action_pairs_repeat_top15_match
- [x] all_1728_repeat_cutoff_scores_independent_match
- [x] candidate_and_query_budgets
- [x] no_query_truncation
- [x] all_48_gold_original_passages_match
- [x] all_12_reference_code_links_exist
- [x] all_returned_ids_exist_in_corpus
- [x] aggregate_transition_and_latency_recomputed
- [x] reference_intervention_trace_not_just_rerank
- [x] no_new_model_calls_on_unchanged_queries

## 테스트 실행

- 신규 단위 테스트 7개 통과
- 저장소 테스트 57개 통과
- 기존 실행부 테스트 9개 통과
- 두 차례의 보고서 생성 코드 괄호 오류를 수정했다. 추론 실행·질문·Gold·순위에는 변경 없음.

## 해석 오류 점검 — 11/11

- Simpson: 전체와 네 유형을 병기; 일별 시간과 유형이 겹치므로 유형 간 속도 원인을 추정하지 않음
- Ecological: 문항 단위 결과이며 전체 건설 사용자로 일반화하지 않음
- Berkson: 목적 표집·기존 문항 노출 한계를 명시; 새 결과로 실패 문항을 골라 넣지 않음
- Collider: 전체48개 결과를 보존; 기본 실패별 표는 조건부 기술통계임
- Base rate: 유형 균형은 실제 질문 빈도가 아님; 회복/악화 분모를 분리
- Regression to mean: 동일 질문·같은 기본 결과의 paired 비교; 반복을 독립 표본으로 세지 않음
- Survivorship: 48개×6방법 모두 포함; 실패/무효 분리 실행도 누락하지 않음
- Look elsewhere: Top5/10/15와 모든6방법 전부 공개; p값/유의성 선별 없음
- Forking paths: 질문과 비교조건을 새 결과 전에 해시 고정; 이전 결과 노출과 후속 해석 표시
- Correlation causation: 작업 적용 결과와 근본 실패 원인을 구분; 인과 진단 성능을 주장하지 않음
- Reverse causality: 기본 결과 후 작업 적용 순서 명시; 사후 실패 태그는 모델 입력 아님

## 남은 한계

- Same-model AI authorship and verification
- Semantic alternatives not exhaustive
- Mixed-session component timings, not dedicated E2E latency
- Purposive48, no population inference
