# 파일 안내와 검증 범위

## 논문 작성용 파일 지도

| 필요한 내용 | 파일 |
|---|---|
| 연구 배경·문헌 연결·논문 구성 | `docs/PAPER_BRIEF.md`, `background/LITERATURE_REVIEW_20260923.md` |
| 전처리·QA 구성·한계 | `docs/DATASET.md`, `provenance/corpus_build_report.json` |
| 현재 질문·답변 초안·Gold 그룹 | `experiments/evidence_response_pilot_v4/questions.jsonl`, `docs/QUESTION_INDEX.md` |
| 질문 변경 이유 | 같은 폴더 `question_changes.json`, `build_dataset.py` |
| 실제 실험 조건 | `docs/EXPERIMENT_CONDITIONS.md`, `provenance/effective_settings.json` |
| 동결·실행·분석 manifest | v4 `dataset_manifest.json`, `PROTOCOL.md`, `run_20260928/manifest.json`, `analysis_manifest.json` |
| 전체 결과·해석 | v4 `RESULTS.md`, `DISCUSSION.md`, `NOTION_REPORT.md` |
| 원 실행 순위·시간·보충 trace | v4 `run_20260928/actions.jsonl` (576행) |
| 질문×방법×cutoff 결과 | v4 `run_20260928/question_action_metrics.csv` (864행) |
| 전체/유형/실행일별 집계 | v4 `run_20260928/summary.json` |
| 문항별 변화·근거 순위 | v4 `run_20260928/cases.json` |
| 이전 전체 환경 검증 결과 | v4 `VERIFICATION.md`, `run_20260928/verification.json` |
| 이번 공개본 검증 결과 | `provenance/HANDOFF_VERIFICATION.json` |
| 파일 출처·공개용 수정 | `provenance/export_manifest.json`, `SHA256SUMS` |

## 1단계: 공개본만으로 결과 재채점

저장소 루트에서 Python 3.10 이상으로 실행한다. 추가 패키지 설치, GPU, 인터넷, DB가 필요 없다.

```bash
python3 tools/verify_handoff.py
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s experiments/evidence_response_pilot_v4 -p 'test_*.py' -v
```

공유용 검증 코드는 실험의 `core.py`를 가져오지 않고 직접 채점한다. 파일 해시, 48문항·유형 균형, 576개 실행 기록, 재사용 360행의 원 기록 일치, 288개 질문×방법의 반복 순위 일치, 1,728개 반복×cutoff 점수, 864행 CSV, 전체/유형/실행일별 집계, 시간·회복·악화, KEEP3 변환과 주요 사례를 확인한다. 모델 재추론은 하지 않는다.

질문 원문은 공개본에도 있지만 **기준 원문 본문은 없으므로**, DB 속 원문 일치·모든 반환 절의 실제 존재·참조 문자열을 새로 검증할 수는 없다. 기존 전체 환경의 검증 기록과 구분한다. 전문가 검수도 아니다.

`SHA256SUMS`는 배포 파일의 변경 탐지용이다. 서명된 제3자 증명은 아니다. 자기 자신, `.git`, Python 캐시와 파생 검증 보고서는 목록에서 제외한다. 검증 스크립트 자체는 목록에 포함한다.

## 2단계: 원문까지 재검증하거나 새 추론

이 저장소만으로는 불가능하다. 필요한 외부 자산은 다음과 같다.

- 해시가 같은 `data/processed/2026-08-12-full/kcsc.sqlite3`
- Dense 캐시 `experiments/retrieval_failure_discovery_v1/runs/cache/7e2408a65f4cdef444db/passages.f32.npy` (약 2.28GB)와 해당 state
- 고정 revision의 두 모델 및 tokenizer 로컬 캐시
- 원문 본문을 제거하기 전의 v4와 v3 질문 파일, 원본 manifest. 공유본의 변경 이력으로 원 파일인지 대조
- 실행 코드가 가정하는 CUDA/GPU/torch/numpy 및 나머지 라이브러리 환경

DB 해시·임베딩 해시는 포함된 manifest에 있다. 원 자산을 복구한 **별도 작업 사본**에서만 원 실행 코드의 `--check-only`를 먼저 실행한다. 결과 폴더를 덮어쓰지 말고 `--out`에 새 이름을 준다. 원 `analyze.py`와 `verify.py`는 DB와 원 경로를 필요로 하며 일부 보고서를 쓰므로, 공개본의 간단한 검증 명령으로 사용하지 않는다.

현재 환경에 대한 임의의 `pip freeze`를 당시 완전한 환경 lock이라고 제시하지 않는다. 저장된 runtime 정보가 제한적이라는 점을 함께 적는다.

## 과거 폴더를 넣은 이유

`response_selection_pilot_v3`는 v4가 실제로 import한 `Engine`과 기존 추론 360행의 출처다. 이전 질문 v1/v2와 실행 기록을 동봉해 질문·Gold·순위·시간이 유지됐는지 확인할 수 있게 했다. **현재 논문 결과로 집계하지 않는다.**

`retrieval_failure_discovery_v1`은 FTS 질의 규칙·Dense 코드·입력 조립·RRF 설정과 동결 코퍼스/임베딩 출처다. 이전 manifest의 46개 분모, 옛 cutoff, SCOPE 또는 학습 정책은 v4 실험 조건이 아니다. 원본 스냅샷 코드를 수정해 옛 설정 흔적을 감추지 않았다.

## 기존 검증과 새 검증을 구분

원 작업환경에서 73개 테스트와 17개 기계적 점검이 통과했다는 기록을 포함했다. 이번 공개본에서는 **공유본 채점 검증과 포함한 단위 테스트를 별도로 다시 수행**한다. 원 저장소의 전체 73개 테스트를 이 공개본만으로 다시 실행했다고 쓰지 않는다.
