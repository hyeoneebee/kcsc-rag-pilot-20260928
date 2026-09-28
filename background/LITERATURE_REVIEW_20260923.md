# 문헌 검토 기록 — 도구 학습 vs 대응 선택 학습

검색일: 2026-09-23. 상태: 목적형 문헌 검토, 체계적 문헌고찰이나 전체 문헌의 망라가 아님.

## 검색·선별 범위

- 범위: 2022–2026년의 검색 모델 도메인 적응, 적응형 RAG, 비용·지연 평가, 규정/건설 기준 RAG.
- 검색어 묶음: `adaptive retrieval augmented generation cost latency learned routing`, `MBA-RAG bandit`, `rule based learned policy RAG efficiency`, `GPL domain adaptation dense retrieval`, `reranker domain adaptation synthetic queries`, `AI Agents That Matter`, `building regulations compliance reference retrieval`, 이후 정확한 논문 제목 검색.
- 포함: 공식 학회 저장소·출판사·저자 arXiv 원문에서 방법이나 실험 설계와의 관련성을 확인한 자료.
- 제외: 검색 결과 요약만으로 성능 수치를 확정하는 자료, 블로그·뉴스·재인용만 있는 주장, 현재 연구 질문과 거리가 먼 응용 사례.
- 읽기 범위를 아래에 구분했다. 모든 논문의 전 페이지를 읽었다는 의미가 아니다. 아래의 ‘시사점’은 이 프로젝트에 대한 설계 판단이며, 해당 논문이 KCSC에서 효과를 입증했다는 뜻이 아니다.

## 주요 자료 12건

| 문헌 / 발표 상태 | 확인 범위 | 핵심 내용과 이 프로젝트에서의 사용 |
|---|---|---|
| [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://aclanthology.org/2022.naacl-main.168/) — Wang 외, NAACL 2022 | 공식 서지·초록, PDF 도입·방법 일부 | 합성 질문과 cross-encoder 학습 신호로 Dense 적응. 도구 자체 학습의 대표 비교. KCSC 효과 크기는 미확인. |
| [UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers](https://aclanthology.org/2023.emnlp-main.693/) — Saad-Falcon 외, EMNLP 2023 | 공식 서지·초록; PDF 열람, 상세 재현 검토 전 | 재정렬기 학습 후 효율적 검색기로 증류. 학습이 실행 비용을 낮출 수도 있지만 자료 생성·학습 비용과 증류 효과를 분리해야 함. |
| [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://aclanthology.org/2024.naacl-long.389/) — Jeong 외, NAACL 2024 | 공식 서지·초록, PDF 도입·방법 일부 | 작은 분류기로 검색 전략 선택. 논문의 무검색 답변을 근거 필수인 기준 QA에 그대로 적용하지 않음. 여기서는 최초 검색 후 추가 대응 생략을 뜻함. |
| [MBA-RAG: a Bandit Approach for Adaptive Retrieval-Augmented Generation through Question Complexity](https://aclanthology.org/2025.coling-main.218/) — Tang 외, COLING 2025 | 공식 서지·초록, PDF §2.2–3.3 | 성공 여부뿐 아니라 검색 단계 비용을 반영하는 선택 학습. 여러 유효한 대응을 인정할 근거. 단계 수를 실제 지연·금전 비용으로 치환하지 않음. |
| [Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home](https://aclanthology.org/2025.acl-long.319/) — Moskvoretskii 외, ACL 2025 | 공식 서지·초록, PDF 도입·평가 개요 | 다양한 적응형·불확실성 기반 방법 비교에서 단순 방법도 경쟁력. 복잡한 Agent만 비교하면 안 됨. LLM 불확실성과 Dense 점수 차이는 같은 지표가 아님. |
| [AI Agents That Matter](https://arxiv.org/abs/2407.01502) — Kapoor 외, arXiv 2024 | 공식 메타데이터·초록, HTML 비용-정확도·공동 최적화·준비 비용 관련 부분 | 정확도만으로 Agent를 비교하면 비용이 가려짐. 준비 비용과 추론 비용, 단순한 기준선, 독립 평가를 함께 다룰 근거. 이 검토에서는 확인한 arXiv 판본으로 표기. |
| [SPARKLE: A Structured and Plug-and-play Agentic Retrieval Policy for Adaptive RAG Models](https://aclanthology.org/2026.acl-long.1793/) — Fang 외, ACL 2026 | 공식 서지·초록, PDF 도입·방법 일부 | 기반 모델과 검색기를 고정한 채 별도 정책을 학습하는 예. 정책 학습과 도구 학습 분리가 가능함. 전체 RL 구조를 첫 소규모 Pilot의 기본값으로 삼지 않음. |
| [SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems](https://arxiv.org/abs/2608.08237) — arXiv 2026, preprint | 공식 메타데이터·초록, HTML 방법·실험 관련 절 | 가벼운 특징으로 검색량을 고르는 접근의 참고 후보. 본문의 Random Forest 설명과 Adam 학습 설명 등 설정 연결을 추가 확인해야 하므로 재현 설계나 성능 근거로 채택하지 않음. |
| [Quantifying the Accuracy and Cost Impact of Design Decisions in Budget-Constrained Agentic LLM Search](https://aclanthology.org/2026.lrec-1.808/) — McCleary·Ghawaly, LREC 2026 | 공식 서지·초록, PDF §3.8–3.10 및 §6–7 | 검색·토큰 예산의 효과와 한계. 비Agent retrieve-read 기준선 부재가 명시되어 있어, 이 논문만으로 Agent의 우위를 주장하지 않음. |
| [Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems](https://aclanthology.org/2025.coling-industry.4/) — de Lima 외, COLING Industry 2025 | 공식 서지·초록 | 데이터의 요구 능력과 생성 방식이 평가를 좌우함. 실패 사례만 모으거나 합성 질문의 비율을 실제 사용 분포로 간주하지 않을 근거. |
| [RIRAG: Regulatory Information Retrieval and Answer Generation](https://arxiv.org/abs/2409.05677) — Gokhan 외, arXiv 2024 | 공식 메타데이터·초록 | 금융 규정 QA와 의무사항 포함·모순 평가. 근거 한 개 적중보다 요구사항 전체 충족을 봐야 한다는 도메인 참고. 건설·한국어용 평가로 검증된 것은 아님. |
| [Retrieval-Augmented Generation and Multi-Agent Arbitration for Construction Code Compliance in AEC](https://www.iaarc.org/publications/2026_proceedings_of_the_43rd_isarc_singapore/retrieval_augmented_generation_and_multi_agent_arbitration_for_construction_code_compliance_in_aec.html) — Yin 외, ISARC 2026 | 공식 발표 페이지·초록; PDF 전문 미검토 | 조항 중심 검색과 선택적 중재를 사용하는 건설 기준 연구. ‘건설+Agent’만으로 신규성을 주장할 수 없음. 중재 후 판단 변경률은 정확도 개선률과 다름. |

## 접근 제한 후보

[Building regulation question-answering system using retrieval-augmented generation with dual-stage fine-tuned large language model](https://www.sciencedirect.com/science/article/pii/S1474034625009826), DOI `10.1016/j.aei.2025.104089`: 출판사 페이지 접근이 제한되어 상세 방법·결과를 이 검토의 근거로 사용하지 않았다. 관련 후보로만 유지한다.

## 종합 판단

1. 도구를 개선하는 학습과 도구를 선택하는 학습은 이미 서로 다른 연구 계열로 존재한다. 둘 중 하나가 항상 저렴하거나 정확하다는 결론은 낼 수 없다.
2. 이번 과제는 동일 도구 아래 실행 정책을 비교하면 효과를 분리하기 쉽다. 다만 강한 도구 추가 학습 기준선을 끝까지 배제하면 ‘선택이 아니라 검색기 개선이면 충분하지 않은가?’라는 반론을 남긴다.
3. 선행 연구의 검색 횟수 절감은 실제 시간·비용 절감의 대체 지표일 수 있다. 한국어 기준 코퍼스와 현재 장비에서 직접 측정해야 한다.
4. 공개 QA나 다른 규정 도메인의 결과가 KCSC로 그대로 옮겨온다는 증거는 아직 없다. 한국어·참조 관계·적용 범위의 검증은 필요하지만, 이 사실만으로 연구 신규성이 확정되지는 않는다.
5. 권장 순서: 대응 효과 측정 → 단순 고정·규칙 기준 → 필요한 경우 학습 선택 → 실패 원인이 도구에 있으면 도구 추가 학습 → 남는 순차 의사결정 문제에 한해 다단계 Agent.

이 문서는 9/23의 문헌 검토 기록이다. 이번 실험 조건은 ../docs/EXPERIMENT_CONDITIONS.md를 따른다.
