# 챗봇 RAG 모듈 구축 — 최종 결과 보고서 (test_1 → test_2 → test_3)

> 학교 데이터(13개 PDF · 969페이지)를 근거로 답하는 챗봇용 RAG 모듈을, 세 세대에 걸쳐 발전시킨
> 전 과정의 종합 보고서다. 세대별 상세는 각 폴더의 `final.md`, 세대 전환은 `CHANGES_from_*.md` 참조.

---

## 1. 한눈에 보는 3세대

| | **test_1** | **test_2** | **test_3 (rag3)** | test_3 실험(rag3x) |
|---|---|---|---|---|
| 정체성 | 쿼리타임 VLM 멀티모달 RAG | MinerU 사전파싱 텍스트 RAG | 청크색인+리랭커+검증/롤백 RAG | +생성·검증 LLM을 Gemini로 |
| 파싱 | PyMuPDF+pdfplumber(질문 시) | **MinerU pipeline**(ingest 전량) | MinerU pipeline **+ vlm-engine(도표)** | (동일) |
| 검색 | Chroma 4컬렉션+BM25 RRF | Chroma 2컬렉션+BM25 RRF | **flat(numpy+BM25)+bge 리랭커** | (로컬 동일) |
| 답변 LLM/VLM | gemma4:12b(요약·청킹·판독) | gemma4:12b(전사후답변) | gemma4:12b(text 우선) | **gemini-3.1-flash-lite** |
| 검증 | LLM verify | **없음** | **controller S1~S8(검증·롤백·CRAG)** | +문장인용 |
| 질문당 호출 | ~29 VLM | **2** | 2.97 | 2.06 |
| 지연/문항 | 868s(냉시작) | 20~37s | 63.4s | **6.2s** |
| page_hit@3 | — | 0.556 | **0.774** | 0.774 |
| vision 오독 | (구조적) | 86% | **0%** | 0% |
| 환각 / 무관거절 | — | ~0 / — | **0 / 100%** | 0 / 100% |

## 2. 공통 스택 (전 세대 공유)

- **LLM/VLM**: `gemma4:12b`(Ollama), 폴백 `gemma4:e4b`. **임베딩**: `embeddinggemma`(768d, task-prefix, Ollama).
- **희소검색**: rank-bm25(BM25Okapi) + **RRF(k=60)**. **형태소**: test_1 char_bigram → test_2/3 **kiwipiepy**.
- **코퍼스**: 13개 PDF / 969페이지(학내망·무선랜·MDM·이용/통합관제 매뉴얼 등), DCAT 카탈로그 Excel.
- **런타임**: Python 3.11 · conda `intern_chatbot` · RTX 5060 Ti 16GB · Windows. (16GB 제약으로 `gemma4:26b` 미사용.)
- **세대 고유 추가**: test_2/3 **MinerU 3.4.4**(pipeline; test_3은 도표에 vlm-engine 추가), test_3 **bge-reranker-v2-m3**
  (sentence-transformers CrossEncoder), test_3 실험 **Gemini `gemini-3.1-flash-lite`**(생성·검증만).

## 3. 세대별 파이프라인 (단계 · 모델 · 프로그램)

각 세대의 **단계별 사용 모델·프로그램 상세 표**는 다음에 있다:
- test_1: [test_1/final.md §2](test_1/final.md) — 카탈로그 게이트 → PyMuPDF 렌더 → VLM 요약·시각청킹·판독 → LLM 검증.
- test_2: [test_2/final.md §2](test_2/final.md) — MinerU 전량 사전파싱 → embeddinggemma 색인 → 결정론 검색·라우팅 → gemma4:12b 전사답변(≤2 호출).
- test_3: [test_3/final.md §2](test_3/final.md) — MinerU(+vlm-engine 도표 텍스트화) → flat 색인 → **bge 리랭커** → small-to-big → gemma4:12b 답변 → **controller 검증·롤백·CRAG**.

## 4. 핵심 발견 (세대를 관통하는 교훈)

1. **표 숫자는 모델 눈으로 읽히지 말고, 도구로 텍스트화하라.** test_1 VLM은 조밀표 숫자를 8회 중 4회
   오독/환각 → test_2가 MinerU 마크다운 전사로 8/8 정확·40배 속도. (test_1→2의 결정적 이유)
2. **비전이 텍스트화하는 건 '스캔 표'가 아니라 '벡터 도표'다.** test_3 Phase 3의 반전 발견 —
   MinerU vlm-engine으로 **도표 페이지를 텍스트화**해 text 경로로 답하니 vision 오독 86%→0%, 부수적으로
   검색 앵커가 생겨 page_hit@3 0.645→0.774.
3. **검증·롤백이 챗봇 신뢰성을 만든다.** test_2의 무검증 단방향을 test_3 controller(숫자 결정론 대조 +
   groundedness + CRAG 재검색 + 결정론 롤백)로 바꿔 **환각 0 · 무관거절 100% · 정직한 abstain** 확보.
4. **카탈로그의 가치는 '질문 시 남은 모델 비용'에 달려 있다.** 비용이 큰 test_1에선 카탈로그가 이득,
   전량 저렴 색인된 test_2/3에선 오히려 게이트가 손해(no_catalog가 doc_hit 우세) → test_3은 게이트 제거.
5. **속도의 95%는 로컬 LLM 호출 시간이다.** "전 문항 20초"는 로컬로는 구조적으로 불가 →
   Phase 6에서 생성·검증만 Gemini(Flash-lite)로 바꾸니 63s→6s, kw 0.528→0.719.

## 5. 최종 상태 · 판정

- **test_3(rag3)은 챗봇 모듈로 투입 가능한 수준**에 도달: page_hit@3 0.774 · doc_hit 1.0 · vision 오독 0% ·
  환각 0 · 무관거절 100% · 평균 2.97 호출. 유일 미달인 **근거존재-실패율(20.8%)** 은 파이프라인 로직이 아니라
  **래스터 표/카드의 파싱 천장**(로컬 스택 한계)에서 비롯된 것으로, 전부 정직한 abstain(환각 아님).
- **배포 준비(Phase 5)** 완료: `python -m rag3 add`(문서 증분추가), `Rag3Engine` 모듈 API, 근거이미지 export,
  FastAPI 웹데모. 인계 문서 `test_3/코드/README.md`.

## 6. 미결 사항 (사용자 결정 대기)

- **Gemini(Flash-lite) 대체 / 유지 / 병행**: Phase 6 A/B/C/D 비교(`test_3/결과물/probes_results/PHASE6_COMPARISON.md`)
  결과 **C(Flash-lite 기본) + 문서간 D(분해+문장인용) 라우팅**이 최강 후보이나, **채택 결정은 사용자 몫**.
  현 저장소는 rag3(로컬)를 기본으로 두고 rag3x(실험)를 병행 보존한 상태다.
- **종합형 평가셋 사람 검수**: `PHASE6_SYNTH_EVAL_DRAFT.md`(S01~S12) 자체검증 상태, 확정 시 재측정.

## 7. 문서 지도

| 문서 | 내용 |
|---|---|
| [test_1/final.md](test_1/final.md) | 1세대 파이프라인·결과·한계 |
| [test_2/final.md](test_2/final.md) · [CHANGES_from_test1.md](test_2/CHANGES_from_test1.md) | 2세대 + test_1→2 전환 |
| [test_3/final.md](test_3/final.md) · [CHANGES_from_test2.md](test_3/CHANGES_from_test2.md) | 3세대(+Gemini 실험) + test_2→3 전환 |
| test_2/결과물/통합_실험_보고서.md | test_1·2·timecost 통합(발표용) |
| test_3/결과물/rag3_참고_실험팩트정리.md | 5개 실험 정량 팩트 총정리(인용 포함) |
| test_3/결과물/probes_results/ | Phase 0~6 전 보고서 + 원시 JSON 지표 |
