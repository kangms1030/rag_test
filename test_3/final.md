# test_3 최종 보고서 — 청크 색인 + 리랭커 + 검증/롤백 RAG (rag3) · Gemini 실험 포크(rag3x)

> 세대 정체성: **3세대(챗봇 투입 대상)**. test_2의 "MinerU 사전파싱 → LLM 전사" 기조를 유지하되,
> **청크 색인 + 크로스인코더 리랭커 + controller 상태기계(검증·롤백·CRAG)** 를 얹어 **신뢰성**(환각 0·
> 무관거절 100%)과 **검색 정확도**(page_hit@3 0.774)를 확보했다. 여기에 **Phase 6 실험 포크 `rag3x`**
> 가 생성·검증 LLM만 **Gemini**로 치환해 속도(전 문항 <20s)를 실험했다.
> 코드: [코드/rag3](코드/rag3)(로컬 프로덕션) · [코드/rag3x](코드/rag3x)(Gemini 실험) · 변경 이력: [CHANGES_from_test2.md](CHANGES_from_test2.md)

---

## 1. 한 줄 요약

**rag3(로컬·프로덕션)**: MinerU 청크 색인 → 리랭커로 관련 페이지를 정밀 선별 → gemma4:12b가 답 → **controller가
숫자 대조·groundedness로 검증하고, 실패하면 롤백/재검색**. 그림 위주 페이지는 MinerU **vlm-engine으로
텍스트화**해 비전 오독을 원천 회피. **환각 0 · 무관거절 100% · vision 오독 0%**.
**rag3x(Phase 6 실험)**: 위 파이프라인에서 **생성·검증 LLM만 Gemini(`gemini-3.1-flash-lite`)로 교체**(임베딩·리랭커는
로컬 불변). 전 플래그 OFF면 rag3와 바이트 등가. 속도가 63s→6s로 급감했으나 **채택 여부는 사용자 결정 대기**.

## 2. RAG 파이프라인 — rag3 (단계별 사용 모델 · 사용 프로그램)

모델 백엔드 **Ollama**(`num_ctx=16384`, `keep_alive=30m`, temp 0, `num_predict=1536`, `retry_on_length`).
설정 원문: `코드/rag3/config.yaml`. 단계별 근거: `결과물/probes_results/PHASE0~3_*.md`.

| # | 단계 | 사용 모델 | 사용 프로그램 / 라이브러리 | 설명 |
|---|---|---|---|---|
| 1 | PDF 사전 파싱 (offline) | **MinerU `pipeline`** (레이아웃/표/OCR, 결정론, `lang=korean`, cuda→cpu) | `mineru` 3.4.4 · PyMuPDF(fitz) | test_2 파싱 캐시(`cache/parsed_v25`, 13문서 969p) 재사용. |
| 2 | **figure 페이지 재파싱** (offline, Phase 3) | **MinerU `vlm-engine`**(MinerU2.5-Pro-1.2B) | `vlm_reparse.py`(mermaid 평탄화·멱등 병합, `MINERU_API_MAX_CONCURRENT_REQUESTS=1`) | 벡터 도표 페이지 **117개 중 115개를 텍스트화** → pipeline 캐시 병합. **비전 오독 회피의 핵심.** |
| 3 | 청킹 | — | `chunking.py`(MinerU 블록 기반, 텍스트 ~700자, 표 분할) | 969p → ~2,417~2,562 청크(텍스트+표). |
| 4 | 임베딩·색인 | **embeddinggemma** | **`flat_index.py`(numpy 브루트포스 dense + rank-bm25)** · RRF | **Chroma는 대형컬렉션 크로스프로세스 로드 실패(B6)로 flat 인덱스로 대체**. `page_store.json`(페이지 KV). |
| 5 | 하이브리드 검색 | **embeddinggemma**(쿼리) | flat_index(dense+BM25 **kiwipiepy**) RRF(k=60) | 청크 후보 `retrieve_candidates=20`. |
| 6 | **리랭킹** | **`BAAI/bge-reranker-v2-m3`** 크로스인코더 (fp16 CUDA) | **sentence-transformers `CrossEncoder`**(`rerank.py`, max_len 2048) | 무관거절 `rerank_score_floor=0.1`(관련 min 0.688 vs 무관 max 0.013). |
| 7 | small-to-big·라우팅 | — (결정론 메타룰) | `retrieve.py` | 청크→페이지 승격 `final_pages=3`. `figure_area_ratio≥0.5`·`scanned&table`→vision, else text. |
| 8 | 답변 생성 | **`gemma4:12b`** (text) / **`gemma4:12b`**(vision) | Ollama chat/vision, "전사후답변" | text 경로 우선(도표는 2단계로 이미 텍스트화). |
| 9 | **검증 (S7)** | 숫자대조=결정론 · groundedness=`verify_model`(비우면 12b 겸용) | `verify.py` | 답변 숫자를 근거와 결정론 대조 + 근거성 판정 + `is_abstain`. |
| 10 | **롤백·CRAG (S8)** | CRAG 재작성=12b · 롤백=결정론 | `controller.py`·`judge.py` | 빈응답/미지원 시 결정론 롤백 3종(최대 1회), 게이트 거절 시 질의 재작성 1회 재검색. |
| (폴백) | 실패 시 | **gemma4:e4b** | 자동 재시도 · `retry_on_length`(콜드 whitespace 런어웨이 방어, 예산 2회) | `deadline_seconds=180`. |

**controller S1~S8 상태기계**가 위 6~10을 조율한다(검색→답변→검증→결정론 롤백/CRAG). 이 검증 계층이
test_2에 없던 **환각 0 · 정직한 abstain**을 만든다.

## 3. 실험 포크 rag3x (Phase 6) — 무엇이 다른가

`rag3x`는 **rag3를 무수정 import**하고 바뀌는 부분만 flag-gated로 포크한다(전 플래그 OFF ⇒ rag3와 등가,
`결과물/scripts/x_equivalence_check.py`로 증명).

| 축 | 내용 | 파일 |
|---|---|---|
| Gemini 백엔드 | **생성·검증 LLM만 `gemini-3.1-flash-lite`**로 치환. **임베딩·리랭커·검색은 로컬 불변.** 20 RPM 스로틀(3.2s), 장애 시 로컬 12b 폴백, 토큰/비용 계측 | `gemini_backend.py` |
| P1 로컬 속도 | verify-skip·adaptive-trim·length-retry 예산 축소 | `controller_x.py`·`answer_x.py` |
| P3 다문서 종합 | 분해검색 + 합성 + **문장단위 인용 검증** | `decompose.py` |
| 등가 계약 | `X_DEFAULTS` 플래그, 기본 backend=ollama | `xconfig.py`·`backends.py`·`engine_x.py` |

> **Gemini 모델명 경위(각주)**: 지시 프롬프트(`결과물/RAG3_PHASE6_고도화_프롬프트.md`)는 `gemini-2.5-flash`를
> 명시했으나, 신규 API 키에서 2.5-flash는 404("no longer available to new users"), 2.0-flash는 429였다.
> 사용자가 **최저가·고한도 lite 티어인 `gemini-3.1-flash-lite`**를 선택했고, 실측(`PHASE6_COMPARISON.md`)도
> 이 모델로 수행했다. (색인 폴더에 남은 `gemini-gemini-3.5-flash`는 과거 런의 잔여 아티팩트일 뿐 사용 모델 아님.)

## 4. 사용 스택 총정리

| 구분 | rag3 (로컬 프로덕션) | rag3x (실험) |
|---|---|---|
| 파서 | MinerU pipeline **+ vlm-engine(figure)** + PyMuPDF | (동일) |
| 임베딩 | `embeddinggemma` | **로컬 동일(치환 안 함)** |
| 색인 | flat_index(numpy+BM25, RRF) + page_store | (동일) |
| 리랭커 | `BAAI/bge-reranker-v2-m3`(sentence-transformers CrossEncoder) | **로컬 동일** |
| 생성·검증 LLM | `gemma4:12b` (text/vision) | **`gemini-3.1-flash-lite`**(flag) |
| 폴백 | `gemma4:e4b` | 로컬 12b 폴백 |
| 오케스트레이션 | controller S1~S8(검증·롤백·CRAG) | controller_x(+분해·문장인용) |

## 5. 주요 결과

### 5.1 rag3 최종(36문항, `결과물/probes_results/FINAL_REPORT.md` · `varfix_eval.json`)

| 지표 | test_2 | **test_3 rag3** | 목표 | 판정 |
|---|---|---|---|---|
| page_hit@3 | 0.556 | **0.774** | ≥0.75 | ✅ |
| doc_hit@3 | 0.923 | **1.0** | ≥0.92 | ✅ |
| vision 오독률 | 86% | **0%** | ≤30% | ✅ 초과 |
| 무관 질문 거절 | 100% | **100%**(5/5) | 100% | ✅ |
| **환각(무근거 생성)** | ~0 | **0** | 0 | ✅ |
| avg_kw_hit(관련) | ~0.42 | **0.528** | — | ↑ |
| 근거존재-실패율 | 33.3% | **20.8%** | 0% | 미달(파싱천장) |
| avg 모델호출 | ~2 | **2.97** | ≤3.5 | ✅ |
| 지연(평균/최대) | 35~49s | **63.4s / 148.6s** | ≤180s | ✅ |

남은 실패는 **파이프라인 로직이 아니라 래스터 표/카드 파싱 천장**(vp_006/007/013, core_002/008)과
일부 검색 미스로 좁혀졌고, 전부 **정직한 abstain**(환각 0).

### 5.2 Phase 6 A/B/C/D(동일 36문항, `PHASE6_COMPARISON.md`)

| 축 | A 기존(12b) | B 로컬개선 | **C Flash-lite** |
|---|---|---|---|
| avg_kw_hit | 0.528 | 0.593 | **0.719** |
| 평균/최대 지연 | 63.4 / 148.6s | 55.8 / 181.0s | **6.2 / 13.0s** |
| 전 문항 20초 이내 | 아니오 | 아니오 | **예(36/36)** |
| 파싱천장 5문항 개방 | 0/5 | 2/5 | **4/5** |
| 질문당 비용 | ₩0 | ₩0 | ~$0.0004 |
| 환각 / 무관거절 | 0 / 100% | 0 / 100% | 0 / 100% |

- 종합형 12문항: A 0.882 · C 0.921 · **D(Flash+분해+문장인용) 0.916**, 특히 **문서간 2문항 0.625→0.917**.
- 관찰: 속도·정확도·비용 전 축에서 **C가 A/B를 지배**. 로컬 개선(B) 이득은 미미하고 꼬리(최대 지연) 미해결.
- ⛔ **결정(대체/유지/병행)은 사용자 몫**. 종합형 평가셋 정답은 자체검증 상태(사용자 검수 대기).

## 6. 코드 구성

- **`코드/rag3`**: `engine.py`(`Rag3Engine.ask`) · `controller.py`(S1~S8) · `retrieve.py` · `rerank.py`(bge) ·
  `flat_index.py` · `parse_mineru.py`·`vlm_reparse.py`·`chunking.py` · `answer.py`·`verify.py`·`judge.py` ·
  `models.py`(Ollama) · `add_doc.py`(문서 증분추가) · `evidence.py` · `config.yaml`.
- **`코드/rag3x`**: `gemini_backend.py`·`xconfig.py`·`backends.py`·`controller_x.py`·`answer_x.py`·`decompose.py`·`engine_x.py`.
- **`코드/webapp`**: FastAPI 데모(`server.py`+`static/index.html`) — 답변+근거이미지+파이프라인 트레이스+벤치마크.
- **`코드/probes`**: Phase 0 진단 프로브(`probe_*.py`)·평가 하버스트(`eval_phase*.py`).
- **`코드/ask_cli.py`**(로컬 REPL) · **`ask_cli_gemini.py`**(Gemini) · `requirements.txt` · `README.md`(모듈 API·인계 문서).

## 7. 산출물 위치

- **결과물**: `결과물/probes_results`(PHASE0~6 전 보고서 + JSON 지표: `FINAL_REPORT.md`·`PHASE6_COMPARISON.md`·`varfix_eval.json` 등) · `결과물/rag3_outputs`(ingest_summary·evidence) · `결과물/RAG3_설계계획.md`·`CONFIG_파라미터_가이드.md`·`RAG3_GEMINI_외부API_보고서.md` · `결과물/rag3_참고_실험팩트정리.md`·`RAG3_PHASE6_고도화_프롬프트.md` · `결과물/scripts`(재사용 eval/verify 스크립트)
- **사전데이터**: `사전데이터/데이터 카탈로그 작업 파일`(13 PDF, test_2와 동일 corpus 사본) · 카탈로그 xlsx · `synth_eval_dataset.json`(종합형 12Q) · `corpus_doc_*.txt`(전문 덤프)
  ※ 파싱 캐시(`cache/parsed_v25`)·색인(`index/`)은 `코드/rag3` 내부에 실행상태로 보존.

## 8. 재현 · 주의

```
cmd /c conda activate intern_chatbot && python -m rag3 evaluate      # 단일패스(검증X) — FINAL 수치와 직접비교 금지
cmd /c conda activate intern_chatbot && python 코드\ask_cli.py         # controller 기반 대화형(웜)
cmd /c conda activate intern_chatbot && python 코드\webapp\server.py --port 8000   # 웹데모
```

> ⚠️ **경로 주의**: `rag3/config.yaml`의 `documents_dir`는 `../../rag_test/test_2/…`를 가리켰다. 정리로
> `rag_test`가 제거되면 이 경로는 깨지나, **쿼리·웹데모는 `코드/rag3` 내부의 `cache/parsed_v25`+`index`로
> 자립 동작**한다(재-ingest/문서추가 시에만 경로 조정 필요). GPU 2중 프로세스 금지(평가·데모는 단독 실행).
