# test_2 최종 보고서 — MinerU 사전파싱 하이브리드 텍스트 RAG

> 세대 정체성: **2세대**. test_1의 "질문 시 VLM이 페이지를 본다"를 버리고, **ingest 시점에 MinerU로
> 969페이지를 전량 사전 파싱**(표를 마크다운으로 결정론적 추출)해 두고, 질문 시엔 **텍스트를 검색해
> LLM이 전사·답변**한다. 질문당 모델 호출을 최대 2회로 줄여 test_1 대비 ~40배 빠르고 수치 정확도가 높다.
> 코드: [코드/rag2](코드/rag2) · 변경 이력: [CHANGES_from_test1.md](CHANGES_from_test1.md)

---

## 1. 한 줄 요약

MinerU가 표를 **텍스트(마크다운)로 미리 뽑아** 두므로, LLM은 "이미 추출된 표를 그대로 전사"만 하면 된다
→ VLM 오독이 사라지고(수치 8/8 정확), 질문당 임베딩 1 + 답변 1 = **2회 호출**로 20~37초에 응답. 단
**검증 단계가 없고**(단방향), **비전 경로가 뜨는 스캔표·도표에서는 여전히 오독**하는 약점이 남는다.

## 2. RAG 파이프라인 (단계별 사용 모델 · 사용 프로그램)

모델 백엔드는 **Ollama**(`num_ctx=8192`, `keep_alive=10m`). 설정 원문: `코드/rag2/config.yaml`.
설계 원칙(코드 docstring): **무거운 작업은 전부 ingest에서 선처리, 질문 시엔 최대 2회 호출.**

| # | 단계 | 사용 모델 | 사용 프로그램 / 라이브러리 | 설명 |
|---|---|---|---|---|
| 1 | 카탈로그 적재·문서 색인 (offline ingest) | **embeddinggemma** | pandas · openpyxl · rapidfuzz · **chromadb** | DCAT Excel 행 ↔ 13 PDF 매칭, `catalog_index`·`page_index`(cosine) 구축. |
| 2 | **PDF 사전 파싱** (offline ingest, 1회) | **MinerU 3.4.4 `pipeline` 백엔드** (레이아웃+표+OCR 전용 모델, **LLM 아님**, 결정론적) | `mineru`(do_parse, `lang=korean`, `device=cuda`→cpu 폴백) · **PyMuPDF(fitz)** 페이지 PNG 렌더·스캔판정(native text<50자) · **pdfplumber**(폴백 파서) | 969페이지 전량 파싱. **표=마크다운+크롭이미지**, figure/caption/`figure_area_ratio`/`has_table`/`page_type` 사전 계산. **VLM 미호출.** |
| 3 | 쿼리 임베딩 | **embeddinggemma** | Ollama embed(task-prefix) | 질문 1회 임베딩. |
| 4 | 하이브리드 검색·라우팅 (질문 시, **모델 없음·결정론**) | — | **chromadb**(catalog/page) · **rank-bm25**(BM25Okapi, **kiwipiepy** 토큰화) · **dense cosine** · **RRF(k=60)** | catalog 게이트(`min_dense_similarity=0.35`) → `top_docs≤2` → page 검색 `top_pages=3`. 라우팅: `figure_area_ratio≥0.5`→vision, `is_scanned & has_table`→vision, else **text**. |
| 5 | 답변 생성 (질문당 **1회** 호출) | **텍스트 경로 `gemma4:12b`** / **비전 경로 `gemma4:12b`(VLM)** | Ollama chat / vision · **Pillow**(비전 시 표/도표 크롭 리사이즈 200dpi·2048px) | 텍스트: 페이지 텍스트+마크다운 표를 "**셀을 먼저 전사한 뒤 답하라**" 프롬프트. 비전: `selected_pages[0]` 이미지 1장. |
| (폴백) | 실패 시 | **gemma4:e4b** | 자동 1회 재시도 | — |

> **검증 단계 없음**: test_1의 query_analyzer·VLM 요약·시각청킹·verify를 **전부 제거**했다(속도·단순성).
> 이 "검증 부재"가 test_3에서 다시 문제가 되어 controller 검증/롤백이 도입된다.

## 3. 사용 스택 총정리

| 구분 | 사용 | 비고 |
|---|---|---|
| 파서 | **MinerU 3.4.4 pipeline**(레이아웃/표/OCR) + PyMuPDF + pdfplumber | 표→마크다운 결정론 추출 |
| LLM(텍스트 답변) | `gemma4:12b` | Ollama |
| VLM(비전 답변) | `gemma4:12b` (vision) | 라우팅 시에만 |
| 폴백 | `gemma4:e4b` | |
| 임베딩 | `embeddinggemma` | Ollama |
| 벡터DB | Chroma(2컬렉션: catalog/page) | cosine |
| 희소검색 | rank-bm25 + RRF(k=60), **kiwipiepy** 토큰화 | test_1 char_bigram→kiwi 전환 |
| 이미지 | Pillow(200 DPI, 2048px) | |

## 4. 주요 결과

- **수치 정확도**: 신규 3문항 8회 수치 관측 **8/8 정확**(test_1은 4회 오류). 평균 keyword_recall **0.736**
  (test_1 0.646). (`결과물/test12_비교실험/FINAL_REPORT.md`)
- **속도**: 문항당 20.5s(catalog)~37.3s(no_catalog). **test_1 대비 39.7~43.5배 빠름**. 질문당 정확히
  **2회 호출**(embed 1 + text answer 1), 비전 경로 미발생.
- **카탈로그 효과 반전**: test_1과 달리 **no_catalog가 doc_hit에서 우세**(0.923 vs 0.692) — 모든 페이지가
  이미 저렴하게 색인돼 있어 카탈로그 게이트는 조기 거절·형제문서 혼동만 유발.
- **비전 경로 약점**: `test2_vlm_probe`(20Q×2모드, `결과물/test2_vlm_probe`) — 강제로 비전을 태운 경우
  정답 페이지를 실제로 본 7회 중 **6회(86%) 오독**. 즉 test_2의 정확도는 "**비전을 회피**"해서 얻은 것이며,
  스캔표/도표가 비전으로 라우팅되면 test_1의 VLM 오독을 그대로 물려받는다(→ test_3 Phase 3의 표적).

## 5. 코드 구성 (`코드/rag2`, 21개 모듈)

`__main__.py`(CLI: check/ingest/ask/evaluate) · `catalog.py` · `parse.py`(캐시인지 디스패치) ·
`parse_mineru.py`(MinerU) · `parse_pdfplumber.py`(폴백) · `index.py`(Chroma 2컬렉션 BM25+dense RRF) ·
`tokenizer.py`(kiwi) · `retrieve.py`(결정론 라우팅) · `answer.py`(단일호출 text/vision) · `models.py`(Ollama) ·
`ingest.py`·`imaging.py`·`metrics.py`·`evaluate.py`·`schema.py`·`utils.py`·`envcheck.py`.

## 6. 산출물 위치

- **코드**: [코드/rag2](코드/rag2) (내부 `cache/parsed`[MinerU 13문서]·`index/chroma` 포함)
- **결과물**: `결과물/rag2_outputs`(evaluation·ingest_summary·catalog_match) · `결과물/test_2_timecost`(카탈로그 유무 16문항 시간비교) · `결과물/test12_비교실험`(test2 런 + test1↔test2 FINAL_REPORT·ANSWERS·비교 하네스) · `결과물/test2_vlm_probe`(비전경로 검증 20Q×2) · [결과물/통합_실험_보고서.md](결과물/통합_실험_보고서.md)
- **사전데이터**: `사전데이터/데이터 카탈로그 작업 파일`(13 PDF, test_1과 동일 corpus) · 카탈로그 xlsx 2종 · `rag_eval_dataset.json`(16Q) · `final_qa_dataset.json`(3Q) · `vlm_probe_dataset.json`(20Q)

## 7. 재현 · 주의

```
cmd /c conda activate intern_chatbot && python -m rag2 ingest
cmd /c conda activate intern_chatbot && python -m rag2 ask "<질문>"
```

> ⚠️ **경로 주의**: `config.yaml`은 사전데이터를 `../데이터…`로 참조하나 정리로 `사전데이터/`로 분리되어
> 재-ingest 경로가 깨져 있다(재실행 시 경로 조정 필요). 실행 검증은 지양, 비교대조로 검증함.
