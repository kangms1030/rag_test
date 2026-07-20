# test_1 최종 보고서 — 카탈로그 게이트 + 쿼리타임 VLM 멀티모달 RAG

> 세대 정체성: **1세대**. "질문이 들어올 때 그때그때 VLM(비전 언어모델)이 PDF 페이지 이미지를 직접
> 보고 답한다"는 **쿼리타임 멀티모달 RAG**. 데이터 카탈로그(DCAT)로 후보 문서를 먼저 좁힌 뒤,
> 좁혀진 문서의 페이지를 VLM이 요약·청킹·판독한다.
> 코드: [코드/rag_catalog_experiment](코드/rag_catalog_experiment) · 원본 상세 리포트: [결과물/REPORT.md](결과물/REPORT.md)

---

## 1. 한 줄 요약

DCAT 카탈로그 → (후보 문서 ≤2개 선정) → 페이지 이미지 렌더 → **VLM이 페이지를 요약·시각청킹·판독하여
답변** → LLM 검증. 모든 무거운 추론을 **질문 시점에** 수행하는 lazy 구조라 정확하지만 느리고, VLM이
조밀한 표의 숫자를 오독하는 한계가 있었다(→ 이것이 test_2 재설계의 직접 동기).

## 2. RAG 파이프라인 (단계별 사용 모델 · 사용 프로그램)

모든 모델 호출 백엔드는 **Ollama**(`num_ctx=8192`, `keep_alive=10m`, `temperature=0.0`).
모델 태그는 `코드/rag_catalog_experiment/config.yaml` 원문 그대로다.

| # | 단계 | 사용 모델 | 사용 프로그램 / 라이브러리 | 설명 |
|---|---|---|---|---|
| 1 | 카탈로그 적재·색인 (offline ingest) | **embeddinggemma** (임베딩, 768d) | pandas · openpyxl(Excel 적재) · rapidfuzz(파일명 퍼지매칭) · unicodedata(NFC) · **chromadb** PersistentClient | DCAT Excel(`데이터카탈로그_DCAT_선정파일_RAG최적화.xlsx`)의 행을 13개 PDF와 매칭, `catalog_index`·`filename_index` 구축. **VLM은 ingest에서 호출 안 함**(임베딩만). |
| 2 | PDF 파싱·페이지 렌더 (질문 시) | — (모델 없음) | **PyMuPDF(fitz)** 페이지→JPEG **150 DPI** 렌더 · **pdfplumber** 텍스트/표 추출 · **Pillow(PIL)** 리사이즈(최대변 1280) | 페이지를 스캔/텍스트로 분류(`scanned_text_ratio_threshold=0.2`), manifest 캐시. |
| 3 | 쿼리 분석 | **LLM gemma4:12b** | Ollama chat (JSON 출력) | 질문 → intent·keywords·retrieval_query JSON(`retrieval.py::analyze_query`). |
| 4 | 하이브리드 검색 | **embeddinggemma** (쿼리 임베딩) | **chromadb**(catalog/page/visual_chunk/filename 4컬렉션) · **rank-bm25**(BM25Okapi, 질문마다 재구성) · **RRF(k=60)** 정규화 · 토크나이저 **char_bigram**(기본) / **kiwipiepy**(옵션) | 문서 선정 게이트: `min_dense_similarity=0.35`·`min_doc_score=0.35`, `top_docs=2`. |
| 5 | 페이지 요약 (lazy) | **VLM gemma4:12b** | Ollama vision · 디스크 캐시(`cache/summaries/{backend_id}/`) | 상위 후보 페이지만 요약(`page_prefilter_topn=12`/문서) → `page_index`. |
| 6 | 시각적 청킹 | **VLM gemma4:12b** | Ollama vision · bbox 검증/병합 | 상위 `chunk_pages_topk=4` 페이지에서 의미 단위 시각청크 추출 → `visual_chunk_index`. |
| 7 | 답변 생성 | **VLM gemma4:12b** | Ollama vision(이미지 최대 `max_images_per_call=3`, 1280px) · **Pillow**(근거 crop/highlight 이미지 생성) | 페이지 이미지를 직접 보고 `final_answer` + 근거 bbox(정규화 0~1) 출력. |
| 8 | 답변 검증 | **LLM gemma4:12b** | Ollama chat · (옵션) crop 재검증 | `is_answer_supported` 판정. 수치 재검증용 `verify_with_crop_image`는 기본 false. |
| (폴백) | OOM/로드 실패 시 | **gemma4:e4b** | 자동 1회 재시도 | 모든 모델 단계 공통 폴백. |
| (옵션) | OCR | pytesseract | 기본 **비활성**(`enable_ocr=false`) | `evidence_text` 보강용, 미사용(evidence_type는 항상 `visual`). |

**핵심 설계**: 카탈로그가 후보 PDF를 ≤2개로 좁힌 뒤 그 문서에만 VLM RAG를 돌린다. VLM 요약·청킹은
lazy(질문 시)+디스크 캐시라 `ingest`는 임베딩만 수행한다. **MinerU는 test_1에 없다**(test_2에서 도입).

## 3. 사용 스택 총정리

| 구분 | 사용 | 비고 |
|---|---|---|
| LLM(쿼리분석·검증) | `gemma4:12b` | Ollama |
| VLM(요약·청킹·답변) | `gemma4:12b` (vision) | Ollama |
| 폴백 | `gemma4:e4b` | Ollama |
| 임베딩 | `embeddinggemma` (768d, task-prefix) | Ollama(torch 미설치 목적) |
| 벡터DB | Chroma(`PersistentClient`, 4컬렉션) | cosine |
| 희소검색 | rank-bm25(BM25Okapi) + RRF(k=60) | |
| 형태소 | char_bigram(기본) / kiwipiepy(옵션) | |
| PDF | PyMuPDF(fitz, 150DPI) · pdfplumber · Pillow | |
| 런타임 | Python 3.11 · conda `intern_chatbot` · RTX 5060 Ti 16GB | |

## 4. 주요 결과 (신규 3문항 냉시작 비교, `결과물/test12_비교실험_test1런`)

| 지표 | catalog | no_catalog |
|---|---|---|
| doc/page match | 100% / 100% | 100% / 100% |
| 평균 keyword_recall | 0.646 | 0.562 |
| VLM 호출 / 문항 | 29 | 29 |
| 문항당 평균 소요 | 868.4s (14.5분, 냉시작) | 816.2s |

- **수치 정확도 한계**: 8회 수치 관측 중 **4회 오류**(표 오독 2 + 환각 2). bbox(어디를 봤는지)는 맞으나
  표 안 숫자 전사(무엇을 읽었는지)에서 틀림 — VLM 비전 판독의 구조적 약점(`결과물/REPORT.md §5 A1`).
- **속도**: 냉시작 조건상 `page_summary`(41%)+`visual_chunk`(47.5%)가 전체의 ~88%. 같은 문서 반복
  질의 시 캐시 히트로 answer+verify 수준(문항당 40~70초대)으로 감소.
- **카탈로그 효과**: 이 설정에선 `no_catalog_page_prefilter_topn=24`가 catalog의 `2×12=24`와 동일하게
  맞춰져 VLM 호출량 차이는 없고, `embed_calls`만 차이(11 vs 31~33).

## 5. 코드 구성 (`코드/rag_catalog_experiment`)

`__main__.py`(CLI: ingest/ask/evaluate/compare/thresholds/…) · `catalog.py`(DCAT 매칭) ·
`pdf_parse.py`(PyMuPDF+pdfplumber) · `indexes.py`(Chroma+BM25 RRF) · `retrieval.py`(쿼리분석·3검색모드) ·
`page_summary.py`(lazy VLM 요약) · `visual_chunk.py`(시각청킹) · `answer.py`(VLM 답변·근거crop·수치재검증) ·
`models.py`(Ollama/Mock 백엔드) · `tokenizer.py`(char_bigram/kiwi) · `schema.py`·`metrics.py`·`evaluate.py`.

## 6. 산출물 위치

- **코드**: [코드/rag_catalog_experiment](코드/rag_catalog_experiment) (내부 `index/`·`cache/` 빌드 산출 포함)
- **결과물**: [결과물/REPORT.md](결과물/REPORT.md)(원본 상세) · `결과물/rag_catalog_experiment_results`(00_SUMMARY·질문별 답변비교·results.xlsx) · `결과물/rag_catalog_experiment_outputs`(evaluations·comparisons·evidence) · `결과물/test12_비교실험_test1런`(test1_catalog·test1_no_catalog 최종 런)
- **사전데이터**: `사전데이터/데이터 카탈로그 작업 파일`(13 PDF) · 카탈로그 xlsx 3종 · `rag_eval_dataset.json` · `최정현연구원_코드1차`(원본 참고 노트북)

## 7. 한계

VLM 표 숫자 오독 · 냉시작 지연(문항당 10분+) · no_catalog의 환각 경향 · verify가 근거 추출 실패 시 정답도
"미지원" 오판정. → test_2에서 **MinerU 사전파싱 텍스트 RAG**로 재설계하여 해소(→ 없음, [test_2/CHANGES_from_test1.md](../test_2/CHANGES_from_test1.md) 참조).

## 8. 재현 · 주의

```
cmd /c conda activate intern_chatbot && python -m rag_catalog_experiment ingest
cmd /c conda activate intern_chatbot && python -m rag_catalog_experiment ask "<질문>"
```

> ⚠️ **경로 주의**: `config.yaml`은 사전데이터를 `../데이터…` 상대경로로 참조한다. 폴더 정리로
> 사전데이터가 `사전데이터/`로 분리되어 **이 경로는 깨져 있다**(재-ingest 시 `config.yaml`의
> `catalog_excel_path`/`documents_dir`를 `../사전데이터/…`로 조정 필요). 로컬 모델 특성상 실행 검증은
> 지양했으며, 본 폴더의 검증은 원본 대비 비교대조로 수행했다.
