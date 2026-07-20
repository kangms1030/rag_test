# RAG_3 구축 참고 — 실험 팩트 정리 (사실·근거 기반)

> **문서 목적**: test_1 → test_2 → test_2_timecost → test12_total_test → test2_vlm_probe로 이어진 5개 실험에서
> **측정·확인된 사실과 그 근거(파일 경로·수치·코드 위치)만** 모은 문서다. 작성자의 의견·추천·해석은 넣지 않는다.
> rag_3를 설계하는 에이전트가 이 문서 하나로 "지금까지 무엇이 사실로 확인됐고, 원본 근거는 어디에 있는가"를
> 바로 찾을 수 있게 하는 것이 목표다.
>
> **표기 규칙**: 각 사실 뒤에 `[근거: 경로/수치]`로 원본 위치를 명시한다. 근거는 요약하지 않고 원본에 접근 가능하도록 남긴다.
>
> **폴더 재구성 주의(2026-07-15 기준)**: test_1/test_2 코드가 `rag_test/` 하위로 이동했다. 아래 경로는 현재 위치 기준이다.
> - test_1 코드: `rag_test/test_1/rag_catalog_experiment/`
> - test_2 코드: `rag_test/test_2/rag2/`
> - test_2_timecost: `test_2_timecost/`
> - test12 통합실험: `test12_total_test/`
> - test2_vlm_probe(최신): `test12_total_test/test2_vlm_probe/`
> - 이전 통합보고서(vlm_probe 미포함): `통합_실험_보고서.md` (루트, 533줄, 2026-07-14 작성)
>
> **주의**: 폴더 이동으로 캐시 manifest.json 내부의 절대경로(`page_image_path` 등)는 이전 위치(`챗봇\test_2\...`)를
> 가리키는 stale 상태다. 실제 파일은 `rag_test/test_2/rag2/cache/parsed/{slug}/pages/pXXXX.png`에 있다. `[근거: 실측 2026-07-15]`

---

## 0. 원본 근거 소스 맵 (전체 목록)

| 실험 | 원본 리포트 | 원본 데이터(JSON 등) | 코드 |
|---|---|---|---|
| test_1 (VLM query-time RAG) | `rag_test/test_1/rag_catalog_experiment/REPORT.md` | `rag_test/test_1/.../outputs/` | `rag_test/test_1/rag_catalog_experiment/*.py` |
| test_2 (MinerU 사전파싱 RAG) | 코드 주석 + `rag_test/test_2/rag2/outputs/` | `rag_test/test_2/rag2/outputs/evaluation_20260713T100344Z.json`, `ingest_summary.json`, `catalog_match_report.json` | `rag_test/test_2/rag2/*.py` |
| test_2_timecost (catalog vs no_catalog, 16문항) | `test_2_timecost/results/REPORT.md` | `test_2_timecost/results/compare_20260713T163107Z.json` | `test_2_timecost/retrieve_no_catalog.py`, `run_experiment.py` |
| test12 (test1 vs test2, 3문항) | `test12_total_test/FINAL_REPORT.md` | `test12_total_test/all_results_20260714T040610Z.json` | `test12_total_test/run_final_experiment.py`, `run_single_test1.py`, `run_single_test2.py` |
| test2_vlm_probe (VLM 경로 검증, 20문항×2모드) | `test12_total_test/test2_vlm_probe/results/REPORT.md` | `test12_total_test/test2_vlm_probe/results/all_results_20260714T063740Z.json`, `catalog/`, `no_catalog/`, `calibration_*.json` | `test12_total_test/test2_vlm_probe/{run_vlm_probe.py, calibrate_routing.py, vlm_probe_dataset.json}` |

---

## 1. 공통 환경 (모든 실험 공유)

### 1.1 코퍼스
- 대상 PDF 13종, 총 **969페이지**. `[근거: test_2/rag2/outputs/ingest_summary.json → total_pages:969, documents_parsed:13]`
- 카탈로그 엑셀 row 14개 중 13개가 PDF와 매칭(1개 미매칭). `[근거: ingest_summary.json → catalog_rows:14, catalog_matched:13, catalog_unmatched:1]`
- 스캔 문서 판정: MinerU ingest 기준 `scanned_documents: 9` (13개 중 9개에 스캔 페이지 포함). `[근거: ingest_summary.json → scanned_documents:9]`
  - 단, test_2의 `is_scanned` 판정은 "PyMuPDF 네이티브 텍스트 < 50자"이며 페이지 단위다. `[근거: rag_test/test_2/rag2/parse_mineru.py → native_text = fitz_page.get_text(); is_scanned = len(native_text) < 50]`
  - 이전 통합보고서(코퍼스 설명)는 스캔 문서를 2개(현황분석서 100p·MDM 94p중 11p만 텍스트)로 기술 — 문서 단위 관점. `[근거: 통합_실험_보고서.md → 0.공통배경]`
- 표 페이지 수 `table_pages: 326`, 그림(figure) 페이지 수 `figure_pages: 77`. `[근거: ingest_summary.json]`

### 1.2 모델·인프라 스택
| 구분 | 값 | 근거 |
|---|---|---|
| LLM(텍스트 답변) | `gemma4:12b` | `rag_test/test_2/rag2/config.yaml → text_answer_model` |
| VLM(비전 답변) | `gemma4:12b` | `config.yaml → vision_answer_model` |
| 폴백 모델 | `gemma4:e4b` (호출 실패 시 1회 자동 재시도) | `config.yaml → fallback_model`; `rag_test/test_2/rag2/models.py → OllamaBackend._chat` except 절 |
| 임베딩 | `embeddinggemma` (query/doc task prefix 적용) | `config.yaml → embedding_model, embed_query_prefix:"task: search result | query: ", embed_doc_prefix:"title: none | text: "` |
| 벡터 DB | Chroma (cosine, HNSW) | `rag_test/test_2/rag2/index.py → get_or_create_collection(metadata={"hnsw:space":"cosine"})` |
| 검색 융합 | BM25(kiwi 형태소) + dense, RRF(k=60) | `config.yaml → rrf_k:60, tokenizer:"kiwi"`; `index.py → HybridIndex.query` |
| Ollama 컨텍스트 | `num_ctx: 8192`, `keep_alive: "10m"` | `config.yaml` |
| VRAM 제약 | 16GB. `gemma4:26b`(17GB)는 VRAM 초과로 미사용. `[근거: 통합_실험_보고서.md → 공통 스택 표, "미사용: gemma4:26b(17GB) — 16GB VRAM 초과"]` |
| MinerU 파서 | `parser:"mineru"`, `mineru_backend:"pipeline"`, `mineru_lang:"korean"`, `mineru_device:"cuda"`(실패 시 cpu 폴백) | `config.yaml`; `parse_mineru.py → _resolve_device` |

---

## 2. 파이프라인 구조 팩트

### 2.1 test_1 (VLM query-time RAG) — 질문마다 VLM 다단계 호출
파이프라인 단계 순서 `[근거: rag_test/test_1/rag_catalog_experiment/retrieval.py, answer.py, page_summary.py, visual_chunk.py]`:
1. **Query Analyzer** (LLM 텍스트 JSON ×1): 질문을 intent/keywords/domain_terms/retrieval_query로 분석. `[retrieval.py → analyze_query]`
2. **카탈로그 문서 선정** (≤2개): 카탈로그 설명문 하이브리드 검색. `[retrieval.py → select_documents]`
3. **페이지 축소**: 스캔 문서면 전 페이지 후보, 텍스트 문서면 pdfplumber 텍스트 BM25 prefilter(문서당 12페이지). `[retrieval.py → _prefilter_pages_by_text, config page_prefilter_topn]`
4. **1차 VLM — 페이지 요약**: 후보 페이지 각각을 VLM으로 한 문단 요약 → page_index 재색인. `[page_summary.py → ensure_page_summaries, PROMPT_SUMMARY]`
5. **2차 VLM — 시각 청킹**: 상위 페이지를 VLM으로 semantic chunk+bbox 분해. `[visual_chunk.py → PROMPT_CHUNK]`
6. **3차 VLM — 최종 답변**: 선정 페이지 **원본 이미지**를 VLM에 보내 답변+evidence bbox 생성. `[answer.py → PROMPT_ANSWER, chat_vision_json]`
7. **4차 (조건부) — 검증**: 숫자 주장이 있으면 crop 이미지를 다시 보여주며 재대조. `[answer.py → verify_answer, PROMPT_VERIFY, PROMPT_VERIFY_NUMERIC, config.verify_with_crop_image]`

**측정된 호출 횟수(냉시작 기준, test12 3문항)**: 질문당 VLM **29회**(summary 24 + chunk 4 + answer 1), embed 11~33회, LLM(analyzer/verify) 1~2회. `[근거: test12_total_test/FINAL_REPORT.md → §3.1 표]`

### 2.2 test_2 (MinerU 사전파싱 RAG) — 질문마다 모델 호출 최대 2회
- **ingest(오프라인 1회)**: 969페이지 전체를 MinerU pipeline으로 파싱해 페이지별 텍스트/표(HTML)/OCR/도표판정까지 전부 선연산 → page_index/catalog_index에 색인. VLM 미호출. 소요 `elapsed_seconds: 200.64초`. `[근거: rag_test/test_2/rag2/ingest.py; ingest_summary.json → elapsed_seconds:200.64, parser_used_counts:{mineru:13}]`
- **ask(질문 시)**: ① 질문 임베딩 1회 → ② catalog_index 게이트+문서선정 → ③ page_index 페이지선정 → ④ 결정론적 라우팅(`_route`, 모델 미호출) → ⑤ 최종 답변 1회(text 또는 vision). `[근거: rag_test/test_2/rag2/retrieve.py, answer.py]`
- **측정된 호출 횟수**: 질문당 정확히 **2회**(embed 1 + text_answer 1) 또는 vision일 때 embed 1 + vision_answer 1. test2_vlm_probe 40건 중 vision 실행 13건 전부 `vision_answer_calls=1`. `[근거: test2_vlm_probe/results/all_results_20260714T063740Z.json → model_calls]`
- **test_2에는 별도 verify/rollback 단계가 없다.** 답변 생성은 단일 호출(transcribe-then-answer 프롬프트만). `[근거: rag_test/test_2/rag2/answer.py → generate_answer는 chat_text 또는 chat_vision_text 1회만 호출, verify 함수 부재; models.py에 chat_json/verify 경로 없음]`

### 2.3 test_2 라우팅 규칙 (결정론적, 모델 미호출) — rag_3에서 유지 여부 검토 대상
`_route`는 **1순위 근거 페이지(top_pages[0]) 하나의 메타데이터만** 보고 경로를 정한다. `[근거: rag_test/test_2/rag2/retrieve.py → _route]`
- `page_type=="figure"` AND `figure_area_ratio >= 0.5` → **vision** `[config figure_area_ratio_threshold:0.5]`
- `scanned_table_verify(true)` AND `is_scanned` AND `has_table` → **vision** `[config scanned_table_verify:true]`
- 그 외 → **text** (기본)
- **함의(측정된 구조적 사실)**: 정답이 2~3순위 페이지에 있어도 라우팅은 1순위 기준으로만 결정된다. `[근거: retrieve.py → best = top_pages[0].metadata]`

### 2.4 test_2 게이트 임계값 (무관·저관련 질문 거절 지점)
`[근거: rag_test/test_2/rag2/config.yaml + retrieve.py → _select_documents]`
- `min_dense_similarity: 0.35` — 1순위 후보의 dense cosine 유사도가 이 미만이면 `selected_documents=[]`(문서 없음).
- `min_doc_score: 0.35` — RRF 정규화 점수(0~1) 미만이면 문서 없음.
- `doc_score_gap_ratio: 0.6` — 2등 점수가 1등의 60% 미만이면 1개만 선정.
- `top_docs: 2`, `top_pages: 3` — 문서 최대 2개, 페이지 최대 3개.

### 2.5 test_2 최종 답변에 전달되는 입력 (text vs vision)
- **text 경로**: 선정 페이지 top_pages(최대 3장)의 **텍스트 전부**를 컨텍스트로 프롬프트에 삽입. 표는 MinerU가 뽑은 HTML 구조 그대로 포함. `[근거: rag_test/test_2/rag2/answer.py → _format_context, PROMPT_TEXT_ANSWER]`
- **vision 경로**: **1순위 페이지 이미지 1장만** VLM에 전달. `table_crop_path`가 있으면 그 표 크롭, 없으면 페이지 전체 이미지. `[근거: answer.py → generate_answer: image_path = table_crop_path if best.get("table_crop_path") else page_image_path; 이미지 1장만 chat_vision_text에 전달]`
- **vision 프롬프트**는 "표/도표를 먼저 그대로 옮겨 적고(전사) 그 값만으로 답하라"는 transcribe-then-answer 구조. `[근거: answer.py → PROMPT_VISION_ANSWER]`

---

## 3. 이미지 해상도 팩트 (rag_3 프롬프트 4번 관련)

### 3.1 test_1 (이전 세대) — 해상도를 낮춘 것이 문서에 명시됨
- `page_render_dpi: 150` (원형 노트북은 200이었으나 캐시 용량 때문에 낮춤). `[근거: rag_test/test_1/rag_catalog_experiment/config.yaml → page_render_dpi:150; REPORT.md → D5 "page_render_dpi 200→150, 969페이지 캐시 용량 때문에 낮춤"]`
- `image_max_side: 1280` — VLM 입력 전 최대 변 1280px로 리사이즈. `[근거: config.yaml → image_max_side:1280; models.py → _resize_for_vlm(p, config.image_max_side)]`
- 최종 답변 시 **페이지 전체 이미지**를 VLM에 전달(표만 크롭하지 않음). `[근거: rag_test/test_1/.../answer.py → image_paths = [Path(img["image_path"]) for img in images]]`
- test_1 자체 REPORT가 숫자 오독을 한계로 명시: **"VLM이 표의 숫자를 잘못 전사(27,054→27,004, 엘텍→엘벨). 답변 숫자를 신뢰 불가"**, 보완방향으로 "표 crop을 고해상도(원본 DPI)로 다시 렌더링 — 현재는 1280px로 축소해서 넣음". `[근거: rag_test/test_1/rag_catalog_experiment/REPORT.md → 한계 A1]`

### 3.2 test_2 (현재 세대) — 해상도 실측값
- `answer_image_dpi: 200`, `answer_image_max_side: 2048`. `[근거: rag_test/test_2/rag2/config.yaml]`
- VLM 전송 전 `_resize_for_vlm`이 `scale = min(1.0, max_side / max(w,h))` — **확대는 하지 않고 축소만** 한다. `[근거: rag_test/test_2/rag2/models.py → _resize_for_vlm]`
- **실측 해상도(2026-07-15)** `[근거: 실측, 파일 열어 PIL로 측정]`:
  | 페이지 | 페이지 렌더(200 DPI) | 표 크롭(MinerU 원본) | VLM 전달 시 |
  |---|---|---|---|
  | 현황분석서 p9(스캔+표) | 1654×2339 | 1262×854 | 표크롭 1262(2048 미만→그대로) |
  | MDM p83(스캔+표) | 2167×1500 | 1176×543 | 표크롭 1176(그대로) |
  | 무선랜 p13(figure) | 1528×2167 | 없음 | 페이지 2167→2048 축소 |
- **정리된 사실**: test_2의 표 크롭은 max_side 약 1176~1262px(2048 미만이라 축소 안 됨). figure 페이지는 페이지 전체(max 2167~2339)를 2048로 축소해 전달. test_1은 모든 이미지를 1280px로 축소. `[근거: 위 실측 + 양 config]`

---

## 4. 실험별 정량 결과 (측정치 전부)

### 4.1 test_2_timecost — catalog vs no_catalog (16문항: core 13 + irrelevant 3)
`[근거: test_2_timecost/results/REPORT.md → 요약 비교표]`
| mode | doc_hit | page_hit | kw_hit | 무관거절 | 경로(text/vision/none) | avg_docs | avg_모델호출 | avg_초 | 총_초 |
|---|---|---|---|---|---|---|---|---|---|
| catalog | 0.692 | 0.538 | 0.423 | 1.000 | 11/0/5 | 1.375 | 1.688 | 35.339 | 565.420 |
| no_catalog | 0.923 | 0.769 | 0.705 | 1.000 | 12/1/3 | 1.312 | 1.812 | 30.491 | 487.860 |
- **이 16문항에서 vision 경로: catalog 0회, no_catalog 1회.** `[근거: 위 표 경로 컬럼]`
- catalog가 no_catalog보다 doc_hit·page_hit·kw_hit 모두 낮고 평균 소요시간도 더 김. `[근거: 위 표]`
- catalog baseline 정합성: `rag_test/test_2/rag2/outputs/evaluation_20260713T100344Z.json`과 지표 완전 일치(doc_hit 0.692 등). `[근거: test_2_timecost/results/REPORT.md → "baseline 정합성 대조"]`

### 4.2 test12_total_test — test1 vs test2 (신규 3문항)
`[근거: test12_total_test/FINAL_REPORT.md]`
| 비교군 | 총 소요시간(3문항) | 문항당 평균 | doc/page_match | 평균 kw_recall | LLM/문항 | VLM/문항 | Embed/문항 |
|---|---|---|---|---|---|---|---|
| test1.catalog | 2605.4s | 868.4s | 100%/100% | 0.646 | 1.67 | 29 | 11 |
| test1.no_catalog | 2448.5s | 816.2s | 100%/100% | 0.562 | 2.00 | 29 | 32 |
| test2.catalog | 61.4s | 20.5s | 100%/100% | 0.736 | 1 | 0 | 1 |
| test2.no_catalog | 112.0s | 37.3s | 100%/100% | 0.736 | 1 | 0 | 1 |
- **test_2가 test_1보다 약 40~44배 빠름**(문항당 20.5s vs 868.4s). `[근거: FINAL_REPORT.md §1.3]`
- **이 3문항에서 test_2는 vision을 한 번도 타지 않음**(VLM/문항 = 0). MinerU가 표를 텍스트로 뽑아 text 경로로 처리됨. `[근거: FINAL_REPORT.md §1.2, §3.2]`
- 숫자 정확도: 8회 수치 관측 중 **test_1(VLM 판독) 4회 오류, test_2(텍스트 추출) 0회**. `[근거: FINAL_REPORT.md §2.3, §8.1]`
  - test_1 오독 실측: Cat.5 속도 10Mbps→100Mbps, 고객센터 1899-0979→1899-0970, 802.11ax "1GHz~6GHz"(환각). `[근거: FINAL_REPORT.md §4.1~4.3]`
- **test_1의 냉시작 소요시간(문항당 13~15분)은 매 질문 캐시를 비운 조건**이며 반복 질의 시 요약/청킹 캐시로 대부분 사라짐. `[근거: FINAL_REPORT.md §6.2]`
- **이 코드베이스 설정에서 catalog 유무가 test_1 VLM 호출량에 차이를 만들지 않음**(no_catalog_page_prefilter_topn=24가 catalog의 top_docs×page_prefilter_topn=24와 동일하게 맞춰짐). 실질 차이는 embed 호출 수(11 vs 31~33)뿐. `[근거: FINAL_REPORT.md §3.1, §8.1-4]`

### 4.3 test2_vlm_probe — VLM 경로 강제 유발 실험 (20문항×2모드=40건, 전부 성공)
문항 구성: vision 기대 15(scan_table 8 + figure 7) + text 대조 3 + 무관 2. 정답은 캐시 페이지 이미지를 사람이 직접 열람해 작성(파이프라인 OCR 비의존). `[근거: test2_vlm_probe/vlm_probe_dataset.json; results/REPORT.md §0.1]`

**(A) 라우팅 도달**: `[근거: results/REPORT.md §1; all_results JSON]`
| 모드 | vision 기대 15 중 실제 vision 도달 | 답변경로(vision/text/none) |
|---|---|---|
| catalog | 3/15 | 3 / 14 / 3 |
| no_catalog | 10/15 | 10 / 8 / 2 |
- 40건 중 **실제 vision 호출 13건**. no_catalog가 catalog보다 3배 이상 자주 vision 유발. `[근거: results/REPORT.md §1]`

**(B) vision 정확도**: vision 실행 13건 중 **1순위 페이지가 정답 페이지와 일치한 것은 7건**. 그 7건 중 **6건(86%)에서 오독 발생.** `[근거: results/REPORT.md §2; 아래 원문 대조]`
| qid/mode | 정답 페이지 봤나 | 판정 | 오독 내용(정답→모델출력) |
|---|---|---|---|
| vp_011/catalog | 예(p22) | 정답 | 전 항목 일치 |
| vp_009/catalog | 예(p13) | 오독 | "WiFi 개요 360"→"WIFI 캐너 360", "갤럭시 앱 스토어"→"앱 스토어"(누락) |
| vp_009/no_catalog | 예(p13) | 오독(동일 재현) | 위와 완전 동일 오류 |
| vp_006/no_catalog | 예(p83) | 오독 | 시리얼 "R54KB00TXWF"→"R54K000TWW", 이력 "37건"→"10건" |
| vp_007/no_catalog | 예(p88) | 오독/오귀속 | 다른 표(실사용자목록) 값 답함, 대여자 "135건"→"5건" |
| vp_010/no_catalog | 예(p11) | 오독 | "무선 집선 스위치"→"무선 확장 스위치", 정상/장애 IP 미구분 |
| vp_014/no_catalog | 예(p71) | 오독 | "인증"→"운영", 네트워크 과제 문구 원문과 무관하게 왜곡 |
- 오독 유형: 숫자, 문자열(시리얼), 용어(개요→캐너/집선→확장/인증→운영) 전 범주. `[근거: results/REPORT.md §2]`
- **동일 페이지·동일 오독 재현**: vp_009는 catalog/no_catalog 두 모드가 같은 p13을 보고 정확히 같은 오독. `[근거: 위 표]`

**(C) 답변 거절 분석 (무관 2문항 제외, content 실행 36건)**: `[근거: results/REPORT.md §3; all_results JSON 재집계]`
| 분류 | 정의 | catalog | no_catalog |
|---|---|---|---|
| ① 검색 게이트 실패 | 모델 호출 전 문서 후보 없음(none) | 1 (vp_005) | 0 |
| ② 생성 실패·빈 응답 | 정답 페이지가 컨텍스트에 있었으나 LLM이 빈 문자열 반환("선택된 문서에서 확인 불가"로 대체) | 3 (vp_002,006,007) | 0 |
| ③ 생성 실패·"확인 불가" 명시 | 정답 페이지가 있었으나 모델이 스스로 확인 불가라 답함 | 2 (vp_010,014) | 0 |
| ④ 검색 실패로 거절 | 검색된 페이지가 애초에 오답 페이지(정당한 못 찾음) | 4 | 5 |
| ⑤ 답변 시도(정답/부분/오독/오답) | 거절 안 하고 뭔가 답함 | 8 | 11 |
- **"정답 근거가 실제 주어졌는데 실패(①+②+③)": catalog 6/18(33.3%), no_catalog 0/18(0%).** 6건 전부 catalog 모드, 그중 5건 text 경로. `[근거: results/REPORT.md §3]`
- 6건 모두 정답 페이지가 검색 1순위가 아니라 2~3순위였음(3페이지 컨텍스트 안에 정답이 있었으나 답을 못 냄). `[근거: results/REPORT.md §3, 각 문항 selected_pages]`
- **최종 답변 산출 여부(무관 제외 18문항 기준)**: catalog 답변시도 8 / 부분거절 0 / 완전거절 10; no_catalog 답변시도 11 / 부분거절 2 / 완전거절 5. `[근거: all_results JSON 재집계]`

**(D) 시간 비용**: `[근거: results/REPORT.md §4]`
| 답변경로 | 실행 수 | 평균 총소요 | 평균 retrieve | 평균 answer |
|---|---|---|---|---|
| vision | 13 | 66.3s | 6.6s | 59.7s |
| text | 22 | 49.5s | 3.9s | 45.6s |
| none | 5 | 3.3s | - | - |
- 모드별 총 소요: catalog 802.1s(문항당 40.1s), no_catalog 1165.2s(문항당 58.3s). no_catalog가 45% 더 김(vision 호출 3배 많음). `[근거: results/REPORT.md §4]`
- Ollama 응답시간 자연변동 큼(예: vp_003 두 모드 모두 116~117s). `[근거: results/REPORT.md §4, §7]`

**(E) 문서·페이지 선정**: `[근거: results/REPORT.md §5]`
| 지표 | catalog | no_catalog |
|---|---|---|
| doc_hit_rate | 83.3% | 88.9% |
| page_hit_rate(top3) | 55.6% | 55.6% |
| avg_keyword_recall | 0.269 | 0.431 |
| 무관 거절 | 2/2 | 2/2 |

---

## 5. 카탈로그 효과에 대해 측정된 사실 (rag_3 프롬프트 1번 관련)

- **파이프라인 구조에 따라 카탈로그 효용의 방향이 반대로 나옴** `[근거: 통합_실험_보고서.md TL;DR; 각 REPORT]`:
  - test_1(VLM query-time): 문서 선정 정확도 catalog 1.000 vs no_catalog 0.250 (catalog 우세). `[근거: rag_test/test_1/rag_catalog_experiment/REPORT.md §10; 통합_실험_보고서.md]`
  - test_2(MinerU 사전파싱): doc_hit catalog 0.692 vs no_catalog 0.923 (catalog 열세). `[근거: test_2_timecost/results/REPORT.md]`
- **test_2에서 no_catalog가 우세한 원인(리포트 명시)**: test_2는 969페이지를 이미 저비용 사전색인해 둬서, 카탈로그로 문서를 좁히든 안 좁히든 페이지 검색 비용이 동일. 카탈로그가 없앨 수 있는 비용이 test_1(비카탈로그 시 query-time VLM 요약 폭증)보다 훨씬 작음. `[근거: test_2_timecost/results/REPORT.md → "원인 1 — 비용 구조 자체가 다르다"]`
- **카탈로그가 정답 문서를 놓친 실측 사례**(test_2_timecost): core_002(UTM 제조사) catalog 게이트 거절(dense 0.342<0.35), core_003·core_012 형제 문서 혼동. `[근거: test_2_timecost/results/REPORT.md → "원인 2"]`
- **카탈로그가 이득을 준 실측 사례**(test2_vlm_probe): vp_011(스쿨넷 이용관리 등록)에서 catalog는 정답 문서를 찾았으나 no_catalog는 형제 시스템(weiss.or.kr 통합로그인센터)으로 혼동. `[근거: test2_vlm_probe/results/REPORT.md §5]`
- **카탈로그가 조기 거절로 손해 준 실측 사례**(test2_vlm_probe): vp_005(전국 AP 집계) catalog 게이트에서 문서 못 찾아 none, no_catalog는 답변 시도. `[근거: test2_vlm_probe/results/REPORT.md §5]`
- **소규모 코퍼스(13~14 문서) 한정 결과라는 단서**: 리포트가 "문서 수가 훨씬 많아지면(예: Distribution 249행 전체) 카탈로그의 문서 범위 좁히기 효과가 다시 우세해질 가능성"을 한계로 명시. `[근거: test_2_timecost/results/REPORT.md → 한계 마지막 항목]`

---

## 6. 카탈로그 검색에 실제 사용되는 필드 (구현 사실)

- 카탈로그 엑셀에서 role은 13종 추론되나(title/theme/publisher/issued/purpose/description/keyword/scope/format/folder/sample_questions/download_url/latest), **검색 텍스트로 합쳐지는 것은 6종뿐**: 제목/분류/범위/설명/키워드/대표질문. `[근거: rag_test/test_2/rag2/catalog.py → _COLUMN_ROLE_KEYWORDS 및 load_catalog의 catalog_search_text 조합부]`
- 이 6종을 `" | "`로 합친 **문자열 1개**를 임베딩 1개 + BM25 토큰집합 1개로 만들어 질문과 통으로 비교(필드별 개별 비교 아님, 필드 가중치 없음). `[근거: catalog.py → search_text = " | ".join(text_parts); ingest.py → catalog_index.upsert(texts=[row.catalog_search_text])]`
- publisher/issued/purpose/format/folder/download_url/latest는 검색 스코어링에 미사용. `[근거: catalog.py → catalog_search_text 조합에 미포함]`
- 카탈로그·페이지 임베딩은 **ingest 시점에 1회 생성해 Chroma에 저장**, 질문 임베딩만 질의 시 생성. `[근거: index.py → upsert시 backend.embed(is_query=False); retrieve.py → backend.embed([question], is_query=True)]`

---

## 7. 알려진 실패 모드·한계 (측정된 것만)

1. **vision 경로 진입 시 오독률 높음**: 정답 페이지를 정확히 본 7건 중 6건(86%) 오독. `[근거: §4.3(B)]`
2. **catalog 모드의 "근거 있음에도 생성 실패"**: 정답 페이지가 top3 컨텍스트에 있어도 1순위가 아니면 빈 응답/확인불가로 실패 — catalog 33.3%(6/18), no_catalog 0%. `[근거: §4.3(C)]`
3. **반복 서식 스캔 문서에서 페이지 특정 실패**: 현황분석서(WEISS-DS-001)는 동일 표 서식(구분·제조사·모델명·수량·비고)이 지역명만 바꿔 17회+ 반복 → calibration에서 앵커 보강 2라운드에도 vision 도달 catalog 3/15, no_catalog 10/15에 그침. 문서는 맞혀도(doc_hit 높음) 페이지 특정 실패(page_hit 낮음). `[근거: test2_vlm_probe/results/REPORT.md §0.2, §5; calibration_*.json]`
4. **라우팅이 1순위 페이지에만 의존**: top_pages[0]만으로 경로 결정, 2~3순위에 정답 있어도 반영 안 됨. `[근거: §2.3]`
5. **단방향·무검증**: test_2 ask 경로는 검색→답변 단일 통과, verify/rollback/재검색 단계 없음. `[근거: §2.2]`
6. **표본 크기**: test2_vlm_probe N=20(vision 실제 도달 13), test12 N=3, test_2_timecost N=16 — 전부 소표본. `[근거: 각 REPORT 한계 절]`
7. **vision 유발률은 문항 설계 의존**: 일반 질의 16문항(rag_eval_dataset)에서는 vision 0회, vision 유발 목적 20문항에서 13/40(32%). 실제 사용 분포에서의 vision 비중은 미측정. `[근거: test_2_timecost §4.1(경로 11/0/5), test2_vlm_probe §1, §7]`

---

## 8. 재현 커맨드 (원본 환경)

```
# test_2 파이프라인 점검(모델 미호출)
cmd /c conda activate intern_chatbot && python rag_test/test_2/rag2/__main__.py check --skip-indexes

# test_2_timecost (catalog vs no_catalog, 16문항)  ※ 폴더 재구성으로 상대경로(../test_2) 조정 필요할 수 있음
cmd /c conda activate intern_chatbot && python test_2_timecost/run_experiment.py

# test12 통합(test1 vs test2, 3문항)
cmd /c conda activate intern_chatbot && python test12_total_test/run_final_experiment.py

# test2_vlm_probe (라우팅 사전점검 → 본 실행 40건)
cmd /c conda activate intern_chatbot && python test12_total_test/test2_vlm_probe/calibrate_routing.py
cmd /c conda activate intern_chatbot && python test12_total_test/test2_vlm_probe/run_vlm_probe.py
```

> **폴더 이동 영향**: `test_2_timecost/run_single_test2.py`·`test12_total_test/run_single_test2.py`는 `../test_2`(구 위치)를
> sys.path에 넣는다. test_2가 `rag_test/test_2`로 이동했으므로 재실행 시 경로 수정 필요. 단, **이미 저장된 결과 JSON·REPORT는
> 그대로 유효**하며 이 문서의 모든 수치는 그 저장된 산출물 기준이다. `[근거: 실측 2026-07-15 폴더 구조]`

---

## 9. 이 문서가 다루지 않는 것 (경계 명시)

- rag_3 설계안·개선 방향·기술 추천은 이 문서에 없다(사실 정리 문서이므로). 그 판단은 rag_3 기획 단계에서 수행한다.
- 각 실험 REPORT.md에는 작성자 해석/사견 절이 별도로 존재한다(예: `test_2_timecost/results/REPORT.md`의 "해석" 절, `통합_실험_보고서.md`의 "작성자 사견" 절). 그 해석이 필요하면 원본을 직접 참조할 것 — 이 문서에는 그 해석을 옮기지 않았다.
