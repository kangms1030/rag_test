# rag_catalog_experiment

카탈로그(Excel)로 **후보 문서를 2개 이하로 먼저 좁힌 뒤**, 그 PDF에만 VLM 기반 Multimodal RAG를
수행하는 실험 파이프라인. 모든 PDF를 한 번에 RAG하지 않는다.

> 실험 결과·의사결정·한계·보완 방향은 → **[REPORT.md](rag_catalog_experiment/REPORT.md)**
> 이 문서는 **어떻게 쓰는가 / 어떻게 동작하는가**만 다룬다.

---

## 빠른 시작

```bash
# 0) 환경 (최초 1회)
cmd /c conda activate intern_chatbot && pip install -r rag_catalog_experiment/requirements.txt
cmd /c ollama pull gemma4:12b
cmd /c ollama pull embeddinggemma

# 1) 카탈로그 + PDF 색인 (약 7분, VLM 호출 없음)
python -m rag_catalog_experiment ingest

# 2) 질문
python -m rag_catalog_experiment ask "무선 AP 장애 발생 시 조치 절차를 알려줘"
```

결과는 콘솔 + `outputs/answer_{run_id}.json`, 근거 이미지는 `outputs/evidence/{run_id}/`.

> `python -s`(no-user-site)로 실행하기를 권장한다. `AppData\Roaming\Python\...`에 그림자
> site-packages가 있어 재현성이 흔들린 사례가 있었다.

---

## 답변 JSON을 읽는 법 (근거 추적)

`final_answer` 본문은 근거를 `(이미지 1)`처럼 지목한다. 그 번호를 따라가면 원본 페이지가 나온다.

```
final_answer:  "... 전원 및 케이블 연결을 점검합니다(이미지 1)."
                                                        │
     ┌──────────────────────────────────────────────────┘
     ▼
source_images[]   image_index: 1  →  학교 무선인터넷 자가진단 체크리스트(안).pdf, p11
     │
     ▼
page_evidence[]   image_index: 1  →  bbox {x1:0.04, y1:0.16, x2:0.93, y2:0.58}
                                     highlighted_page_path: ..._p0011_highlighted.jpg
                                     crop_image_path:       ..._p0011_ev1_crop.jpg
                                     confidence: 0.95
```

| 파일 | 용도 |
|---|---|
| `*_highlighted.jpg` | 원본 페이지 전체 + 근거 영역을 색 박스로 표시 (맥락 확인) |
| `*_ev{N}_crop.jpg` | 근거 영역만 잘라낸 확대본 (내용 확인) |

**검증 방법**: `(이미지 N)` → `source_images`에서 문서·페이지 확인 → `page_evidence`의
`highlighted_page_path`를 열어 박스 안 내용이 답변을 뒷받침하는지 눈으로 대조.

> **주의**: 답변 속 숫자를 그대로 믿지 말 것. VLM이 조밀한 표의 숫자를 잘못 전사한 실측 사례가
> 있다 (REPORT.md §5 A1). 반드시 crop 이미지와 대조할 것.

**bbox 계약**: normalized `0.0~1.0`, 왼쪽위 `(x1,y1)` ~ 오른쪽아래 `(x2,y2)`.

---

## 파이프라인

```
질문
 └─▶ Query Analyzer (LLM, JSON)
      └─▶ catalog_index 검색 ──▶ 후보 PDF ≤2개   ※ 관련 없으면 0개 → VLM 호출 없이 종료
           └─▶ 문서별 페이지 축소
           │     · 텍스트 문서: pdfplumber 텍스트로 BM25 → 상위 12페이지
           │     · 스캔 문서  : prefilter 불가 → 전 페이지 (--limit-pages로 상한)
           └─▶ 해당 페이지만 VLM 요약 (lazy, 디스크 캐시) ──▶ page_index
                └─▶ 상위 4페이지만 semantic visual chunking ──▶ visual_chunk_index
                     └─▶ page image 최대 3장을 VLM에 전달
                          └─▶ 답변 + 근거 bbox ──▶ crop / highlight 저장
                               └─▶ verification (LLM) ──▶ outputs/answer_{run_id}.json
```

**왜 lazy인가**: 969페이지 전체를 미리 요약+청킹하면 로컬에서 5시간이 넘는다. 그래서 `ingest`는
VLM을 **한 번도** 부르지 않고(임베딩만 호출), 페이지 요약·청킹은 `ask` 시점에 선정된 문서의
상위 후보 페이지에만 만들어 `cache/`에 저장한다. 질문할수록 인덱스가 따뜻해진다.

**왜 스캔/텍스트를 나누는가**: 문서 13개 중 스캔본은 **2개뿐**이다. 나머지 11개는 정상 텍스트
레이어가 있어 값싼 BM25 prefilter로 VLM 호출 대상을 좁힐 수 있다. (실측치는 REPORT.md §3.1)

---

## 구성 요소

| 파일 | 역할 |
|---|---|
| `catalog.py` | Excel 파싱, 컬럼 역할 자동 추론, PDF 파일명 매칭(NFC 정규화 + fuzzy) |
| `pdf_parse.py` | PyMuPDF 렌더링, pdfplumber 텍스트/표 추출, 페이지 분류, manifest 캐시 |
| `page_summary.py` | VLM 페이지 요약 (lazy + 디스크 캐시, backend별 분리) |
| `visual_chunk.py` | VLM semantic visual chunking, bbox 검증·병합 |
| `indexes.py` | Chroma 4개 컬렉션(catalog/page/visual_chunk/filename) + BM25/dense 하이브리드(RRF), 손상 자가복구 |
| `retrieval.py` | Query Analyzer, 3모드(catalog/no_catalog/filename_only) 문서/페이지/청크 검색, `depth` 절단점 |
| `metrics.py` | 캐시 독립적 비용 계측 (contextvar 누산기) |
| `answer.py` | 이미지 매니페스트 기반 VLM 답변, 근거 crop/highlight, 숫자 재검증 |
| `ocr.py` | `evidence_text` 채우기 전용(pdfplumber 좌표 우선, OCR optional) |
| `schema.py` | 최종 JSON 조립 + 스키마 검증(v2) + 분류 저장 |
| `eval_sets.py` | 평가셋 로드/생성(human 결정론적/synthetic LLM)/QA 엑셀 임포트 |
| `evaluate.py` | 평가셋 실행 + 지표 집계 |
| `experiments/compare.py`, `experiments/thresholds.py` | 모드 비교 리포트, 게이트 임계값 분포 리포트 |
| `runs.py`, `envcheck.py` | 산출물 정리(`clean-runs`/`validate-outputs`), 환경 점검(`check-env`) |

### 모델

| 용도 | 모델 | 비고 |
|---|---|---|
| LLM (질문 분석, 검증) | `gemma4:12b` | 7.6GB, 16GB VRAM에 여유 |
| VLM (요약·청킹·답변) | `gemma4:12b` | vision 지원 |
| Fallback | `gemma4:e4b` | OOM/로드 실패 시 1회 자동 재시도 |
| Embedding | `embeddinggemma` | task prefix 적용, provider를 LLM과 통일해 torch 설치 회피 |

`gemma4:26b`(17GB)는 16GB VRAM 초과로 미사용.

### 인덱스

Chroma `PersistentClient`, `index/chroma/{backend_id}/`에 4개 컬렉션:
`catalog_index`(문서 설명문) · `page_index`(페이지 요약) · `visual_chunk_index`(청크) ·
`filename_index`(파일명+폴더명만 — `filename_only` 모드 전용).

검색은 **BM25 + dense를 RRF(k=60)로 융합**하되, RRF 점수는 0~1로 정규화한다.
컬렉션이 작아(수백 건) 매 쿼리마다 BM25를 즉석 재구성한다 — 별도 인덱스 파일이 필요 없다.
한국어 토크나이저는 기본이 형태소 분석기 없이 정규식 + 한글 char bigram 근사(`char_bigram`)이고,
`config.tokenizer: "kiwi"`로 kiwipiepy 형태소 분석기를 켤 수 있다(미설치 시 자동 폴백).

디스크 캐시(`cache/summaries/`, `cache/chunks/`)와 Chroma 인덱스 모두 `backend_id`로 디렉터리를
나눈다 — mock(64차원 해시)과 실제(embeddinggemma 768차원)의 임베딩 차원이 다를 뿐 아니라,
분리하지 않으면 `--mock` 스모크 테스트가 실제 문서 캐시에 `[MOCK]` 요약을 남길 수 있다
(실제로 겪은 문제, REPORT.md §10.2).

### 문서 선정 컷오프 (retrieval_mode별로 다른 게이트)

`top_docs`(기본 2)는 **상한일 뿐** 억지로 채우지 않는다. `catalog`/`filename_only`/`no_catalog`는
서로 다른 텍스트 분포(카탈로그 설명문/파일명/페이지 요약) 위에서 동작하므로 게이트 값을
공유하지 않는다(`min_dense_similarity`/`min_filename_dense_similarity`/`min_page_dense_similarity`).

1. **절대 관련성**: 최상위 후보의 dense cosine 유사도 < 게이트값 → `[]`
2. **상대 점수**(catalog/filename_only만): RRF 정규화 점수 < `min_doc_score`(0.35) → `[]`
3. **격차**(catalog/filename_only만): 2등 < 1등 × `doc_score_gap_ratio`(0.6) → 1개만 선정
4. 0개면 VLM을 호출하지 않고 즉시 `"선택된 문서에서 확인 불가"`

> 1번이 없으면 무관한 질문("김치찌개 레시피")도 문서를 고른다. RRF는 *상대 순위*만 반영하므로
> 문서가 13개뿐이면 무엇이든 1등이 존재하기 때문이다. (REPORT.md §5 B4)
>
> 게이트값을 손으로 고르지 않으려면 `thresholds` 커맨드로 relevant/irrelevant dense 유사도
> 분포를 실측해 참고할 것 (REPORT.md §10.3 — 실측 결과 filename_only는 현재 게이트가 일부
> 정답 질문보다 높아 조정이 필요함을 확인).

---

## CLI

```bash
# 환경 점검 (모델 미호출)
python -m rag_catalog_experiment check-env

# 색인
python -m rag_catalog_experiment ingest
python -m rag_catalog_experiment ingest --limit-docs 2 --limit-pages 5 --mock   # 배선 검증
python -m rag_catalog_experiment ingest --vlm-summary --visual-chunking         # 사전 워밍(느림)

# 질문 (retrieval-mode: catalog | no_catalog | filename_only, 기본은 config.yaml)
python -m rag_catalog_experiment ask "질문"
python -m rag_catalog_experiment ask "질문" --retrieval-mode no_catalog
python -m rag_catalog_experiment ask "질문" --top-docs 2 --top-pages 5 --top-chunks 8
python -m rag_catalog_experiment ask "질문" --limit-pages 12   # 스캔 문서 VLM 요약 상한
python -m rag_catalog_experiment ask "질문" --out path.json

# 평가셋 만들기
python -m rag_catalog_experiment gen-eval --out eval_sets/human_20.json --mode human --target-count 20
python -m rag_catalog_experiment import-qa --qa-file "../QA_샘플파일_100개.xlsx" --out eval_sets/qa.json

# 평가/비교 (depth: docs=VLM 0회 | pages | answer=전체)
python -m rag_catalog_experiment evaluate --eval-file eval_sets/human_20.json --retrieval-mode catalog
python -m rag_catalog_experiment compare --eval-file eval_sets/human_20.json --modes catalog no_catalog filename_only --depth docs
python -m rag_catalog_experiment thresholds --eval-file eval_sets/human_20.json --modes catalog no_catalog filename_only

# 산출물 정리
python -m rag_catalog_experiment clean-runs --dry-run
python -m rag_catalog_experiment validate-outputs
```

`--config`는 서브커맨드 **앞**에 온다: `python -m rag_catalog_experiment --config X.yaml ask "..."`

`gen-eval --mode human`(기본)은 카탈로그의 `대표 예상 질문` 컬럼(사람이 작성)만 쓰고 LLM을
호출하지 않는다 — 결정론적이라 재현 가능하다. `--mode synthetic`은 모자란 만큼 LLM으로 채우고
`"source": "synthetic"`을 붙인다. `evaluate`/`compare`는 human/synthetic/sample_qa를 분리
집계한다 — 카탈로그 실효성 비교 실험의 실측 결과와 방법론은 → **[REPORT.md §10](rag_catalog_experiment/REPORT.md)**.

---

## 주요 설정 (`config.yaml`)

| 키 | 기본 | 의미 |
|---|---|---|
| `retrieval_mode` | `catalog` | `ask`/`evaluate`의 기본 모드 (catalog\|no_catalog\|filename_only) |
| `page_prefilter_topn` | 12 | 텍스트 문서에서 BM25로 뽑아 VLM 요약할 페이지 수 |
| `chunk_pages_topk` | 4 | visual chunking 수행 페이지 수 |
| `max_images_per_call` | 3 | 답변 생성 시 VLM에 넣는 이미지 수 상한 (VRAM 보호) |
| `image_max_side` | 1280 | VLM 입력 이미지 리사이즈 |
| `page_render_dpi` | 150 | 페이지 렌더링 DPI |
| `scanned_text_ratio_threshold` | 0.2 | 이 미만이면 "스캔 문서"로 판정 |
| `min_dense_similarity` | 0.35 | catalog 모드 문서 선정 게이트 |
| `min_filename_dense_similarity` | 0.30 | filename_only 모드 게이트 |
| `min_page_dense_similarity` | 0.35 | no_catalog 모드 게이트 |
| `no_catalog_page_prefilter_topn` | 24 | no_catalog가 전역에서 실제 요약으로 승격할 페이지 상한(비용 예산을 catalog와 맞춤) |
| `min_doc_score` | 0.35 | RRF 정규화 점수 하한 |
| `verify_with_crop_image` | false | 숫자 포함 답변을 crop 이미지로 재검증(REPORT.md A1/A4) |
| `tokenizer` | `char_bigram` | `char_bigram`\|`kiwi`(kiwipiepy 미설치 시 자동 폴백) |
| `enable_ocr` | false | `evidence_text` 채우기 전용. `page_index` 색인에는 절대 연결 안 함(카탈로그 스캔 문서 비용 우위 보존) |

---

## Mock 모드

`--mock`은 모델을 부르지 않고 결정론적 값을 반환한다. **배선 검증 전용**이다.

| 단계 | mock 동작 |
|---|---|
| Query Analyzer | 고정 dict |
| 임베딩 | SHA1 해시 기반 64차원 벡터 — **의미 없음** |
| 페이지 요약 | `"[MOCK] ... 요약"` |
| visual chunking | 페이지 전체를 덮는 chunk 1개 |
| 답변 / 검증 | 고정 답변, 항상 `is_answer_supported: true` |

임베딩이 무의미한 해시라 `ask --mock`은 대개 컷오프에 걸려 `selected_documents: []`를 낸다.
정상 경로의 **품질** 검증에는 쓸 수 없다. 출력 JSON에는 `"mock": true`가 붙는다.

---

## 이 코드가 참고 노트북에서 고친 것

`vlm_test_v2.ipynb` 기준.

| 노트북의 문제 | 이 코드의 처리 |
|---|---|
| evidence bbox를 항상 첫 이미지(`img_paths[0]`)에 귀속 — 3장을 보여주고도 | 프롬프트에 이미지 매니페스트를 넣고 VLM이 `image_index`를 명시하게 강제 |
| 결과 파일명이 질문 앞 20자 → 다른 질문끼리 덮어씀 | `run_id = {UTC timestamp}_{sha1(question)[:6]}` |
| JSON 파싱 실패 시 조용히 evidence를 버림 | 중괄호 밸런싱 추출 + 1회 재프롬프트, 실패 시 명시적 로깅 |
| crop은 2% 패딩, highlight는 패딩 없음 → 서로 다른 영역 | 둘 다 동일한 패딩 좌표 사용 |
| `MAIN_DIR = "/data/manual/network_test"` 리눅스 하드코딩 | `config.yaml` + `pathlib` |
| `gemma4:latest` / `embeddinggemma` 로컬에 없음 | `gemma4:12b` pull (Ollama 업그레이드 필요했음) |

기타 세부(카탈로그 컬럼 추론, 파일명 NFC 매칭 등)는 각 모듈 docstring 참조.

---

## 한계

요약만 적는다. 각 항목의 **보완 방향과 우선순위**는 → [REPORT.md §7](rag_catalog_experiment/REPORT.md).

- **VLM이 조밀한 표의 숫자를 잘못 전사할 수 있다** — 답변 숫자는 crop과 대조 필수 (가장 중요)
- `verification`이 텍스트만 보고 검증 → VLM 오독을 잡지 못함
- `evidence_text`는 항상 빈 문자열, `evidence_type`은 항상 `"visual"` (OCR 미구현)
- `min_dense_similarity`는 embeddinggemma·문서 13개 기준 손튜닝 값 → 모델/규모 바뀌면 재보정
- BM25 한국어 토크나이저는 형태소 분석기가 아닌 char-bigram 근사
- `catalog_index`가 13행뿐 → 문서 선정 난이도가 실제보다 낮음
- 스캔 문서 첫 질의는 느림 (100p ≈ 10분대), 이후 캐시 히트
- Chroma 1.5.9 컬렉션 손상 경험 있음 (자가복구하지만 재색인 비용 발생)
- **실제 모델 정량 평가(`evaluate`) 미실시** — 현재는 smoke test 3건의 일화적 증거뿐
