# RAG3 — 한국어 문서 RAG 모듈

사내 문서(PDF) 기반 질의응답 파이프라인. MinerU 파싱 → 하이브리드 검색(BM25+dense, RRF) →
크로스인코더 리랭크 → small-to-big 페이지 승격 → LLM 답변 → 검증/결정론 롤백까지 포함한
완성형 모듈로, LangGraph 등 외부 오케스트레이터의 한 노드로 바로 연결할 수 있다.

**전 구간 로컬 실행**(Ollama + HuggingFace 리랭커). 외부 API·API 키 불필요.

## 성능 (36문항 회귀셋, FINAL_REPORT.md 기준)

| 지표 | 값 |
|---|---|
| page_hit@3 / doc_hit@3 | **0.774 / 1.0** |
| 환각(근거 없는 생성) | **0건** |
| 무관 질문 거절률 | **100%** (5/5) |
| vision 오독률 | **0%** |
| 평균/최대 응답 시간 | 63.4s / 148.6s (180s 데드라인 내) |
| 평균 모델 호출 | 2.97회 |

상세 근거·실패 분석: [probes/results/FINAL_REPORT.md](probes/results/FINAL_REPORT.md)

## 설치

```bat
:: 1) conda 환경 (Windows)
conda create -n intern_chatbot python=3.11
conda activate intern_chatbot
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

:: 2) Ollama 모델 (로컬 서버 127.0.0.1:11434)
ollama pull gemma4:12b
ollama pull gemma4:e4b
ollama pull embeddinggemma
:: 리랭커(BAAI/bge-reranker-v2-m3)는 첫 실행 시 HuggingFace에서 자동 다운로드
```

권장 하드웨어: VRAM 16GB (12b 8.4GB + 리랭커 1.5GB + 임베딩 0.7GB 동시 상주).
환경 점검: `python -m rag3 check`

## 디렉터리 구조

```
test_3/
├── ask_cli.py            대화형 질의 CLI (모델 1회 로드 후 REPL)
├── requirements.txt
├── rag3/                 핵심 패키지
│   ├── engine.py         Rag3Engine — 외부 연동 진입점 (아래 "모듈 API")
│   ├── controller.py     완성 파이프라인 상태기계 (S1~S8)
│   ├── add_doc.py        신규 문서 증분 추가/교체/제거
│   ├── config.yaml       모든 튜닝 파라미터 (CONFIG_파라미터_가이드.md 참고)
│   ├── cache/parsed_v25/ 문서별 파싱 캐시 (manifest + content_list + 페이지 PNG)
│   ├── index/            검색 인덱스 (flat_chunk npz+json, page_store.json)
│   └── outputs/          리포트 + evidence/ (근거 이미지 내보내기)
├── webapp/               발표용 데모 웹 (FastAPI + 단일 HTML)
└── probes/results/       Phase 0~4 실험 리포트 (FINAL_REPORT.md 포함)
```

원본 PDF·Excel 카탈로그 위치는 `rag3/config.yaml`의 `documents_dir`/`catalog_excel_path`
(현재 `../../rag_test/test_2/...` 상대경로).

## 실행 방법

모든 명령은 `test_3/`에서, `intern_chatbot` 환경으로:

```bat
cmd /c conda activate intern_chatbot && cd test_3 && python ask_cli.py            :: 대화형(권장)
python ask_cli.py --file 질문.txt --save-evidence                                 :: 1회 질의 + 근거 이미지 저장
python -m rag3 check                                                              :: 환경 점검
python -m rag3 ingest                                                             :: 전체 재색인 (보통 불필요)
python -m rag3 add --pdf 새문서.pdf                                               :: 신규 문서 증분 추가 (아래 참고)
python -m rag3 evaluate --eval-file ..\rag_test\test_2\rag_eval_dataset.json      :: 검색 지표 평가 (주의: 아래 참고)
python webapp\server.py --port 8000                                               :: 데모 웹서버
```

> **평가 경로 주의**: `python -m rag3 evaluate`/`ask`는 검증·롤백이 없는 **단순 단일패스**로 답변해
> 검색 지표(doc/page_hit) 확인용이다. FINAL_REPORT의 답변 품질 수치(avg_kw_hit 0.528 등)는
> **완성 파이프라인(controller)** 기준이므로 직접 비교하면 안 된다. 답변 품질 회귀 확인은
> controller를 쓰는 `tmp/verify_controller_regression.py`(대표 4문항, ~4분) 또는
> `tmp(프로젝트 루트)/eval_varfix.py`(36문항 전체)를 사용할 것.

## 모듈 API (LangGraph 연동용)

```python
import sys; sys.path.insert(0, r"...\test_3")   # 또는 test_3를 PYTHONPATH에
from rag3 import Rag3Engine

engine = Rag3Engine()          # 프로세스당 1회 (설정+백엔드+리랭커+인덱스 웜 로딩, ~10s)
engine.warm_up(deep=True)      # 선택: LLM까지 VRAM 상주 (첫 질문 지연 제거)
result = engine.ask("질문", save_evidence=True)
```

`ask()` 반환(`rag3.engine.AskResult`, dict):

| 키 | 타입 | 설명 |
|---|---|---|
| `final_answer` | str | 최종 답변 (근거 없으면 정직한 "확인 불가" 회피) |
| `answer_path` | str | `text` \| `vision` \| `none`(거절/회피) |
| `confidence` | str | `high` \| `low` \| `abstain` \| `unknown` |
| `evidence` | list | 근거 페이지: `document_name`, `page_number`, **`page_image_resolved`**(페이지 PNG 절대경로), `table_crop_resolved` |
| `selected_pages` | list | evidence와 같은 순서 + `page_score`(리랭크 점수) 등 상세 메타 |
| `rerank_top_score` | float | 최상위 근거 점수 (무관 질문이면 낮음) |
| `verification` | dict | 숫자 대조·groundedness 검증 결과 |
| `metrics` | dict | 모델 호출수 + `timings_seconds{retrieve,answer,total}` |
| `evidence_files` | list | `save_evidence=True`일 때 `outputs/evidence/<run_id>/`에 복사된 이미지 목록 |
| `run_id`, `question`, `selected_documents`, `route_reason`, `rollback_history` | | 부가 정보 |

LangGraph 노드 예시(langgraph 미의존):

```python
_ENGINE = Rag3Engine()          # 모듈 로드 시 1회

def rag3_node(state: dict) -> dict:
    r = _ENGINE.ask(state["question"])
    return {
        "answer": r["final_answer"],
        "confidence": r["confidence"],
        "evidence": [
            {"doc": e["document_name"], "page": e["page_number"], "image": e["page_image_resolved"]}
            for e in r["evidence"]
        ],
    }
```

주의: 엔진은 **프로세스당 1개**, 질문은 **직렬 처리**(GPU 1장 — 동시 호출 금지).
`ollama_keep_alive=30m`이라 30분 유휴 후 첫 질문은 콜드스타트로 느릴 수 있다
(length-retry가 자동 방어, `warm_up(deep=True)` 재호출 권장).

## 신규 문서 추가

1. **Excel 카탈로그**(`데이터카탈로그_DCAT_선정파일_RAG최적화.xlsx`, 시트 `선정파일_Distribution_파일목록`)
   **마지막 행에** 새 문서 행 추가 — `dct:title(파일명)` 필수, 분류/범위/키워드는 검색 정확도에 직결되니 채울 것.
   (중간 삽입 금지: row_id가 위치 기반이라 뒤 행들이 밀린다)
2. PDF를 `documents_dir` 아래에 복사 (파일명 = 카탈로그 title과 일치)
3. ```bat
   python -m rag3 add --pdf 새문서.pdf
   ```

동작: MinerU pipeline 파싱(기존 13개 문서와 동일 품질) → figure 페이지가 있으면 vlm-engine
텍스트화 자동 수행(Phase 3 품질 단계, `--skip-vlm`으로 생략, 페이지당 1~2분) → **신규 청크만
임베딩**해 기존 인덱스에 증분 병합. 같은 문서를 다시 add하면 교체(replace).

- 제거: `python -m rag3 add --remove 문서명.pdf` (파싱 캐시는 보존 → 재추가 빠름)
- 내용이 바뀐 동명 PDF: `--force-parse`
- **떠 있는 웹서버/REPL은 add 후 재시작해야 새 문서를 본다** (인메모리 인덱스 캐시)
- add는 GPU를 쓰므로 답변 서빙 중이 아닐 때 실행할 것

## 데모 웹페이지 (발표용)

```bat
cmd /c conda activate intern_chatbot && cd test_3 && python webapp\server.py --port 8000
```

브라우저에서 `http://127.0.0.1:8000` — 질문 입력 → 답변 + 근거 페이지 이미지 + 소요시간/신뢰도/
리랭크 점수/파이프라인 단계 표시. 우측에 벤치마크 성능 패널(FINAL_REPORT 수치).
발표 직전 `POST /api/warmup`(페이지의 워밍업 버튼)으로 모델을 VRAM에 상주시킬 것.
질문은 한 번에 하나만 처리(동시 요청은 429). 데모 등급 — 인증/HTTPS 없음, 외부 공개 금지.

## 설정/튜닝

`rag3/config.yaml` 단일 파일. 파라미터별 의미·재색인 필요 여부는
[CONFIG_파라미터_가이드.md](CONFIG_파라미터_가이드.md) 참고. 현재 값이 Phase 4 ablation으로
확정된 최적값이므로 변경 시 `python -m rag3 evaluate`로 회귀 확인 권장.

## 알려진 한계 (FINAL_REPORT §7)

1. 래스터 표/카드 UI 스크린샷 페이지는 파싱 한계로 답변 불가(정직 회피, 환각 아님)
2. 반복 서식 페이지 검색 미스 일부 (7/31)
3. 코퍼스가 ~10만 청크를 넘으면 `flat_index.py`의 1차 검색을 FAISS 등으로 교체 필요
   (변경 범위는 해당 파일로 격리됨)
