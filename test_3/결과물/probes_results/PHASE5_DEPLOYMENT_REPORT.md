# Phase 5 보고서 — RAG3 배포 준비 (문서추가 CLI · 모듈화 · 근거 이미지 · 데모 웹)

> 작성: 2026-07-17. 대상: Phase 4 완료본(FINAL_REPORT.md, varfix baseline) 이후의 배포 준비 작업.
> 환경: intern_chatbot(conda), MinerU 3.4.4, FastAPI/uvicorn(기설치), RTX 5060 Ti 16GB, 전 구간 로컬.

## 1. 배경과 목표

Phase 4까지 완성된 RAG3(page_hit@3 0.774 · doc_hit@3 1.0 · 환각 0 · 무관거절 100%)를
**실제 챗봇 시스템에 연결 가능한 형태**로 정리한다. 요구사항 4건:

| # | 요구사항 | 결과 |
|---|---|---|
| 1 | 신규 문서를 CLI 한 줄로 기존 데이터셋에 추가 | ✅ `python -m rag3 add --pdf <파일명>` |
| 2 | 다른 개발자(LangGraph 오케스트레이터 담당)에게 인계 가능한 모듈화 | ✅ `Rag3Engine` + README/requirements |
| 3 | 답변 근거 문서 페이지를 이미지로 함께 출력 | ✅ 경로 자동 부착 + 파일 내보내기 |
| 4 | 발표용 데모 웹페이지 (답변+근거이미지+성능지표) | ✅ FastAPI 단일 페이지 데모 |

제약: 기존 기능·인덱스 포맷 유지, test_3 원본 코드 최소 수정(추가 위주), 실험 코드는 tmp/ 격리.
사용자 결정: 신규 문서 파싱은 기존과 동일한 MinerU 흐름, 메타데이터는 Excel 카탈로그 행 추가 방식 유지.

## 2. 구현 내용

### 2.1 신규 문서 증분 추가 (`rag3/add_doc.py`, `python -m rag3 add`)

기존 ingest는 전체 재빌드(전 문서 재임베딩) 방식이라 증분 추가가 불가능했다. 새 흐름:

```
[선행] Excel 카탈로그 끝에 행 추가 + PDF를 documents_dir에 복사
python -m rag3 add --pdf 새문서.pdf
  1) 카탈로그 로드/매칭 → 대상 행 확정 (실패 시 근접 후보 3개 안내)
  2) MinerU pipeline 파싱 (기존 13개 문서와 100% 동일 경로; 캐시 있으면 재사용)
  3) 파싱 산출을 parsed_v25로 이전  ← ingest 폴백의 "청크 0개" 함정 회피(§4.1)
  4) figure 페이지 있으면 vlm-engine 텍스트화 자동 병합 (Phase 3 품질 단계 재현, --skip-vlm 가능)
  5) 공유 헬퍼(collect_chunk_records)로 청크 생성 → 신규 청크만 임베딩 → flat 인덱스 append
  6) page_store 병합 + add_doc_report.json / ingest_summary.json 갱신
```

- **신규 청크만 임베딩** — 기존 2,562개 벡터는 그대로 유지 (테스트 문서 2쪽 추가 전 과정 28초).
- 같은 문서 재추가 = **교체(replace)**, `--remove <파일명|slug>` = 제거(파싱 캐시는 보존 → 재추가 저렴).
- BM25는 디스크에 저장되지 않고 로드 시 재구성되므로 append만으로 자동 일관.

지원 변경: `flat_index.py`에 `append()`/`remove_doc()`/`_save()` **추가**(기존 메서드 무변경),
`ingest.py`의 청크 레코드 생성부를 `collect_chunk_records()`로 추출(색인 텍스트 포맷이 리랭킹에
load-bearing이라 add와 ingest가 반드시 한 곳을 공유해야 함 — byte-identical 검증됨, §3.2).

### 2.2 모듈화 (`rag3/engine.py`, `rag3/__init__.py`)

```python
from rag3 import Rag3Engine
engine = Rag3Engine()          # 프로세스당 1회 (설정+백엔드+리랭커+인덱스 웜 로딩)
engine.warm_up(deep=True)      # 선택: LLM까지 VRAM 상주 (발표/서비스 직전)
result = engine.ask("질문", save_evidence=True)   # AskResult (TypedDict로 스키마 명시)
```

- 반환은 controller의 dict를 **그대로 유지 + 추가 키만 부착**(기존 소비자 완전 호환).
- 입출력 스키마·LangGraph 노드 예시(langgraph 미의존)·설치/실행법을 `README.md`에 문서화,
  `requirements.txt`는 intern_chatbot 실측 버전으로 고정.
- 무거운 import(torch 등)는 lazy — `import rag3` 자체는 가벼움.

### 2.3 근거 이미지 (`rag3/evidence.py`)

- 페이지 PNG(969장, 200dpi)는 파싱 캐시에 이미 존재했고 evidence에 경로도 실려 있었으나 stale
  가능성이 있었다. `resolve_evidence()`가 `answer.resolve_cached_path`로 실존 절대경로를
  `page_image_resolved`/`table_crop_resolved` 키로 부착(기존 키 무변경).
- `export_evidence()`는 `outputs/evidence/<run_id>/`에 **ascii-safe 파일명**(`ev1_p0011.png`)으로
  복사 + manifest.json 기록 — 한글/Windows 경로가 URL·콘솔에 노출되지 않게 하는 장치.
- `ask_cli.py`: `[근거 페이지]` 아래 이미지 경로 1줄씩 출력 + `--save-evidence` 플래그(기존 출력 불변).

### 2.4 데모 웹 (`webapp/server.py` + `webapp/static/index.html`)

```
cmd /c conda activate intern_chatbot && cd test_3 && python webapp\server.py --port 8000
```

- 기동 시 엔진 1회 로드 + 딥 워밍업(12b VRAM 상주). 발표 직전 재워밍업 버튼(POST /api/warmup).
- 화면: 질문 입력 → 경과 초 카운터("보통 25~90초") → ① 답변 카드(경로/신뢰도/검증 배지)
  ② **근거 페이지 이미지**(클릭 확대, 캡션에 문서명·페이지·rerank 점수) ③ 파이프라인 단계표
  (ask_cli 출력 미러: S1~S8, 모델 호출수, retrieve/answer/total 타이밍) ④ 우측에 이번 질문
  지표 + FINAL_REPORT 벤치마크 패널(36문항 기준 명시) + 코퍼스 현황(add 후 자동 갱신).
- GPU 1장 전제: 동시 질문은 429로 거절(threading.Lock). 데모 등급 — 무인증, 외부 공개 금지.
- 이미지 서빙은 파싱 캐시 직접 마운트가 아니라 evidence 내보내기 폴더만 노출.
- 잘못된 환경(시스템 파이썬)으로 실행 시 트레이스백 전에 한국어 안내를 출력하는 가드 포함.

### 2.5 파일 변경 요약

| 구분 | 파일 | 내용 |
|---|---|---|
| 신규 | `rag3/engine.py` `rag3/evidence.py` `rag3/add_doc.py` | 모듈 API / 근거 이미지 / 증분 추가 |
| 신규 | `webapp/server.py` `webapp/static/index.html` | 데모 웹 |
| 신규 | `README.md` `requirements.txt` | 인계 문서 |
| 수정(최소) | `rag3/__init__.py`(re-export) `rag3/flat_index.py`(메서드 추가만) `rag3/ingest.py`(헬퍼 추출, 동작 동일) `rag3/__main__.py`(add 서브커맨드) `ask_cli.py`(이미지 줄 +10줄) | |
| 무수정 | config/controller/retrieve/answer/page_store/models/chunking/catalog/parse*/vlm_reparse/evaluate | 디스크 인덱스 포맷 불변 |

## 3. 검증 결과 (전부 통과)

| 검증 | 방법 | 결과 |
|---|---|---|
| 3.1 무회귀 베이스라인 | 작업 전 core_001 골든 출력 확보 → 각 단계 후 재실행 | 답변·근거 페이지·rerank 점수 동일, 신규 줄만 추가 |
| 3.2 ingest 리팩토링 동등성 | 13개 문서 전체 청크 재생성 vs 프로덕션 docs.json | **2,562/2,562 byte-identical**, 불일치 0 |
| 3.3 인덱스 증분 왕복 | 복사본에서 remove_doc→append 후 비교 | count 복원, top10 검색 교집합 10/10, npz 포맷/정규화 유지, 중복 id 방어 동작 |
| 3.4 문서 추가 E2E | 테스트 PDF(2쪽) 실제 추가→질문→교체→제거 | 신규 질문 신뢰도 high 정답+이미지, 기존 검색 점수까지 불변, 제거 후 2562청크/969페이지 완전 원복 |
| 3.5 모듈 API | `Rag3Engine.ask()` 실행 | 베이스라인 동일 답변, resolved 경로 실존, evidence 파일 생성 |
| 3.6 웹 데모 | 전 엔드포인트 + 브라우저 흐름 | 정답(웜 36초), 이미지 3장 200 OK, 동시 요청 429, stats/health 정상 |
| 3.7 답변 품질 회귀 | **controller 경로** 대표 4문항 vs varfix 기준치 | core_001 kw **1.0**(기준 1.0) · core_004 **0.6**(0.6) · vp_009 **1.0**(1.0) · irrelevant_001 **1초 거절** — 회귀 0 |

재사용 가능한 검증 스크립트는 `test_3/tmp/verify_*.py` 6종으로 보존
(controller 회귀 체크: `verify_controller_regression.py`).

## 4. 작업 중 발견 사항 (주의 필요)

1. **`python -m rag3 evaluate`/`ask`는 단일패스(generate_answer) 경로** — 검증·롤백이 없는
   구버전 경로라 FINAL_REPORT의 답변 품질 수치(controller 기준)와 **직접 비교 금지**.
   실측: 단일패스는 core_001을 웜에서도 결정론적으로 오답(문맥 요약, kw 0.14)하지만 controller는
   kw 1.0. 검색 지표(doc/page_hit) 확인용으로만 사용할 것. README에 명시.
2. **Chroma page_index는 add에서 upsert하지 않음** — B6(HNSW 크로스프로세스 로드 실패)로 이미
   기본 경로에서 배제된 인덱스인데, upsert 시도가 로드 실패→**컬렉션 파괴적 재생성**을 유발함을
   실측 확인. catalog_index만 try/except로 동기화. 게이트 모드 재활성화 시 풀 ingest 필요.
3. **GPU 프로세스 단독 실행 원칙 재확증** — 평가 프로세스 2개가 겹치자 둘 다 로그 없이 무증상
   크래시. add/evaluate/서버는 반드시 한 번에 하나만.
4. **add 후 떠 있는 웹서버/REPL은 재시작 필요** — 인메모리 인덱스 캐시 때문 (README 명시).
5. ingest 폴백 함정(§2.1의 3단계 근거): parsed_v25에 없는 문서를 풀 ingest가 파싱하면
   `cache/parsed/`에 쓰지만 청크화는 `parsed_v25`만 읽어 **청크 0개**가 됨 — 신규 문서는 반드시
   `add` 명령 사용.

## 5. 남은 과제 (Phase 4 이월 + 신규)

- 래스터 표/카드 UI 파싱 천장(vp_006/007/013 등) — 전용 표 OCR 검토 (FINAL_REPORT §7).
- 검색 미스 7문항 — 반복 서식 페이지 앵커 강화.
- 코퍼스 ~10만 청크 초과 시 flat index → FAISS 교체 (`flat_index.py`로 격리됨).
- 평가셋 확장(~50문항) 및 holdout 검증 — 사람 검수 협업 필요(보류 유지).
- LangGraph 실연동은 오케스트레이터 측 개발자와 `README.md`의 노드 예시 기준으로 진행.

## 6. 인수인계 체크리스트

- [ ] `conda create` + `pip install -r requirements.txt` + Ollama 모델 3종 pull → `python -m rag3 check`
- [ ] `python ask_cli.py`로 질의 동작 확인 (모든 실행은 **intern_chatbot 환경 필수**)
- [ ] `README.md`의 모듈 API 스키마 표로 LangGraph 노드 연결
- [ ] 발표 리허설: 서버 기동 → 워밍업 버튼 → 준비된 질문 2~3개 (첫 질문이 가장 느림)
