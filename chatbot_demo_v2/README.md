# chatbot_demo_v2 — LangGraph 적극 활용 학교 유무선 장애상담 챗봇

v1(`chatbot_demo`)이 LangGraph를 **전진 DAG(관측용 라우터)** 로만 쓰던 것을 재설계해,
**사이클 · HITL 인터럽트 · 대화 메모리 · 서브그래프 · 커스텀 스트리밍**을 실제로 사용하는 데모입니다.
기존 `test_3/코드`의 RAG 엔진(rag3/rag3x)은 **무수정 vendoring**해 그대로 재사용하되,
블랙박스 호출이 아니라 **LangGraph 서브그래프로 재편성**해 내부 단계를 관측·스트리밍합니다.

> 이 폴더는 **독립 동작**합니다. 코드(`ragcore/`)와 데이터(`ragdata/`)를 자체 보유하며
> 다른 폴더(`test_3`, `chatbot_demo` 등)를 읽기만 하고 수정하지 않습니다.

**함께 볼 문서**
- [최종_구축_보고서.md](최종_구축_보고서.md) — 초기 구축 시점 기록(계획 대비 이행 점검 · 설계 판단 근거)
- [**2차_개선_보고서.md**](2차_개선_보고서.md) — 2026-07-27 품질 개선. **무엇이 생기고/바뀌고/없어졌는지**와
  실측으로 **기각한 가설**까지. 검색을 손대기 전에 §3 을 꼭 읽으세요
- [docs/파이프라인_비교.md](docs/파이프라인_비교.md) — v1 vs v2 파이프라인 그림 3장 + 실측 트레이스 해설
- [docs/개선작업_진행상황.md](docs/개선작업_진행상황.md) — 작업 진행 이력 · 다음 할 일

---

## 1. 빠른 시작 (Windows cmd)

터미널(cmd)에 **위에서부터 순서대로** 입력하세요.

```bat
:: 0) 프로젝트 폴더로 이동
cd /d "C:\Users\minsoo\Desktop\아이티지엔 인턴\챗봇"

:: 1) conda 환경 활성화  → 프롬프트가 (intern_chatbot) 으로 바뀌면 성공
conda activate intern_chatbot

:: 2) 데이터 부트스트랩 (최초 1회. test_3의 색인·파싱캐시를 ragdata/로 복사, 약 590MB)
python chatbot_demo_v2\scripts\bootstrap_data.py

:: 3) FAQ 근거링크 생성 (최초 1회 / faq.json 을 고쳤을 때)
python -m chatbot_demo_v2.scripts.build_faq_doc_links

:: 4) 서버 실행 (v1과 병행 가능 — v1은 8001)
python -m chatbot_demo_v2 --port 8002
```
→ 브라우저에서 **http://127.0.0.1:8002** 접속. 종료는 `Ctrl+C`.

2·3번은 **최초 1회만** 필요합니다. 2회차부터는 0·1·4번만 하면 되고, 이 셋을 묶은 스크립트도 있습니다:

```bat
chatbot_demo_v2\scripts\run_demo.cmd          :: 포트 지정: run_demo.cmd 8003
```

### 그 외 명령
```bat
:: 테스트 (GPU·API키 없이 실행)
python -X utf8 -m pytest chatbot_demo_v2\tests -q

:: 실기 통합(원본 엔진과의 등가성) 테스트 — 실제 GPU·Gemini 사용
set RUN_RAG_INTEGRATION=1
python -X utf8 -m pytest chatbot_demo_v2\tests\test_rag_subgraph.py -q

:: 파이프라인 다이어그램 다시 그리기 → docs/*.svg
python -m chatbot_demo_v2.scripts.render_pipeline
```

### 품질 측정 · 데이터 도구 (2026-07-27 추가)
```bat
:: 골든셋 37문항으로 그래프 전체 평가 → runtime/reports/eval_<tag>_*.json
python -X utf8 chatbot_demo_v2\scripts\run_eval.py --tag mytest --no-cache
python -X utf8 chatbot_demo_v2\scripts\run_eval.py --category ambiguous --limit 5

:: 평가 리포트 A/B 비교 (개선/악화 문항까지 표시)
python -X utf8 chatbot_demo_v2\scripts\ab_compare.py baseline mytest

:: FAQ 의미 매칭용 임베딩 생성 (FAQ 를 고쳤으면 다시 실행)
python -X utf8 chatbot_demo_v2\scripts\build_faq_embeddings.py

:: FAQ 매칭 임계값 캘리브레이션 (점수 분포 → 권고 임계)
python -X utf8 chatbot_demo_v2\scripts\calibrate_faq_threshold.py

:: 색인 재구축 — 원자적 교체. 기존 index 를 절대 덮어쓰지 않는다
python -X utf8 chatbot_demo_v2\scripts\reindex.py            :: index_new 에 빌드
python -X utf8 chatbot_demo_v2\scripts\reindex.py --promote  :: 검증 통과 후 교체
python -X utf8 chatbot_demo_v2\scripts\reindex.py --rollback :: 되돌리기
```

> ⚠ **골든셋 없이 검색 파라미터를 바꾸지 마세요.** 개선 착수 전 조사에서 유력해 보이던
> 4가지를 리랭커로 실측했더니 **3가지가 오히려 악화**였습니다(카탈로그 프리픽스 제거 ·
> 표 행단위 분할 · 질의 재작성). 어떤 설정 변경도 `run_eval.py` → `ab_compare.py` 로
> 판정한 뒤에만 채택하세요. 자세한 내용은 [2차_개선_보고서.md](2차_개선_보고서.md) §3.

> **PowerShell** 을 쓴다면 `set RUN_RAG_INTEGRATION=1` 대신 `$env:RUN_RAG_INTEGRATION="1"`,
> 주석은 `::` 대신 `#` 입니다. 나머지 명령은 동일합니다.
>
> `conda activate` 가 안 되면 **Anaconda Prompt** 에서 실행하거나 `conda init cmd.exe` 를 한 번 수행하세요.
> 그래도 안 되면 `python` 대신 절대경로
> `C:\Users\minsoo\anaconda3\envs\intern_chatbot\python.exe` 를 그대로 써도 됩니다.

### 전제 서비스
| 대상 | 필요 시점 | 없으면 |
|---|---|---|
| **Ollama**(임베딩 `embeddinggemma`) | RAG 검색 | RAG 경로만 503, 시나리오·FAQ는 정상 |
| **GEMINI_API_KEY**(최상위 `.env`) | RAG 생성·검증, composer/grader | 소형 LLM 노드는 pass-through(원문 답변 유지) |
| GPU | 리랭커(bge-reranker-v2-m3) | CPU 폴백(느림) |

---

## 2. 그래프 구조

> 그림·실측 트레이스 해설은 **[docs/파이프라인_비교.md](docs/파이프라인_비교.md)** 에 있습니다
> (v1 DAG · v2 사이클/HITL/서브그래프 · LangSmith 실행 타임라인 3장).

### 메인 그래프 (노드 14개)
```
START → normalize_input → load_or_update_session
  ├(버튼)→ scenario_action_handler ─────────────────→ route_decider
  └(자유입력)→ contextualize_query → scenario_matcher → route_decider

route_decider ─┬─ "scenario"(버튼) ─→ scenario_answer ─→ final_formatter   (합성 안 함)
               ├─ "faq"(유사도 통과) → scenario_answer ─→ compose_answer
               ├─ "clarify"(애매+고점) → clarify_node ⟦interrupt⟧
               │        재개 시 Command(goto): 후보선택→scenario_answer / 해당없음→rag3x_answer
               └─ "rag3x" ───────────→ rag3x_answer → rag_result_evaluator
                                          ├→ web_search_answer ─┬ 범위 안 → 웹검색 → final_formatter
                                          │   (도메인 게이트)     └ 범위 밖/판정불가 → final_formatter(보류)
                                          ├→ compose_answer            (답변 있음)
                                          └→ final_formatter           (abstain)
compose_answer → answer_grader ─┬─ 미해결 & 예산>0 & FAQ → rag3x_answer   ⟲ 에스컬레이션 사이클
                                ├─ 미해결 & RAG & 웹검색ON → web_search_answer  (마지막 보루)
                                └─ 그 외 ───────────────→ final_formatter → END
```

### RAG 서브그래프 (`controller_x.py` S0~S8 재편성 · 노드 10개)
```
prepare → retrieve ─┬─ 리랭크 점수 바닥 → crag_rewrite ⟲ retrieve   (CRAG 사이클, 1회)
                    ├─ 근거없음 ───────→ finalize
                    └─ 근거있음 ───────→ grade_evidence ★
grade_evidence ─┬─ 남은 근거 있음 ──→ answer_node → verify_node
                ├─ 전부 무관 & 예산 → crag_rewrite ⟲ retrieve
                └─ 전부 무관 & 소진 → finalize
verify_node ─┬─ text 빈응답 ────→ rollback_top1   ┐
             ├─ 숫자 미지원 ────→ rollback_vision ├→ finalize → END
             ├─ 전사-OCR 불일치 → rollback_ocr    ┘
             └─ 정상 ──────────→ finalize
```

**★ `grade_evidence`** (2026-07-27 신설) — 회수된 페이지를 LLM 1회로 3등급 판정합니다.

| 등급 | 뜻 | 컨텍스트 대우 |
|---|---|---|
| `primary` | 이 근거만으로 답할 수 있음 | 페이지 전문 |
| `supporting` | 주제는 맞으나 핵심은 없음 | 검색이 고른 청크만 축약 |
| `irrelevant` | 질문과 다른 상황 | 제외 |

- `primary` 가 0개여도 버리지 않고 **최상위 `supporting` 을 승격**합니다(판정 오류 안전장치).
- 전부 `irrelevant` 면 CRAG 재질의로 갑니다 → CRAG 가 "리랭크 점수가 바닥일 때"가 아니라
  **"의미상 근거가 없을 때"** 발동하게 됩니다.
- 이 노드가 붙어 무관 페이지가 걸러지므로 회수를 `final_pages 3 → 6` 으로 넓혔습니다.

노드 본문은 vendored 원시함수(`run_retrieval`/`answer_text_from_pages_x`/`verify_answer`/
`rewrite_query`/`_finalize`)만 호출합니다 — **rag3 로직 무수정**(예외 1건: `verify.is_abstain`,
아래 §3 참조). 등가성은 원본 `Rag3xEngine.ask()`와 비교하는 통합 테스트로 검증했습니다.

### LangGraph 기능 사용처
| 기능 | 사용처 |
|---|---|
| **사이클** | RAG `crag_rewrite ⟲ retrieve`(1회) / 메인 `answer_grader ⟲ rag3x_answer`(1회) |
| **HITL 인터럽트** | `clarify_node`의 `interrupt()` → API가 `Command(resume={"choice"})`로 재개 |
| **체크포인터 메모리** | `messages: Annotated[list, add_messages]` + `InMemorySaver` + 세션별 thread epoch |
| **서브그래프** | RAG 파이프라인(자체 `RagState`) |
| **커스텀 스트리밍** | `get_stream_writer()` → SSE `progress` 이벤트 |
| **`Command(goto=)`** | clarify 재개 후 동적 분기 |
| **조건부 엣지** | 6곳(입력종류/라우팅/검색후/검증후/합성후/평가후) |
| **`RetryPolicy`** | RAG 서브그래프의 `retrieve`/`answer_node`/`verify_node`(외부 백엔드 호출) |
| **`draw_mermaid()`** | `/api/health`의 `graph_mermaid` |

> **RetryPolicy를 메인그래프 LLM 노드에 붙이지 않은 이유**: `LlmHelper`가 예외를 삼키고
> `None`을 반환해 pass-through 하므로 재시도가 발동하지 않습니다. Gemini 429/5xx 지수백오프는
> `GeminiBackend` 내부에 이미 있습니다. (형식적 no-op 대신 실제 의미 있는 곳에만 적용)

---

## 3. 핵심 동작 원칙

### 결정론 경로는 LLM 0회
시나리오 버튼 이동·종단답변, FAQ 정확일치 매칭은 LLM을 쓰지 않습니다.
신규 LLM 노드(contextualize/composer/grader)는 **전부 `.env` 토글 + 실패 시 pass-through**입니다.
- **시나리오 버튼 종단답변은 합성하지 않습니다** — 절차 안내 왜곡 방지.
- FAQ 답변은 합성하되(토글) **원문을 항상 `original_answer`로 동봉**해 UI에서 "원문 보기"로 확인 가능.

### 답변 합성(composer)의 환각 방어
합성 후 `rag3.verify.check_claims_supported`(**LLM 0회** 결정론 숫자/코드 대조)로
근거에 없는 수치가 생기면 **합성을 폐기하고 원문/초안으로 복귀**하며 `composer_fallback`에 사유를 남깁니다.
검증 컨텍스트는 `근거 + 초안 + 프롬프트 원본 템플릿`입니다 — 템플릿을 포함하지 않으면
프롬프트가 지시한 안내 상수(지원센터 `1899-0979`)를 모델이 인용했을 때 오탐으로 정상 합성이 폐기됩니다(실측 확인).

**실측 예시** (RAG 경로)
- 초안: `무선 AP는 가온, 다보링크, 올레디오, 대유플러스 총 4개 제조사로 구축되어 있습니다.`
- 합성: `학교에 구축된 무선 AP는 … 총 4개 제조사의 장비로 구성되어 있습니다. '24년 7월 기준으로 총 89,464대가
  구축되어 있으며, 제조사별 구축 수량은 다보링크, 가온, 올레디오, 대유플러스 순입니다.`
  → 초안이 빠뜨린 근거(89,464대·순위)를 종합했고 해당 수치는 근거에 실재해 검증 통과.

### 근거 3등급 + 등급별 예산 (2026-07-27 신설)
검색은 청크 단위로 하고 답변은 페이지 전문으로 되돌리는 구조라, 예산이 **선착순**으로 소진되며
관련도와 무관하게 배분됐습니다. 실측(LangSmith `8e0815cd`)에서 컨텍스트 5,949자 중 **정답 근거는
9.6%**, 무관한 3순위 페이지가 **64.6%**, 한 문서가 **90.1%** 를 독점했습니다.

`grade_evidence` 등급에 따라 예산을 배분합니다.
- `primary` → 페이지 전문(장당 상한 4,000자)
- `supporting` → 매칭된 청크만, 장당 800자 · **합계는 primary 분량의 50%** (절대 상한 2,000자)
  - 상대 예산이 중요합니다. 절대 상한만 두면 primary 가 574자일 때 supporting 2,000자가
    정답을 묻어버립니다(실측: `final_pages` 확대 후 정답 비중 61% → 45% 희석).
- 한 문서가 supporting 예산의 60% 를 넘지 못합니다(문서 다양성).

### abstain 판정 — vendoring 무수정 원칙의 유일한 예외
`ragcore/rag3/verify.py` 의 회피 마커에 `"제공된 근거"` 가 있어, **정상 답변이
"제공된 근거에 따르면…" 으로 시작한다는 이유로 회피로 오판**됐습니다. 피해는 세 겹이었습니다.
① groundedness 검증이 통째로 스킵되고 ② 30.4초짜리 롤백이 발동한 뒤 같은 오탐으로 폐기되고
③ 멀쩡한 답변에 "신뢰도 낮음" 경고가 붙었습니다.

→ 마커를 `"제공된 근거에서 확인"`(프롬프트가 지시한 회피 문구)으로 좁히고, 마커가 있어도
답변이 160자를 넘으면 '부분 유보'로 보는 길이 가드를 넣었습니다.

### 무한루프 방지
모든 사이클에 예산이 있습니다 — CRAG `crag_budget=1`, 롤백은 전용 노드 단발(A/B/C 각 1회),
에스컬레이션 `escalate_budget=1`. 여기에 rag3 원본의 모델호출 상한(<5)·deadline이 그대로 적용됩니다.

**CRAG 재작성 안전가드** — 재작성이 항상 나은 게 아닙니다. 실측에서 재작성 질의는 정답 근거를
1위 → 7위로 악화시켰는데, 서브그래프는 결과를 무조건 덮어쓰고 있었습니다. 이제 재작성 결과의
리랭크 점수가 더 낮거나 근거를 못 찾으면 **원본 검색 결과를 유지**하고 사유를 `history` 에 남깁니다.

---

## 4. API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 웹 UI |
| GET | `/api/health` | 상태 + 토글 + `graph_mermaid` |
| GET | `/api/scenarios/root` | 초기 시나리오 버튼 |
| POST | `/api/chat` | 단발 응답(`message` \| `action` \| `clarify_response` 중 하나) |
| POST | `/api/chat/stream` | **SSE 스트리밍**(권장 — 프론트가 사용) |
| POST | `/api/feedback` | 👍/👎 → LangSmith `create_feedback` |
| POST | `/api/reset` | 세션 초기화(새 thread epoch) |
| POST | `/api/warmup` | RAG 엔진 예열(백그라운드) |
| GET | `/evidence/{run_id}/{file}` | 근거 이미지(경로 3중 검증) |

### SSE 이벤트
| 이벤트 | payload | 의미 |
|---|---|---|
| `progress` | `{stage, msg}` | 진행상황(검색/생성/검증/롤백/합성/판정…) |
| `node` | `{node}` | 완료된 그래프 노드 |
| `clarify` | `{session_id, candidates}` | HITL 되묻기로 일시정지 |
| `final` | ChatResponse 전체 | 최종 응답 |
| `error` | `{detail, status}` | 429/503/500 |

**실측 스트리밍**(RAG 질의, 34초): `내부 자료 검색 → 근거 문서 검색·리랭킹 → 근거 3페이지로 답변 생성 →
답변이 근거와 맞는지 검증 → 답변이 부실해 1순위 근거로 다시 시도(롤백 발동) → 근거 종합 정리`

오류: 빈 질문/잘못된 action 400, RAG 동시요청 429, 엔진 미가용 503 (응답에 키·절대경로 미포함).

---

## 5. 환경변수

로딩 우선순위: **프로세스 env > `chatbot_demo_v2/.env` > 최상위 `.env` > 코드 기본값**

| 변수 | 기본값 | 설명 |
|---|---|---|
| `RAG_BACKEND` | `gemini` | `gemini` \| `ollama` |
| `RAGCORE_CONFIG` | `ragcore/rag3/config.yaml` | vendored 설정(경로만 v2화) |
| `RAG_DEEP_WARMUP` | **`true`** | 기동 시 임베딩·LLM까지 예열 → 첫 질문 콜드로드(23s) 제거 |
| `SCENARIO_MATCH_BACKEND` | **`semantic`** | `semantic`(임베딩+리랭커) \| `fuzz`(문자 편집거리) |
| `SCENARIO_MATCH_THRESHOLD` / `_MARGIN` | **`0.80` / `0.30`** | FAQ 의미 유사도 채택 기준 |
| `CLARIFY_ENABLED` / `CLARIFY_MIN_SCORE` | `true` / **`0.35`** | 애매할 때 되묻기(HITL) |
| `COMPOSER_RAG_ENABLED` / `COMPOSER_FAQ_ENABLED` | `true` / `true` | 답변 종합·정리 |
| `CONTEXTUALIZE_ENABLED` | `true` | 후속질문 재작성 |
| `GRADER_ENABLED` | `true` | 해결도 판정 (FAQ=에스컬레이션 / RAG=경고만) |
| `RAG_CACHE_TTL_S` | `3600` | 같은 질문 재요청 시 즉답(0=비활성) |
| `WEB_SEARCH_ENABLED` / `_SCOPE` | **`false`** / `in_domain_unresolved` | 마지막 보루 웹검색. `_SCOPE=in_domain_unresolved` 면 도메인 게이트 통과 질문만 |
| `WEB_SEARCH_PROVIDER` / `_MODEL` | `gemini_grounding` / `gemini-3.1-flash-lite` | Gemini + Google 검색 grounding (`mock`/`disabled` 선택 가능) |
| `WEB_SEARCH_DAILY_BUDGET` / `_MAX_SOURCES` / `_TIMEOUT_S` | `100` / `5` / `30` | 하루 호출 상한(과금 방지) · 출처 표시 수 · 타임아웃 |
| `WEB_SEARCH_GEMINI_API_KEY` | — | **웹검색 전용 키**. 검색 grounding 은 결제(유료) 티어에서만 동작하므로 무료 `GEMINI_API_KEY`(RAG용)와 분리해 쓴다. 비우면 `GEMINI_API_KEY` 로 폴백 |
| `LANGSMITH_TRACING` / `_API_KEY` / `_PROJECT` | `false` / — / `…-v2` | 추적·피드백 |
| `DEMO_PORT` | `8002` | v1(8001)과 병행 |

> ⚠ **FAQ 매칭 임계는 스케일 의존적입니다.** `semantic`(크로스인코더)과 `fuzz`(문자 편집거리)는
> 점수 분포가 완전히 달라 임계값이 호환되지 않습니다(semantic 0.80/0.30/0.35 ↔ fuzz 0.90/0.05/0.75).
> `data/faq_embeddings.json` 이 없으면 매처가 자동으로 `fuzz` 로 폴백하면서 **임계도 함께 되돌립니다**.
> 임계를 직접 조정할 때는 `scripts/calibrate_faq_threshold.py` 로 분포를 먼저 확인하세요.
>
> 실측 요약: 모호 질문은 점수가 높아도(best 0.971) **margin(1·2위 차)이 0.272 이하**라
> margin 이 핵심 판별자입니다. 범위 밖 질문은 best 최대 0.004 로 확실히 분리됩니다.

---

## 6. 프롬프트 수정 (재시작 불필요)

`prompts/*.md`를 수정하면 **다음 호출부터 즉시 반영**됩니다(mtime 핫리로드).
`string.Template`의 `$변수`를 쓰므로 본문에 JSON 중괄호가 있어도 안전합니다.

| 파일 | 변수 | 용도 |
|---|---|---|
| `composer_rag.md` | `$question $answer $evidence_text $history_summary` | RAG 근거 종합 |
| `composer_faq.md` | `$question $original_answer $history_summary` | FAQ 원문 재구성 |
| `contextualize.md` | `$history $question` | 후속질문 → 독립 질문 |
| `answer_grader.md` | `$question $answer` | RESOLVED/UNRESOLVED |
| `clarify.md` | `$candidates` | 되묻기 안내 문구 |
| `evidence_grader.md` | `$question $candidates` | **근거 3등급 판정**(primary/supporting/irrelevant) |

> `evidence_grader.md` 는 RAG 서브그래프가 씁니다. 응답 형식은 `1=primary` 처럼 한 줄에 하나이며,
> 파싱에 실패한 항목은 **`supporting` 으로 안전하게 유지**됩니다(판정 실패가 근거 유실로 이어지지 않게).
> 프롬프트에 "용어가 겹친다는 이유로 primary 를 주지 마라"는 지시가 들어 있는데, 이는 실측에서
> 리랭커가 "AP 설치"라는 표현만 겹치는 장애조치표를 1순위로 올렸기 때문입니다.

---

## 7. LangSmith에서 보이는 것

`LANGSMITH_TRACING=true` + 키 설정 시, `chat_turn` 아래에 노드가 child run으로 기록됩니다.
v1과 달리 **RAG 내부(retrieve/crag/answer/verify/rollback/finalize)가 개별 run으로 보입니다**.

```
chat_turn
├ normalize_input · load_or_update_session · contextualize_query · scenario_matcher · route_decider
├ rag3x_answer
│  └ rag3x.ask
│     └ rag_subgraph            ← compile(name=...) 로 이름을 준 서브그래프
│        ├ prepare · retrieve · after_retrieve
│        ├ grade_evidence · after_grade   ← 2026-07-27 신설(근거 3등급 판정)
│        ├ crag_rewrite · retrieve        (사이클이 돌면 retrieve 가 2번 찍힘)
│        ├ answer_node · verify_node · after_verify · rollback_*
│        └ finalize
├ rag_result_evaluator · compose_answer · answer_grader
└ final_formatter
```
실제 트레이스 해석은 [docs/파이프라인_비교.md §4](docs/파이프라인_비교.md) 참고.
- 노드별 metadata: 매칭 점수·라우팅 근거·리랭크 점수·검증 결과·합성 폐기 사유·프롬프트 파일/mtime
- 태그: `route:*`, `match:*`, `composed:ok|fallback`, `grader:*`, `clarify_resolved:*`, `turn_route:*`
- 👍/👎는 턴 `run_id`에 `user_score` 피드백으로 기록

**결과 dict 에 실리는 관측용 필드** (2026-07-27 추가)
- `evidence_grades` — 회수된 페이지별 `{document_name, page_number, page_score, grade}`.
  어떤 근거가 왜 빠졌는지 사후에 확인할 수 있습니다.
- `context_pages_used` — 실제 답변 컨텍스트에 들어간 페이지와 **글자 수·등급·승격 여부**.
  근거 정밀도가 낮게 나올 때 어디서 예산이 샜는지 바로 보입니다.
- `rollback_history` 에 `crag_rewrite_accepted` / `crag_rewrite_rejected` 가 남아
  재작성 결과를 채택했는지 기각했는지 알 수 있습니다.
- `metrics.gemini_tokens_think` — 사고 토큰. 이전에는 집계에서 누락돼 비용이 과소계상됐습니다.

**공개 트레이스를 코드로 내려받기** — 이번 진단에 쓴 방법입니다.
```bash
curl -X POST "https://api.smith.langchain.com/public/<share-id>/runs/query" \
     -H "Content-Type: application/json" -d '{"trace":"<trace-id>","limit":200}'
curl "https://api.smith.langchain.com/public/<share-id>/run/<run-id>"   # 개별 run 전문
```
루트 run 의 `child_run_ids` 로 자식들을 순회하면 트리 전체를 얻을 수 있습니다.

---

## 8. 데이터

### 코퍼스
13문서·969페이지·2,562청크(MinerU 파싱 캐시 + 청크 색인). `ragdata/`에 복사본 보유(590MB, gitignore).
스캔 페이지 199개(20.5%) · 표 포함 페이지 326개(33.6%).

**색인 위생** (2026-07-27, `ragcore/rag3/chunk_hygiene.py`) — 서빙 색인을 직접 열어 실측한 결과
완전중복 138개(5.4%), 줄 반복 노이즈 65개(2.5%), 임베딩 컨텍스트(2,048토큰) 초과 151개(5.9%)가
있었고 초과분은 **조용히 잘린 채** 색인돼 있었습니다. 노이즈 청크가 가장 길다는 점이 문제였습니다
(9,587자×5, 12,775자×10 — 웹 UI 스크린샷 OCR 에서 브라우저 북마크바와 887개 학교명 세로 나열).
`ingest` 가 flat 색인을 만들기 직전에 중복 제거 · 반복줄 압축 · 초과 경고를 수행합니다.

### 평가 데이터
| 파일 | 내용 |
|---|---|
| `data/eval_set.json` | 골든셋 37문항 — `faq_exact`(5) `faq_paraphrase`(10) `rag`(12) `ambiguous`(5) `out_of_scope`(5) |
| `data/faq_embeddings.json` | FAQ 236행 질문 임베딩(embeddinggemma, 2.5MB). 의미 매칭용 |

골든셋은 `expected_route` 외에 `expected_doc`/`expected_pages` 를 갖고 있어 **근거 정밀도**
(컨텍스트 중 정답 근거가 차지하는 비중)를 측정할 수 있습니다. FAQ 를 수정했다면
`build_faq_embeddings.py` 를 다시 실행하세요.

### FAQ 근거링크 (`scripts/build_faq_doc_links.py` → `runtime/reports/corpus_gap.md`)

FAQ 236행의 `source_files`("문서명 N쪽")를 파싱해 코퍼스 문서와 연결합니다. 현재 **150/236**:

| 상태 | 행수 | UI 표시 |
|---|---:|---|
| 근거 **페이지**까지 확보 | 34 | 근거 이미지 |
| 근거 **문서**만 확보(쪽 미인용) | 116 | "근거 문서: …(쪽 미인용)" |
| 미연결 | 86 | 인용 표기만 |

**핵심**: 근거 이미지가 34행에 그치는 주된 원인은 코퍼스 격차가 아니라 **FAQ 원본의 인용 방식**입니다.
236행 중 **199행이 쪽번호 없이 문서명만** 인용합니다. 코퍼스에 없는 4종(`★05__FAQ.docx` 47행,
`학내전산망 설치 기준(2023.6)` 19행, `★홈페이지_쳇봇문의_Q_A.pdf` 15행, `학교 무선 인프라 관리활용 가이드(세종)` 13행)의
참조 **94건도 전부 쪽번호가 없습니다** → 이들을 추가 색인해도 **근거 이미지는 늘지 않습니다**.
(색인 추가의 가치는 'RAG 검색 대상 확대'로 별개이며, `학내전산망 설치 기준`·`세종` 원본은 `데이터 카탈로그/`에 존재합니다.)

문서명 매칭은 **완전일치 → 포함관계 → 유사매칭(partial_ratio ≥75 ∧ 2위와 격차 ≥15)** 순이며,
포함·유사 단계 모두 **동점이면 거절**합니다(오매칭 방지). 유사매칭으로 연결한 건은
`corpus_gap.md`의 감사표에 점수·격차와 함께 기록됩니다 — 예: FAQ `스쿨넷서비스 학내망 구축 및 운영·관리 가이드(2018.12)`
↔ 코퍼스 `8-1. … 운영관리 **개선을 위한** 가이드.pdf` (78점, 격차 23 · 1페이지 제목으로 동일 문서 확인).

> 파싱 품질 자체는 재작업이 불필요하다고 판단했습니다 — MinerU가 bbox·의미태그를 보존하고
> figure 페이지 117개 중 115개를 vlm-engine으로 텍스트화 병합했으며, 잔여 "래스터 표 파싱천장" 5문항은
> rag3x(Gemini)에서 4/5가 이미 개방됐습니다.

---

## 9. 구조

```
chatbot_demo_v2/
  config/settings.py          설정(dotenv 우선순위 + 토글)
  prompts/                    외부화 프롬프트 6종 + loader(핫리로드)
  graph/
    state.py                  ChatState(+messages) / RagState
    routing.py                순수 분기 함수(테스트 용이)
    nodes.py                  메인그래프 노드 14개
    rag_nodes.py              RAG 서브그래프 노드 10개(+grade_evidence·등급별 예산)
    builder.py                메인그래프 + RAG 서브그래프 조립
  ragcore/rag3, rag3x         vendored 엔진(예외 2건: verify.is_abstain, chunk_hygiene 신설)
    rag3/chunk_hygiene.py     색인 직전 청크 위생(중복·노이즈·길이)
  ragdata/index, parsed_v25   복사 데이터(gitignore, bootstrap으로 재생성)
  rag/
    adapter_util.py           SubgraphRagAdapter(+TTL캐시·락·근거사본) / Fake / 유틸
    llm_helper.py             소형 LLM 전용 채팅 백엔드(RAG 엔진과 분리)
  scenario/                   FAQ·시나리오 트리·매처(Semantic + fuzz 폴백)
  observability/langsmith.py  추적·metadata·피드백
  app/                        FastAPI(main/api/dependencies/schemas)
  static/                     대화 UI + 근거·파이프라인 인스펙터(SSE 소비, [p53] 출처 칩)
  docs/                       파이프라인_비교.md + 다이어그램 3장(SVG)
                              개선작업_진행상황.md
  data/                       faq · scenarios · faq_doc_links
                              eval_set(골든셋 37) · faq_embeddings(의미 매칭)
  _backup/                    원복 스냅샷 + RESTORE.md (gitignore)
  scripts/                    bootstrap_data · build_faq_doc_links · render_pipeline
                              run_eval · ab_compare · reindex
                              build_faq_embeddings · calibrate_faq_threshold
                              run_demo.cmd · share_tunnel.cmd (시연용)
  tests/                      147개(+통합 1, 기본 skip)
```

---

## 10. v1 대비 달라진 점

| 항목 | v1 | v2 |
|---|---|---|
| 그래프 | 전진 DAG(사이클 0) | 사이클 2종 + HITL + 서브그래프 |
| RAG | 블랙박스 `ask()` 1노드 | 9노드 서브그래프(내부 관측·스트리밍) |
| 대화 | 턴 독립(이력 없음) | `messages` 메모리 + 후속질문 재작성 |
| 애매한 질문 | 곧바로 RAG(25~150초) | **되묻고** 결정론 경로로 착지 |
| 답변 | DB 원문 낭독 | 근거 종합·정리(+원문 동봉, 환각 대조) |
| 응답 | 단발(무피드백) | SSE 진행상황 + 👍👎 |
| 독립성 | test_3 참조 | 코드·데이터 자체 보유 |

## 11. 후속 실험 후보 (미구현)

검토했으나 이번 범위에서 제외한 것들입니다. 근거와 예상 효과는
[2차_개선_보고서.md](2차_개선_보고서.md) §8 에 정리돼 있습니다.

| 후보 | 기대 효과 | 왜 안 했나 |
|---|---|---|
| **Contextual Retrieval**(Anthropic) | 청크별 LLM 컨텍스트 부착 → top-20 실패율 67% 감소 보고 | 오프라인 2,562콜 + 재색인. 현재 프리픽스가 문서 공통이라 청크 변별에 기여 0인 점은 확인됨 |
| **임베딩 모델 교체** | embeddinggemma → BGE-M3(MTEB 63.0)/Qwen3-Embedding(70.6). 컨텍스트도 넓어 5.9% 잘림 해소 | 재색인 + 골든셋 A/B 필수 |
| **ColPali/ColQwen 시각 검색** | 스캔 199p·표 326p(코퍼스 54%)를 OCR 없이 페이지 이미지로 검색 | 새 GPU 모델·별도 색인·멀티벡터 검색기 필요(아키텍처 변경) |
| **답변 토큰 스트리밍** | 체감 지연 감소(실제 속도는 동일) | 우선순위 밖 |
| ~~실제 웹검색 provider~~ | 코퍼스 밖 질문 대응 | **구현됨** — Gemini Grounding(§ `WEB_SEARCH_*`). 기본은 꺼져 있고, 켜면 검색 호출당 과금 |
| 영속 체크포인터(SQLite) | 서버 재시작 후 대화 유지 | 현재 "새로고침 전까지만 기억"이 의도된 사양 |
| `Send` 팬아웃 병렬화 | 분해검색 하위질문 동시 실행 | GPU 단일 처리(`_ask_lock`)라 이득 제한적 |
| `content_list_v2` bbox 하이라이트 | 근거 이미지에서 인용 영역 강조 | 문장별 출처 표기(`[p53]` 칩)로 1차 대응함 |

## 12. 팀원에게 시연하기 (Cloudflare Quick Tunnel)

로컬에 뜬 데모를 임시 공개 URL로 감싸 링크만 전달하면 팀원이 바로 체험할 수 있습니다.
**터미널 2개**를 씁니다.

```bat
:: [터미널 A] 서버
cd /d "C:\Users\minsoo\Desktop\아이티지엔 인턴\챗봇"
chatbot_demo_v2\scripts\run_demo.cmd

:: [터미널 B] 터널  — 서버가 뜬 것을 확인한 뒤 실행
cd /d "C:\Users\minsoo\Desktop\아이티지엔 인턴\챗봇"
chatbot_demo_v2\scripts\share_tunnel.cmd
```

터미널 B에 아래 같은 주소가 뜨면 그 링크를 전달하면 됩니다.

```
+------------------------------------------------------------+
|  https://<임의문자열>.trycloudflare.com                      |
+------------------------------------------------------------+
```

스크립트를 쓰지 않고 직접 실행해도 동일합니다:
```bat
cloudflared tunnel --url http://127.0.0.1:8002
```

**시연 전 체크리스트**
1. `python -m chatbot_demo_v2 --port 8002` 로 서버가 떠 있고 `http://127.0.0.1:8002` 가 열리는가
2. **첫 질문 전에 예열** — 브라우저를 열어두거나 `curl -X POST http://127.0.0.1:8002/api/warmup` 실행.
   예열 없이 첫 RAG 질문을 던지면 엔진 초기화로 **30초가 더 걸립니다**(실측 트레이스 확인).
3. Ollama 가 떠 있는가(임베딩) · 최상위 `.env` 에 `GEMINI_API_KEY` 가 있는가
4. 시연 시나리오: 버튼 클릭(즉답) → FAQ 질문(근거 이미지) → 자유 질문(SSE 진행표시) → 후속 질문(기억 확인)

**검증됨**: 터널 너머에서도 `/api/health`·웹 UI·**SSE 스트리밍**(`event: progress`/`node`/`final`)이
정상 동작하는 것을 실기로 확인했습니다. Cloudflare 가 SSE 를 버퍼링하지 않도록
`Cache-Control: no-cache` + `X-Accel-Buffering: no` 헤더를 스트림 응답에 붙여 두었습니다.

**주의**
- Quick Tunnel 주소는 **임시**이며 터널을 종료하면 링크도 사라집니다. 고정 주소·인증이 필요하면
  Cloudflare 계정 기반 Named Tunnel 을 별도 설정해야 합니다.
- 터널 URL 은 **인증이 없습니다**. 아는 사람에게 짧게만 공유하세요.
- 동시 접속자가 RAG 질문을 겹쳐 던지면 뒤의 요청은 429 로 거절됩니다(GPU 단일 처리).
- `run_demo.cmd` / `share_tunnel.cmd` 는 **의도적으로 영문 메시지**입니다 — cmd.exe 가 배치파일을
  OEM 코드페이지로 읽어 한글이 섞이면 구문이 깨집니다(실측 확인).

---

## 13. 주의
- 인증 없음 — 외부 공개 시 짧게만(§12).
- 실험용 데모이며 상용 서비스가 아닙니다.
- RAG는 GPU 단일 처리라 동시 요청을 429로 제한합니다.
- 체크포인터가 `InMemorySaver` 라 **서버를 재시작하면 모든 대화가 초기화**됩니다(의도된 선택).

### 손대기 전에 읽을 것 (2026-07-27 추가)
- **골든셋 없이 검색 파라미터를 바꾸지 마세요.** 유력해 보이는 개선안이 실측에서 악화인 경우가
  흔합니다(착수 전 조사에서 4개 중 3개). `run_eval.py` → `ab_compare.py` 로 판정하세요.
- **재색인은 `scripts/reindex.py` 로만.** 기존 `ragdata/index` 를 직접 덮어쓰지 마세요.
  이 스크립트는 `index_new` 에 만든 뒤 검증 후 rename 으로 교체하고, `--rollback` 을 제공합니다.
- **`ragdata/` 와 `.env` 는 git 에 없습니다.** 복원 경로는 [_backup/RESTORE.md](_backup/RESTORE.md).
  `.env` 는 다른 사본이 없는 **유일본**입니다.
- **원본(`test_3/사전데이터`)은 읽기 전용입니다.** 여기가 `ragdata/` 의 최후 복구 원천이므로
  재색인 전후로 파일 수·크기를 대조하세요(`reindex.py` 가 자동 출력).
- **FAQ 를 수정하면 `build_faq_embeddings.py` 를 다시 실행**해야 의미 매칭에 반영됩니다.
- 진행 이력·다음 할 일은 [docs/개선작업_진행상황.md](docs/개선작업_진행상황.md).
