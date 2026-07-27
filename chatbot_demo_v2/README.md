# chatbot_demo_v2 — LangGraph 적극 활용 학교 유무선 장애상담 챗봇

v1(`chatbot_demo`)이 LangGraph를 **전진 DAG(관측용 라우터)** 로만 쓰던 것을 재설계해,
**사이클 · HITL 인터럽트 · 대화 메모리 · 서브그래프 · 커스텀 스트리밍**을 실제로 사용하는 데모입니다.
기존 `test_3/코드`의 RAG 엔진(rag3/rag3x)은 **무수정 vendoring**해 그대로 재사용하되,
블랙박스 호출이 아니라 **LangGraph 서브그래프로 재편성**해 내부 단계를 관측·스트리밍합니다.

> 이 폴더는 **독립 동작**합니다. 코드(`ragcore/`)와 데이터(`ragdata/`)를 자체 보유하며
> 다른 폴더(`test_3`, `chatbot_demo` 등)를 읽기만 하고 수정하지 않습니다.

**함께 볼 문서**
- [최종_구축_보고서.md](최종_구축_보고서.md) — 계획 대비 이행 점검 · 검증 결과 · 설계 판단 근거
- [docs/파이프라인_비교.md](docs/파이프라인_비교.md) — v1 vs v2 파이프라인 그림 3장 + 실측 트레이스 해설

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
                                          ├→ web_search_answer → final_formatter
                                          ├→ compose_answer            (답변 있음)
                                          └→ final_formatter           (abstain)
compose_answer → answer_grader ─┬─ 미해결 & 예산>0 & FAQ → rag3x_answer   ⟲ 에스컬레이션 사이클
                                └─ 그 외 ───────────────→ final_formatter → END
```

### RAG 서브그래프 (`controller_x.py` S0~S8 재편성)
```
prepare → retrieve ─┬─ 근거없음 & 경계점수 → crag_rewrite ⟲ retrieve   (CRAG 사이클, 1회)
                    ├─ 근거없음 ─────────→ finalize
                    └─ 근거있음 ─────────→ answer_node → verify_node
verify_node ─┬─ text 빈응답 ────→ rollback_top1   ┐
             ├─ 숫자 미지원 ────→ rollback_vision ├→ finalize → END
             ├─ 전사-OCR 불일치 → rollback_ocr    ┘
             └─ 정상 ──────────→ finalize
```
노드 본문은 vendored 원시함수(`run_retrieval`/`answer_text_from_pages_x`/`verify_answer`/
`rewrite_query`/`_finalize`)만 호출합니다 — **rag3 로직 무수정**.
등가성은 원본 `Rag3xEngine.ask()`와 비교하는 통합 테스트로 검증했습니다.

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

### 무한루프 방지
모든 사이클에 예산이 있습니다 — CRAG `crag_budget=1`, 롤백은 전용 노드 단발(A/B/C 각 1회),
에스컬레이션 `escalate_budget=1`. 여기에 rag3 원본의 모델호출 상한(<5)·deadline이 그대로 적용됩니다.

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
| `SCENARIO_MATCH_THRESHOLD` / `_MARGIN` | `0.90` / `0.05` | FAQ 유사도 채택 기준 |
| `CLARIFY_ENABLED` / `CLARIFY_MIN_SCORE` | `true` / `0.75` | 애매할 때 되묻기(HITL) |
| `COMPOSER_RAG_ENABLED` / `COMPOSER_FAQ_ENABLED` | `true` / `true` | 답변 종합·정리 |
| `CONTEXTUALIZE_ENABLED` | `true` | 후속질문 재작성 |
| `GRADER_ENABLED` | `true` | 해결도 판정 + 에스컬레이션 |
| `RAG_CACHE_TTL_S` | `3600` | 같은 질문 재요청 시 즉답(0=비활성) |
| `WEB_SEARCH_ENABLED` / `_SCOPE` | `false` / `in_domain_unresolved` | 웹검색(구조만) |
| `LANGSMITH_TRACING` / `_API_KEY` / `_PROJECT` | `false` / — / `…-v2` | 추적·피드백 |
| `DEMO_PORT` | `8002` | v1(8001)과 병행 |

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

---

## 8. 데이터

### 코퍼스
13문서·969페이지(MinerU 파싱 캐시 + 청크 색인). `ragdata/`에 복사본 보유(590MB, gitignore).

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
  prompts/                    외부화 프롬프트 5종 + loader(핫리로드)
  graph/
    state.py                  ChatState(+messages) / RagState
    routing.py                순수 분기 함수(테스트 용이)
    nodes.py                  메인그래프 노드 14개
    rag_nodes.py              RAG 서브그래프 노드 9개
    builder.py                메인그래프 + RAG 서브그래프 조립
  ragcore/rag3, rag3x         vendored 엔진(무수정 — config.yaml 경로만 v2화)
  ragdata/index, parsed_v25   복사 데이터(gitignore, bootstrap으로 재생성)
  rag/
    adapter_util.py           SubgraphRagAdapter(+TTL캐시·락·근거사본) / Fake / 유틸
    llm_helper.py             소형 LLM 전용 채팅 백엔드(RAG 엔진과 분리)
  scenario/                   FAQ·시나리오 트리·유사도 매처
  observability/langsmith.py  추적·metadata·피드백
  app/                        FastAPI(main/api/dependencies/schemas)
  static/                     대화 UI + 근거·파이프라인 인스펙터(SSE 소비)
  docs/                       파이프라인_비교.md + 다이어그램 3장(SVG)
  scripts/                    bootstrap_data · build_faq_doc_links · render_pipeline
                              run_demo.cmd · share_tunnel.cmd (시연용)
  tests/                      111개(+통합 1, 기본 skip)
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
- `Send` 팬아웃으로 분해검색 하위질문 병렬화 — 현재 GPU 단일 처리(`_ask_lock`)라 이득 제한적
- `content_list_v2`의 bbox를 이용한 근거 이미지 하이라이트
- 코퍼스 갭 문서 증분 색인(§8)
- 영속 체크포인터(SQLite)로 서버 재시작 후 대화 유지 — 현재는 재시작 시 초기화(의도된 선택)

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
