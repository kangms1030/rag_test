# chatbot_demo — 학교 유무선 장애상담 실험용 챗봇 데모

기존 `test_3/코드/rag3x` RAG 엔진, 엑셀 모범 질답(236행), PPT 시나리오 트리를
**LangGraph**로 연결하고 **LangSmith**로 각 단계를 관찰하는 독립 데모입니다.
상용 서비스가 아니라, 라우팅·관측 실험용 데모입니다.

- 클릭형 **시나리오**(트리) + 자유 입력 **모범 질답 유사도 매칭** + **rag3x**(RAG) + (구조만) **웹검색**
- FastAPI(포트 8001) + 정적 HTML/JS 프론트엔드
- 모든 라우팅은 LangGraph 노드가 결정(FastAPI에는 라우팅 if/else 없음)

> 이 폴더는 기존 프로젝트와 독립적입니다. `test_1/2/3`, `rag3x`, `webapp`,
> 최상위 파일·`.env`는 **읽기만** 하며 수정하지 않습니다. rag3x는 외부 모듈로 import합니다.

---

## 1. 요구 환경

- Windows, conda 환경 `intern_chatbot` (Python 3.11)
- rag3x/rag3 의존성(torch, sentence-transformers, chromadb, ollama 등)은 기존 환경 사용
- 이 데모가 추가로 요구하는 패키지(설치 완료): `langgraph`, `langsmith`, `pytest`, `httpx`
  - (기존 설치: `fastapi`, `uvicorn`, `openpyxl`, `rapidfuzz`, `python-dotenv`, `python-pptx`)

추가 설치가 필요할 때:

```
conda activate intern_chatbot
pip install langgraph langsmith pytest httpx
```

---

## 2. 실행

```
cd C:\Users\minsoo\Desktop\아이티지엔 인턴\챗봇
conda activate intern_chatbot
python -m chatbot_demo --port 8001
```

브라우저: http://127.0.0.1:8001

- 시나리오 버튼은 즉시 응답합니다.
- 자유 입력은 모범 질답과 엄격히 비교(임계값 0.90) 후, 통과하지 못하면 rag3x로 이동합니다.
- **rag3x 질의는 25~150초** 걸릴 수 있습니다. 좌측 "RAG 엔진 예열(warmup)"으로 미리 모델을 올릴 수 있습니다.

### rag3x(RAG) 사용 전제

- 최상위 `.env`의 `GEMINI_API_KEY` (백엔드 `gemini`일 때 필요) — 자동으로 읽습니다.
- 로컬 **Ollama** 서버 가동(임베딩용). 키/GPU/Ollama가 없어도 **시나리오·FAQ 기능은 정상 동작**하며,
  RAG 경로만 503으로 안전하게 실패합니다.

---

## 3. 환경변수

로딩 우선순위: **프로세스 env > `chatbot_demo/.env` > 최상위 `.env` > 코드 기본값**

`.env.example`을 복사해 `chatbot_demo/.env`를 만드세요. **실제 키는 `.env`에만** 두세요
(`.env`는 `.gitignore` 대상, `.env.example`은 커밋되므로 키를 넣지 마세요).

| 변수 | 기본값 | 설명 |
|---|---|---|
| `RAG3X_ROOT` | `..\test_3\코드` | rag3x/rag3 패키지 폴더(sys.path 추가) |
| `RAG3X_CONFIG` | `...\rag3\config.yaml` | rag3 설정 파일 |
| `RAG3X_BACKEND` | `gemini` | `gemini` \| `ollama` |
| `RAG3X_DEEP_WARMUP` | `false` | warmup 시 모델을 VRAM에 상주 |
| `SCENARIO_MATCH_THRESHOLD` | `0.90` | 유사도 채택 임계값 |
| `SCENARIO_MATCH_MARGIN` | `0.05` | 1~2위 점수 최소 차(애매성 방지) |
| `WEB_SEARCH_ENABLED` | `false` | 웹검색 활성화(기본 비활성) |
| `WEB_SEARCH_SCOPE` | `in_domain_unresolved` | `in_domain_unresolved` \| `any_unresolved` |
| `LANGSMITH_TRACING` | `false` | LangSmith 추적 |
| `LANGSMITH_API_KEY` | (빈값) | 키가 없으면 tracing 자동 비활성(경고만) |
| `LANGSMITH_PROJECT` | `school-network-chatbot-demo` | 프로젝트명 |
| `DEMO_PORT` | `8001` | 서버 포트 |

---

## 4. LangSmith 추적

키를 `chatbot_demo/.env`에 넣고 tracing을 켜면 LangGraph 실행과 각 노드가
child run으로 자동 기록됩니다(별도 코드 불필요).

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<smith.langchain.com 발급 키>
LANGSMITH_PROJECT=school-network-chatbot-demo
```

- 키가 없으면 앱은 정상 실행되고 "tracing disabled" 경고만 출력합니다(키 값은 로그에 남기지 않음).
- 태그: `chatbot_demo`, `langgraph`, `scenario`, `rag3x`, `web_search_disabled|enabled` (+ turn별 `turn_route:...`)
- turn 메타데이터: `session_id`, `scenario_id`, `current_node_id`, `route`, `route_reason`,
  `answer_source`, `scenario_match_score`, `confidence`, `answer_path`, `elapsed_seconds`, `rag_run_id`
- API 키·비밀번호·절대경로·환경변수 전체는 절대 기록하지 않습니다.

### 트레이스에서 무엇을 볼 수 있나

LangSmith 좌측 "Threads/Turns" 뷰는 턴별 입력/출력만 보여줍니다. **노드 트리를 보려면
각 `chat_turn`을 클릭해 Trace(Run) 뷰**로 들어가세요. 각 노드 run에 판단 근거가 metadata로 붙습니다:

- `scenario_matcher` — `match_decision`, `best_score`, `second_score`, `matched_id`, `matched_question`
- `route_decider` / `rag_result_evaluator` — 선택한 `route`와 근거
- `scenario_action_handler` — `from_node → to_node`, `option_id`, `terminal`
- `rag3x_answer` 하위 **`rag3x.ask`** child run — `rerank_top_score`, `confidence`, `answer_path`,
  `evidence_docs`(문서/페이지), `verification`, `metrics`(검색·생성 시간·모델 호출수). rag3x 내부가
  블랙박스로 보이지 않도록 핵심 결과를 이 run의 outputs로 노출합니다(절대경로 미포함).

프로젝트 확인: https://smith.langchain.com → 프로젝트 `school-network-chatbot-demo`

---

## 5. 데이터

### 엑셀 → faq.json 재생성

원본 엑셀을 수정한 뒤 다음을 실행하면 `data/faq.json`이 재생성됩니다(원본은 읽기 전용):

```
python -m chatbot_demo.scripts.import_excel
```

- 4개 시트(스쿨넷/학내망/무선망/유무선통합관제) = 236행
- 헤더 변형(`질문유형`/`질문 유형`), 스쿨넷 잉여 열, 유령 빈 행을 자동 처리합니다.

### 시나리오 트리(scenarios.json)

PPT `★학교 유무선 장애상담 AI 챗봇 시나리오_0720.pptx` 슬라이드 1~7의 흐름과
**종단 답변 원문 그대로** 작성되었습니다. 종단 답변은 LLM으로 재생성하지 않고 저장 문장을 반환합니다.
데이터 모델은 `answer_ref`(엑셀 시트/행)도 지원하므로, 원하면 종단 답변을 엑셀 답변으로 전환할 수 있습니다.

---

## 6. API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 웹 UI |
| GET | `/api/health` | 앱/엔진/LangSmith/웹검색/라우팅 상태(항상 200) |
| GET | `/api/scenarios/root` | 초기 시나리오 버튼 |
| POST | `/api/chat` | 대화(자유 입력 `message` 또는 버튼 `action` 중 하나) |
| POST | `/api/reset` | 세션 시나리오 상태 초기화 |
| POST | `/api/warmup` | rag3x 엔진 예열(백그라운드) |
| GET | `/evidence/{run_id}/{filename}` | 근거 이미지(안전 경로만) |

오류: 빈 질문/잘못된 action 400, RAG 동시 요청 429, 엔진 미가용 503, 내부 오류 500
(응답에 키·절대경로 미포함).

---

## 7. 테스트

실제 Gemini/Ollama/LangSmith 없이 실행 가능합니다(mock 어댑터 사용):

```
python -m pytest chatbot_demo/tests -q
```

rag3x 실제 호출은 별도 통합 스모크(수동)로 확인합니다.

---

## 8. Cloudflare Tunnel (외부 공개 시)

애플리케이션 코드에는 포함하지 않습니다. 별도로 실행하세요:

```
cloudflared tunnel --url http://127.0.0.1:8001
```

- Quick Tunnel 주소는 **임시 주소**이며, 터널을 종료하면 링크도 사라집니다.
- 고정 주소·인증이 필요하면 Cloudflare 계정과 별도 Tunnel 설정이 필요합니다.

### 공개 데모 주의사항

- 인증 기능 없음 — 짧게만 공유하세요.
- API 키를 프론트엔드에 노출하지 않습니다.
- 민감한 개인정보를 입력하지 마세요.
- RAG는 GPU 단일 처리라 동시 요청을 제한(429)합니다.
- **현재 체크포인터는 InMemorySaver라 서버를 재시작하면 모든 대화 상태가 초기화됩니다.**
- 실험용 데모입니다.

---

## 9. 구조

```
chatbot_demo/
  config/settings.py            설정(dotenv 우선순위)
  app/
    main.py                     FastAPI 앱 팩토리 + 예외 매핑
    api.py                      엔드포인트(검증 + graph.invoke)
    dependencies.py             AppContext(DI) + SessionRegistry
    schemas.py                  요청/응답 스키마
    graph/{state,routing,nodes,builder}.py   LangGraph
  scenario/{models,loader,tree,matcher}.py   시나리오/FAQ
  rag/rag3x_adapter.py          rag3x 어댑터(지연 초기화·락·근거 사본)
  web_search/{base,disabled,mock}.py         웹검색 provider
  observability/langsmith.py    LangSmith 연동
  scripts/import_excel.py       엑셀 → faq.json
  data/{faq.json,scenarios.json}
  static/{index.html,app.js,styles.css}
  tests/                        단위/통합 테스트(48개)
  runtime/evidence/             근거 이미지 사본(gitignore)
```
