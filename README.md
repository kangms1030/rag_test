# 챗봇 RAG 모듈 — 저장소 안내

학교 데이터(13 PDF · 969페이지)를 근거로 답하는 챗봇용 RAG 모듈의 3세대 구축 기록이다.
각 세대는 **코드 / 결과물 / 사전데이터** 3버킷 + `final.md`(파이프라인 상세)로 정리돼 있다.

## 구조

```
챗봇/
├── test_1/                     1세대 — 쿼리타임 VLM 멀티모달 RAG
│   ├── 코드/                    rag_catalog_experiment (패키지; 내부 index·cache 포함)
│   ├── 사전데이터/               13 PDF · 카탈로그 xlsx · eval셋 · 원본 노트북
│   ├── 결과물/                   REPORT·results·outputs · test12 비교실험(test1 런)
│   └── final.md
├── test_2/                     2세대 — MinerU 사전파싱 텍스트 RAG
│   ├── 코드/                    rag2 (패키지; MinerU 파싱캐시·Chroma 인덱스 포함)
│   ├── 사전데이터/               13 PDF · 카탈로그 · eval셋(16Q/3Q/20Q)
│   ├── 결과물/                   rag2 outputs · test_2_timecost · test12 비교 · vlm_probe · 통합_실험_보고서
│   ├── final.md
│   └── CHANGES_from_test1.md
├── test_3/                     3세대 — 청크색인+리랭커+검증/롤백 RAG (+Gemini 실험)
│   ├── 코드/                    rag3(로컬) · rag3x(Gemini 포크) · webapp · probes · ask_cli
│   ├── 사전데이터/               13 PDF · 카탈로그 · synth_eval · corpus 덤프
│   ├── 결과물/                   probes_results(PHASE0~6) · rag3 outputs · 설계·config·Gemini 보고서 · scripts
│   ├── final.md
│   └── CHANGES_from_test2.md
├── chatbot_demo/               [데모 v1] 장애상담 챗봇 — LangGraph 전진 DAG
│   ├── app/                    FastAPI 웹 API · LangGraph 라우팅
│   ├── scenario/               PPT 시나리오 트리 · 엑셀 FAQ 유사도 매칭
│   ├── rag/                    rag3x RAG 엔진 연동 어댑터(블랙박스 호출)
│   ├── observability/          LangSmith 트레이싱 연동
│   └── static/                 정적 웹 UI (HTML/JS/CSS)
├── chatbot_demo_v2/            [데모 v2] 같은 챗봇의 LangGraph 적극 활용 재설계 — 독립 실행
│   ├── graph/                  메인그래프 14노드 + RAG 서브그래프 9노드 (사이클·HITL·메모리)
│   ├── prompts/                외부화 프롬프트 5종 (핫리로드)
│   ├── ragcore/                vendored rag3 · rag3x (무수정 — config 경로만 v2화)
│   ├── ragdata/                색인·파싱캐시 복사본 (bootstrap 으로 생성, git 제외)
│   ├── rag/ app/ static/       서브그래프 어댑터 · FastAPI(SSE) · 웹 UI
│   ├── docs/                   파이프라인 다이어그램 3장 + v1 대비 비교
│   └── 최종_구축_보고서.md       계획 대비 이행 점검 · 실측 검증 결과
├── 최종_결과_보고서.md           3세대 종합 보고서
├── .gitignore
├── .env                        GEMINI_API_KEY (git 제외)
└── CLAUDE.md                   프로젝트 실행 규칙
```

## 버킷 규칙

- **코드/**: 실행 RAG 패키지 통째(내부 빌드 산출 `index/`·`cache/` 포함, 실행상태 보존).
- **사전데이터/**: 원천 입력(PDF·카탈로그 Excel·평가 질문셋·원본 참고 노트북).
- **결과물/**: 보고서(.md/.docx)·평가결과 JSON·근거이미지·외부 비교실험 결과.

## 챗봇 데모 (RAG 모듈의 응용)

3세대 RAG(`test_3`)를 실제 상담 챗봇으로 감싼 실험용 데모다. **v1과 v2는 서로 독립 실행**되며,
포트가 달라(8001 / 8002) 동시에 띄워 비교할 수 있다.

| | chatbot_demo (v1) | chatbot_demo_v2 |
|---|---|---|
| LangGraph 사용 | 전진 DAG — 관측 가능한 라우터 | **사이클 3종 · HITL 인터럽트 · 대화메모리 · 서브그래프** |
| RAG | `Rag3xEngine.ask()` 1노드 블랙박스 | **9노드 서브그래프**(내부 단계 관측·스트리밍) |
| 답변 | DB 원문 낭독 | **근거 종합·재구성** + 결정론 환각가드 |
| 응답 | 단발 | **SSE 진행표시** + 👍👎 피드백 |
| 데이터 | `test_3` 참조 | 코드·데이터 자체 보유(독립) |
| 실행 | `python -m chatbot_demo --port 8001` | `python -m chatbot_demo_v2 --port 8002` |

> v2는 `ragcore/`(vendored rag3·rag3x)와 `ragdata/`(색인·파싱캐시 590MB)를 자체 보유한다.
> `ragdata/`는 git에서 제외되며 `python chatbot_demo_v2\scripts\bootstrap_data.py` 로 재생성한다.

## 읽는 순서

1. [최종_결과_보고서.md](최종_결과_보고서.md) — 3세대 전체 조망.
2. 각 세대 `final.md`(파이프라인 단계별 모델·프로그램) → `CHANGES_from_*.md`(세대 전환 이유).
3. 세부 지표: `test_3/결과물/probes_results/` · `test_3/결과물/rag3_참고_실험팩트정리.md`.
4. [chatbot_demo](chatbot_demo/README.md) — v1 데모 실행 및 구조.
5. [chatbot_demo_v2](chatbot_demo_v2/README.md) — v2 실행·API·환경변수·시연 절차.
   → [최종_구축_보고서](chatbot_demo_v2/최종_구축_보고서.md)(계획 대비 점검·검증) ·
   [파이프라인 비교](chatbot_demo_v2/docs/파이프라인_비교.md)(v1↔v2 그림·실측 트레이스).

> ⚠️ **경로 주의**: 각 RAG 패키지의 `config.yaml`은 사전데이터를 `../데이터…` 상대경로로 참조하므로,
> 사전데이터를 `사전데이터/`로 분리한 현 구조에서 **재-ingest 경로는 조정이 필요**하다. 쿼리·데모는
> test_3의 경우 내부 캐시로 자립 동작한다. (로컬 모델 특성상 실행 검증은 지양, 비교대조로 검증함.)
