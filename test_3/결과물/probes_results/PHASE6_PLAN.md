# PHASE 6 실행 계획 — RAG3 자율 고도화 (병행 실험판 rag3x)

> 작성: 2026-07-19. 근거: 프롬프트 `RAG3_PHASE6_고도화_프롬프트.md` + §1 필수 독서 완료
> (README / FINAL_REPORT / PHASE5 / config.yaml / controller·models·retrieve·answer·verify·judge·config·metrics /
> eval_varfix.py / verify_controller_regression.py / varfix_eval.json).
> **이 문서는 계획일 뿐이다. 채택 결정은 §8.3 최종 A/B 표를 본 사용자가 한다.**

---

## 0. 확정 사실 (재조사 불필요, 실측 재확인 완료)

- `.env`에 `GEMINI_API_KEY` 존재(값은 미열람·미노출).
- conda `intern_chatbot`에 **Gemini SDK 미설치**, `requests`·`dotenv` 설치됨
  → **신규 의존성 0개로 Gemini REST 호출**(`generativelanguage.googleapis.com`), 응답 `usageMetadata`로
  질문당 토큰·비용 실측. (틀 유지·환경 동결 원칙에 부합)
- 기존판 baseline(varfix, 36문항): avg_kw_hit **0.5284** / page_hit@3 **0.774** / doc_hit **1.0** /
  환각 **0** / 무관거절 **1.0** / avg **63.4s** / max **148.6s** / avg_calls **2.97** /
  length_retry 총 28회·18문항. 파싱천장 abstain 5문항 = `core_002, core_008, vp_006, vp_007, vp_013`.
- `rag3x/` 미존재 → 신규 생성.

## 1. 우선순위 (실측으로 조정 가능)

| 순위 | 항목 | 승인 | 근거 |
|---|---|---|---|
| P0 | 병행 실험판 골격 + 무회귀 등가성 증명 | 불필요 | 모든 실험의 전제(§8.1) |
| P1 | 로컬 속도(fail-fast·조건부 검증스킵·적응형 트림) | 불필요 | 즉시 착수 가능(프롬프트 §4 P1) |
| P2 | Gemini 2.5 Flash 백엔드(생성·검증만, 검색은 로컬) | **승인 완료** | 이번 Phase 핵심 축(§4 P2) |
| P3 | 다문서 종합·추론(평가셋→라우팅→분해→문장인용) | **P3-a 검수 게이트** | 측정수단부터 부재(§3.2-4) |
| P4 | (선택) 검색미스 7문항 앵커 강화 | 불필요 | 여유 시(§4 P4) |

**절대 불변(어떤 실험도 후퇴 금지): 환각 0 · 무관거절 100%.** 하나라도 후퇴 시 그 실험은 즉시 미채택.

---

## 2. 병행 실험판 구조 — `test_3/rag3x/`

원칙: **기존 rag3 한 줄도 수정 안 함.** 동작 불변부는 import 재사용, 변경부만 명시적 포크(상단 주석에
원본 경로·변경 요지). 공유 자원(인덱스·파싱캐시·카탈로그) **읽기 전용**. 산출물은 `test_3/tmp/`·`probes/results/`만.

```
test_3/rag3x/
├── __init__.py         Rag3xEngine re-export
├── xconfig.py          load_config() 재사용 + 실험 플래그를 setattr로 부착
│                       (기존 코드가 getattr(config,'flag',default)로 읽으므로 Config 무수정으로 확장)
├── gemini_backend.py   Backend 서브클래스: chat_text/chat_vision_text=Flash REST,
│                       embed=OllamaBackend 위임(임베딩·리랭커 절대 불변). 토큰/비용 로깅·
│                       타임아웃·429/5xx 지수백오프·네트워크 장애 시 로컬 12b 폴백.
├── controller_x.py     controller.answer_question 포크. 신규 동작 전부 플래그 게이트,
│                       OFF면 원본과 완전 동일. (fail-fast, 조건부 검증스킵, 라우팅→분해)
├── answer_x.py         적응형 컨텍스트 트림 / 분해검색 합성 / 문장단위 인용 검증
├── decompose.py        질문→하위질문 분해 + 하위질문별 기존 검색스택 독립 실행
└── engine_x.py         Rag3xEngine(engine.Rag3Engine 패턴, ask()는 기존 키 유지+추가만)
```

재사용(import, 무수정): `config.load_config`, `flat_index`, `retrieve`, `rerank`, `index`,
`page_store`, `chunking`, `catalog`, `parse*`, `evidence`, `metrics`, `verify`, `models.OllamaBackend`.

**등가성 계약**: 모든 실험 플래그 OFF일 때 `rag3x` = `rag3` byte-identical 동작.
→ 착수 직후 `verify_controller_regression`을 rag3x 경로로 돌려 원본과 동일 출력임을 증명(P0 게이트).

## 3. 실험 축(최종 A/B 비교표 열)

| 열 | 구성 | 목적 |
|---|---|---|
| **A. 기존판** | rag3 varfix (대조군) | 불변 기준선 |
| **B. 로컬개선** | rag3x + P1(fail-fast·검증스킵·트림), 백엔드=Ollama | 파이프라인 개선효과(모델 고정) |
| **C. Flash** | rag3x + Gemini 생성·검증, 검색=로컬 | 모델 교체효과 |
| **D. Flash+분해** | C + P3 분해·문장인용 라우팅 | 종합추론 대응 |
| (부속) B'·D' | 분해 경로를 12b로도 측정 | "파이프라인 vs 모델" 효과 분리 |

## 4. 단계별 실행·검증

### Phase 6.0 — 골격 + 골든 베이스라인 (승인 불필요, 무비용)
1. `rag3x/` 골격 작성(전 플래그 OFF).
2. 골든 확보: 원본 `verify_controller_regression.py` 실행(~4분, 웜) → 기준 출력 저장.
3. **등가성 증명**: rag3x 경로로 동일 4문항 실행 → 원본과 동일 답변/근거/점수 확인.
   불일치 시 rag3x 수정(기존 rag3는 절대 안 건드림).

### Phase 6.1 — P1 로컬 속도 (승인 불필요)
- (a) **꼬리 fail-fast**: length-retry 2회 소진 문항은 후속 롤백 생략 즉시 abstain
  (대상 vp_006/007·core_002/008 등은 파싱천장이라 abstain이 정답 — FINAL §5-B).
- (b) **조건부 검증 스킵**: 숫자대조 통과 ∧ rerank top 고점 ∧ 단일문서일 때 groundedness LLM 생략.
  차선 ablation: `verify_model: gemma4:e4b` 스왑.
- (c) **적응형 컨텍스트 트림**: rerank 점수 급락 페이지를 컨텍스트에서 제외(prefill 절감).
- 검증: 각 ablation을 verify_controller_regression으로 수시 → 조합 확정 후 **36문항 전체**(rag3x 평가 스크립트).
  **게이트: 환각 0 · 무관거절 100% · page_hit@3 ≥ 0.774 · kw 무회귀.**
- 목표: 평균 63.4→**≤45s**, 최대 148.6→**≤90s**.

### Phase 6.2 — P2 Gemini Flash (승인 완료)
- `gemini_backend.py`: temp 0, `.env` 키 로드(로그·보고서 노출 절대금지), API 타임아웃,
  429/5xx 지수백오프, 장애 시 **로컬 12b 폴백**, 호출별 입출력 토큰·질문당 비용 기록.
  Gemma 전용 방어(retry_on_length·num_predict 튜닝) Flash 경로에서 비활성.
- 무료 티어면 RPM/일일 한도 먼저 확인 후 평가 페이스 조정.
- 검증(36문항): 가설1 속도("전 문항 20s 이내" 명시 측정) / 가설2 kw_hit(0.528 대비) /
  가설3 사고력(P3셋에서) / **가설4 vision**: `vision_answer_model` 자리에 Flash → 파싱천장 5문항이 열리는지.
  **환각 0·무관거절 100% 유지 필수 확인.** 목표: 평균 ≤15s, 최대 ≤30s.

### Phase 6.3 — P3 다문서 종합·추론
- (a) **평가셋 먼저**: 13문서 코퍼스에서 종합형 10~15문항(문서간 비교/공통/집계) 초안 →
  **⛔ 사용자 정답 검수 게이트(사람 검수 없이 개선 착수 금지).**
- (b) 라우팅 신호 조사: rerank 문서간 분산·비교어휘("차이/각각/종합") 실측 → S0 라우팅
  (단일형은 기존 경로 그대로 = 무회귀).
- (c) 분해검색 프로토타입(`tmp/`): 질문→하위질문 2~4개→**기존 검색스택 하위질문별 독립 실행**→
  문서별 그룹 컨텍스트→합성 1회. 라우팅으로만 진입.
- (d) 문장단위 인용 검증: 합성 답변 문장별 인용태그 강제 → 미지원 문장만 제거 부분답변(전체 abstain 대신).
  **환각 0 절대 유지.**
- 분해는 12b판·Flash판 모두 측정(효과 분리). 단일형 경로 무영향 보장.

### Phase 6.4 — 최종 산출물
- `PHASE6_COMPARISON.md`: §8.3 동일조건 비교표(A/B/C/D 열) + 추천안·근거(**결정은 안 함**).
- 중간 실험은 `PHASE6_*.md`에 가설→방법→측정→판정(미채택·실패도 근거와 함께 기록).

## 5. 운영 규칙 (PHASE5 §4 실측 사고 반영)
- **GPU 단독 실행**: 평가/서버/add 동시 2개 = 무증상 크래시 → 장시간 GPU 작업은 한 번에 하나·포그라운드,
  상태는 `Get-Process`로 직접 확인(완료 알림 불신).
- 속도는 **웜 상태 비교**(워밍업 후 측정).
- Chroma page_index upsert 금지. 실험판은 공유 인덱스 **읽기 전용**.
- 모든 실행 `cmd /c conda activate intern_chatbot && ...`(프로젝트 CLAUDE.md).
- **결정 전까지 기존 코드·README·웹데모 벤치마크 절대 갱신 안 함.**

## 6. 사용자 확인 게이트 (이 3가지 외 자율 진행)
1. 기존 파일 수정이 불가피할 때(사유+diff 선제시) — **현 설계상 불필요하게 유지가 목표**.
2. 인덱스/캐시/카탈로그 파괴 가능 작업.
3. **P3-a 종합형 평가셋 정답 검수.**

## 7. 성공 기준(가설, 실측 조정 가능)
| 축 | 목표 |
|---|---|
| 속도(로컬 B) | 평균 ≤45s, 최대 ≤90s |
| 속도(Flash C) | 평균 ≤15s, 최대 ≤30s, "전 문항 20s 이내" 달성여부 명시 |
| 품질(전 실험판) | page_hit@3 ≥0.774, doc_hit 1.0, **환각 0**, 무관거절 **100%** |
| 정확도(Flash) | kw_hit 0.528 대비 개선여부 + 파싱천장 5문항 개선여부 실측(하락해도 그대로 보고) |
| 종합추론 | 신규셋 기존판 baseline + 실험판 격차 정량 |
| 종착 | `PHASE6_COMPARISON.md` 제출 → 대체/유지/병행 결정 대기로 종료 |
