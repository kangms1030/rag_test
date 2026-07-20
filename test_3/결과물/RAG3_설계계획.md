# RAG_3 (test_3) 고도화 설계 계획 — 최종

## Context (왜 이 작업을 하는가)

test_2(MinerU 사전파싱 RAG)는 test_1 대비 약 40배 빠르고(문항당 20.5s) 텍스트 경로 숫자 정확도 100%를 달성했지만, 챗봇 모듈로 쓰기에는 실측된 결함이 남아 있다:

1. **vision 경로 오독 86%** — 정답 페이지를 정확히 본 7건 중 6건 오독 (숫자·시리얼·용어 전 범주, 동일 페이지 동일 오독 재현)
2. **근거가 있어도 실패 33%**(catalog 모드 6/18) — 정답이 2~3순위 페이지일 때 빈 응답/"확인 불가"
3. **카탈로그 게이트의 조기 거절·형제 문서 혼동** — 13문서 소규모에선 no_catalog가 doc_hit 0.923 vs 0.692로 우세
4. **반복 서식 스캔 문서에서 페이지 특정 실패** — page_hit@3 55.6%
5. **단방향·무검증** — 검색→답변 단일 통과, 롤백/재검색/검증 없음

test_3의 목표: **기조(MinerU 사전파싱 → 그림·표는 VLM → LLM 답변)를 유지하면서 시간 비용과 정확도를 동시 개선**하여 챗봇 모듈로 쓸 수 있는 수준으로 끌어올린다.

제약: VRAM 16GB, 전부 로컬(Ollama), conda env `intern_chatbot`, Windows. 모든 터미널 명령은 `cmd /c` (+conda activate).

---

## 이번 계획 수립 과정에서 새로 확인한 사실 (2026-07-15 실측)

| # | 발견 | 함의 |
|---|---|---|
| F1 | catalog 빈 응답 3건(vp_002/006/007)의 컨텍스트가 6,113~15,237자(표 HTML 포함) | num_ctx 8192 오버플로 의심 → Phase 0에서 `prompt_eval_count`로 판정 |
| F2 | "확인 불가" 2건(vp_010/014)은 정답이 2~3순위 **figure 페이지(텍스트 19~158자)**인데 1순위 기준 라우팅이 text 선택 | 라우팅 결함. 모델은 정직하게 거절한 것 — 다중 페이지 근거 라우팅으로 해결 |
| F3 | 표 크롭 = MinerU 산출 JPG 그대로(1262×612 등), 우리 DPI 설정과 무관 (parse_mineru.py:166-170) | 고해상도 개선은 PDF 원본 bbox 재렌더로 해야 함 |
| F4 | **Ollama 런타임 병목**: Gemma3 pan&scan 미구현(ollama#10392), Gemma4 비전 토큰 예산 max_soft_tokens=280 하드코딩(ollama#15626) | 어떤 해상도를 보내도 비전 인코더에서 뭉개짐 — 오독 86%의 유력 근본 원인. 타일링(다중 이미지) 또는 vision 의존 축소로 우회 |
| F5 | **VRAM 실측**: gemma4:12b 7.6GB / **gemma4:e4b 9.6GB** / embeddinggemma 621MB / Ollama 0.31.2 | **12b+e4b 동시 상주 불가(합 17.2GB > 16GB)**. e4b를 중간 단계에 쓰면 매번 모델 스왑 → 경량 판정의 기본값은 "상주 중인 12b 짧은 호출" |
| F6 | 색인이 페이지 통짜 + 임베딩 절단 제어 없음 (embeddinggemma 컨텍스트 2K) | 긴 페이지 꼬리가 dense 검색에 안 보임 → 청크 색인 필요 |
| F7 | BM25가 질의마다 전 레코드 로드 + 재구축 (index.py) | 청크화로 색인이 3~6천 건 되면 악화 → BM25 캐시 필수 |
| F8 | PROMPT_TEXT_ANSWER가 [지시문→질문→근거→답변:] 순서 | 오버플로 시 지시문·질문 소실 가능 — F1과 결합해 빈 응답 가설 강화 |

## 사용자 확정 사항 (AskUserQuestion, 2026-07-15)

- **카탈로그**: 게이트/문서선정 제거 → **메타데이터 주입으로 전환** (제목·키워드·범위를 청크 텍스트에 병합). 코퍼스 확장(249문서) 시 프리필터로 재활성화할 수 있게 config 스위치(`use_catalog_gate: false`)만 유지
- **MinerU 2.5**: **Phase 0 프로브 후 결정** (오독 페이지만 A/B → 개선 확인 시 전체 재파싱). 16GB VRAM 엄수
- **리랭커**: **bge-reranker-v2-m3 도입 허용** (16GB VRAM 예산표 필수)
- **평가셋**: **50문항 내외로 확장** (기존 36문항은 회귀셋으로 동결, 신규 ~14문항은 홀드아웃)

## 사용자 4개 질문에 대한 결론

**1. 카탈로그 제외?** — 객관적 판단: 게이트로서는 제거가 옳다(실측: doc_hit 0.692 vs 0.923, dense 0.342<0.35 조기 거절, 형제 문서 혼동). 단 카탈로그가 이득을 준 실측 사례(vp_011 형제 시스템 혼동 방지)도 있으므로 **완전 폐기가 아니라 "메타데이터 주입"으로 전환** — 카탈로그의 제목/키워드/범위를 청크 텍스트 프리픽스로 병합하면 게이트 부작용 없이 변별력만 취한다. 249문서 확장 시 재활성화 스위치 유지.

**2. 양방향화(롤백·검증)** — 채택하되 **결정론적 트리거 + 고정 상한**으로 설계 (test_1의 다단계 LLM 판정 폭증 회귀 방지). e4b 활용 구상은 VRAM 실측(F5)상 스왑 비용이 커서, 경량 판정의 기본값을 "상주 중인 12b 짧은 호출"로 교체하고 e4b는 오프라인(ingest) 전용으로 격하. Phase 0-E에서 스왑 지연 실측 후 최종 결정.

**3. 이미지 해상도** — test_2는 축소만 하고 확대 안 함(확인). 진짜 문제는 두 겹: (a) MinerU 크롭 자체가 저해상도(F3), (b) Ollama 런타임이 비전 입력을 뭉갬(F4). 처방: PDF 원본 bbox 고DPI 재렌더 + 클라이언트측 타일링 + **vision 의존 자체를 축소**(MinerU 2.5로 스캔 표를 ingest에서 텍스트화). Phase 0-A/B 프로브로 기여도 분리 측정.

**4. 신기술 적용(16GB 내)** — 리랭커(bge-reranker-v2-m3, fp16 ~1.5GB), Anthropic Contextual Retrieval(결정론 프리픽스 기본 + LLM 생성은 ablation 통과 시), 블록 기반 청킹(small-to-big), CRAG식 재검색, groundedness 검증. ColPali류 시각 검색은 스트레치(채택 안 함 — 반복 서식 페이지는 시각적으로도 동일해 효과 불확실).

---

## 설계 원칙 (실측 근거)

1. **Text-first**: text 경로 숫자 정확도 100%(8/8) vs vision 오독 86%(6/7) → vision은 "텍스트 부재 시 최후 수단 + 교차검증 수단"으로 강등. "스캔+표→vision" 라우팅 규칙 폐기(MinerU OCR 텍스트가 더 정확).
2. **랭킹이 상류 원인**: 실패 6건 전부 "정답이 2~3순위" → 리랭커+청크 색인으로 정답을 1순위로 끌어올리는 것이 실패 3종(빈 응답, 라우팅 오류, 페이지 특정)을 동시에 공략.
3. **게이트 제거, 거절은 리랭커 절대점수로**: 무관 질문 거절(현행 100%)은 리랭크 점수 τ_low로 대체, dense 유사도 보조 하한 병행.
4. **모든 재시도는 결정론적 트리거 + 고정 상한**: LLM 판정 단독으로는 롤백 불가(오판 루프 차단).

---

## 아키텍처

### 오프라인 ingest 변경점

| # | 변경 | 근거 |
|---|---|---|
| 1 | 페이지 통짜 색인 → **MinerU 블록 기반 청크 색인**(텍스트 250~450토큰 병합, 표=단독 청크, 초과 표는 헤더 반복 행 분할). 페이지 링크 메타 유지 | F6 임베딩 절단 제거, 반복서식 페이지 특정 개선 |
| 2 | 청크 프리픽스 주입(결정론): `문서: {카탈로그 제목/키워드} | 섹션: {heading 경로} | p{n}` | 카탈로그 메타데이터 주입(사용자 확정), 형제 문서 변별 |
| 3 | (옵션) e4b LLM 컨텍스트 문장 생성(청크 해시 캐시, 오프라인 2~5h 1회) | Contextual Retrieval 검색실패 49%↓ 보고 — Phase 4 ablation 통과 시만 |
| 4 | page_index 존치(small-to-big의 "big"), catalog_index 코드 유지·기본 off | 249문서 확장 대비 |
| 5 | BM25 시작 시 1회 구축·디스크 캐시 | F7 |
| 6 | (Phase 0 결과부) MinerU 2.5 재파싱 → 스캔 표 텍스트화 | vision 의존 근본 축소. 별도 캐시 디렉터리(`cache/parsed_v25/`) 병행 후 전환 |
| 7 | ingest 요약에 청크 수/토큰 분포/절단 발생 수 기록 | 재발 감시 |

### 온라인 ask 파이프라인 (상태기계, controller.py)

```
[Q] 질문
 ├─ S1 질문 임베딩 1회 ..................... embeddinggemma, ~0.3s
 ├─ S2 청크 하이브리드 검색 top20 .......... BM25캐시+dense RRF, ~0.5–1s
 ├─ S3 리랭크 top20 → 정렬 ................. bge-reranker-v2-m3, ~0.3–0.5s
 │    ├─ top1 < τ_low → S3a 질의 재작성 1회(12b 짧은 호출) → S2–S3 재실행
 │    │    └─ 여전히 미달 → "확인 불가" 종료 (무관 거절, ~2–10s)
 │    └─ top1 ≥ τ_low → 진행
 ├─ S4 증거 조립 (small-to-big) ............ 청크→페이지 승격, 리랭크 점수 페이지별 집계
 │    · 최대 3페이지, 컨텍스트 토큰 상한 ~10K 트림, 12b num_ctx 16384
 ├─ S5 라우팅 v2 (결정론, 증거 집합 전체 기준, text 기본)
 │    ├─ text: 선정 페이지에 사용 가능한 텍스트(OCR/표HTML) 존재 ← 대부분
 │    └─ vision: (figure_area_ratio≥0.5 AND char_count<200) OR 표 body 부재
 │       ※ "스캔+표→vision" 규칙 폐기
 ├─ S6 답변 1회 ............................ gemma4:12b
 │    ├─ text: ~30–45s
 │    └─ vision: render.py 고DPI 크롭/타일(≤3장) → transcribe-then-answer, ~45–70s
 ├─ S7 검증 (싼 것부터)
 │    ① 숫자/시리얼 정규식 추출 → 근거 문자열 대조 (결정론, 0s)
 │    ② vision이면 전사 vs MinerU OCR diff (0s)
 │    ③ groundedness 판정 (12b 짧은 호출 ~5s; Phase 0-E 결과에 따라 e4b 대체 검토)
 └─ S8 롤백 (최대 1회, 결정론 트리거만)
      ├─ 빈 응답/"확인 불가" AND 리랭크 top1 ≥ τ_high → 차순위 페이지 단독으로 재시도 1회
      ├─ text 숫자 검증 실패 AND 스캔 페이지 → vision 경로 전환 1회 (교차확인)
      └─ vision 전사 ≠ OCR → OCR 텍스트로 답 재구성 (모델 호출 0)
[A] 최종 JSON: 답변 + 경로 + 검증 결과 + rollback_history + 신뢰도 태그
```

### 모듈 구조 (`test_3/rag3/`)

- **그대로 재사용**: tokenizer.py, utils.py, imaging.py, parse.py, parse_pdfplumber.py, eval_sets.py
- **역할 전환**: catalog.py (게이트 폐기 → 메타 주입 소스)
- **수정 재사용**: config.py/config.yaml (rerank_*/chunk_*/verify_*/rollback_* 키), parse_mineru.py (블록 bbox·heading 보존), index.py (BM25 캐시 + chunk_index), ingest.py (청크 색인+프리픽스), models.py (per-call num_ctx, 다중 이미지, chat_json), retrieve.py (게이트 제거·리랭크·라우팅 v2), answer.py (text-first·타일링·교차검증 훅), evaluate.py (+answer_given_evidence_rate, verify 통과율, rollback 발생률), envcheck.py, metrics.py, schema.py, __main__.py (+probe 서브커맨드)
- **신규**: chunking.py, rerank.py, contextual.py, judge.py (CRAG 관련성 판정+질의 재작성), verify.py, controller.py (상태기계·재시도 예산·데드라인), render.py (PDF bbox 고DPI ask-time 재렌더·스캔 내장 이미지 네이티브 추출·타일링 최대 2×2), probes/ (probe_resolution.py, probe_mineru25.py, probe_ctx_overflow.py, probe_rerank.py)

원본 참조 파일: `rag_test/test_2/rag2/{retrieve,answer,index,parse_mineru,models}.py`

---

## 단계적 마일스톤

### Phase 0 — 진단 프로브 (코드 본체 변경 없음, 1~2일)

| 프로브 | 내용 | 통과/판정 기준 |
|---|---|---|
| P0-A 해상도/타일링 | 오독 7건 페이지(vp_006 p83, vp_007 p88, vp_009 p13, vp_010 p11, vp_014 p71 + 정답 대조 vp_011 p22) × 조건 {현행 크롭, 300/400DPI 재렌더, 내장 이미지 네이티브, 2×2 타일, 타일+고DPI} × 12b 전사 | 기지 오독 토큰의 전사 정확도를 조건별 수치로 판정 → Phase 3 분기 |
| P0-B MinerU 2.5 | 스캔 문서 2~3종을 2.5(VLM 백엔드, 1.2B)로 파싱, 문제 표 7건 텍스트 추출 대조 + 시간/VRAM 측정 | ≥6/7 정확 추출이면 Phase 3 주 대응으로 채택 |
| P0-C 빈 응답 원인 | 실패 6건 재실행: 프롬프트 토큰 수 vs `prompt_eval_count` vs num_ctx 8192 대조 → 16384 재실행, 페이지 순서 교체 재실행 | "오버플로 vs 1순위 방해물" 원인 데이터 판정 |
| P0-D 리랭커 선검증 | 기존 인덱스 그대로 top20 → bge-reranker-v2-m3 재정렬 (36문항) | page_hit@3 +10%p 이상이면 확정, 미달 시 Qwen3-Reranker-0.6B 교차 시험 |
| P0-E VRAM/스왑 실측 | `ollama ps` 상주 크기, reranker 로드 후 여유, 12b↔e4b 스왑 지연 | VRAM 예산표 확정 + 판정모델(12b 겸용 vs e4b) 결정 |

### Phase 1 — 검색 코어 (청크 + 리랭커 + 게이트 제거)
- 작업: chunking.py, rerank.py, index.py(BM25 캐시), ingest.py, retrieve.py v2, num_ctx 16384, τ_low/τ_high 캘리브레이션
- 통과: **doc_hit ≥ 0.92** / **page_hit@3 ≥ 0.75**(현행 0.556) / 무관거절 5/5 유지 / retrieve ≤ 3s
- Ablation(동일 36문항, config 토글 1개씩): 청크 vs 페이지 색인 / 리랭크 on·off / 프리픽스 on·off

### Phase 2 — 답변·검증·롤백 루프
- 작업: controller.py, verify.py, judge.py, 라우팅 v2, answer.py 재작성, evaluate.py 지표 확장
- 통과: **근거 있는데 실패 0/18**(현행 catalog 6/18) / kw_hit ≥ 0.70 / 평균 모델호출 ≤ 3.5 / 정상 text 경로 ≤ 60s
- Ablation: verify off / rollback off / CRAG off 단독 토글 → 실패 회귀를 문항 단위 기록

### Phase 3 — vision 개선 (Phase 0 결과로 분기)
- P0-B 통과 시(주): MinerU 2.5 재ingest → 스캔 표 text 경로화
- P0-A 유효 시(보조): render.py 고DPI+타일링 + 전사-OCR 교차검증 상시화
- 둘 다 무효 시: vision 답변에 신뢰도 낮음 태그 + OCR 텍스트 우선 정책 고정 (런타임 한계 F4 판정)
- 통과: vlm_probe 재실행에서 **정답 페이지를 본 케이스 오독 ≤ 2/7**(86% → <30%) / vision 경로 ≤ 90s

### Phase 4 — LLM Contextual + 평가 확장 + 종합 리포트
- contextual.py v2(e4b, 오프라인) ablation: 결정론 프리픽스 대비 **page_hit@1 +5%p 이상일 때만 채택**
- 평가셋 ~50문항 확정(신규 문항 정답은 사람 검수 필요 — 사용자 협업 항목), 기존 36문항 회귀셋 동결
- 최종 통합 리포트: test_2 대비 전 지표 비교 + 실패 사례 전수 분석

---

## VRAM·지연 예산

### VRAM (16GB, P0-E에서 실측 확정)
| 모델 | 상주 추정 | 정책 |
|---|---|---|
| gemma4:12b (디스크 7.6GB) | ~9–10.5GB (16K KV 포함) | keep_alive 30m 상시 — 주 답변+판정 겸용 |
| embeddinggemma (621MB) | ~0.8GB | 상시 |
| bge-reranker-v2-m3 (fp16, 별도 torch 프로세스) | ~1.5GB | 상주 (Ollama 외부) |
| **동시 합계** | **~11.5–13GB** | 여유 ~3GB (Windows DWM 감안) |
| gemma4:e4b (디스크 9.6GB) | 12b와 **동시 상주 불가**(F5) | 오프라인(ingest 컨텍스트 생성) 전용. 온라인 판정은 12b 겸용이 기본 |

- 온라인 경로에서 모델 스왑 0회가 기본. ingest 시에는 12b keep_alive 0으로 내리고 e4b 사용.

### 지연 (문항당 목표)
| 경로 | 예상 | 목표 |
|---|---|---|
| 정상 text | ~41s (embed 0.3+검색 1+리랭크 0.5+답변 35+검증 5) | **≤ 60s** |
| 정상 vision | ~62s | **≤ 90s** |
| CRAG 재검색 포함 | +5s | 목표 내 흡수 |
| 롤백 1회 | ~80–120s | **≤ 150s** |
| 무관 거절 | ~2–10s | ≤ 10s |
| 하드 데드라인 | — | 180s 초과 시 best-effort 반환 |

---

## 무한루프 방지 규칙 (controller.py 단일 구현)

1. 재시도 예산 각 0→1만: 질의 재작성 ≤1, 재검색 ≤1, 답변 재생성 ≤1, 경로 전환 ≤1, 검증 호출 ≤2
2. 총 모델 호출 상한 5회(임베딩 제외) — 초과 즉시 현재 최선 답 반환
3. `(answer_path, 증거 페이지 집합)` 해시로 방문 상태 재진입 금지
4. 롤백 트리거는 결정론 신호만(빈 응답, "확인 불가", 숫자 대조 실패, 전사-OCR 불일치). LLM 판정 단독은 신뢰도 태그만
5. wall-clock 180s 데드라인
6. rollback_history를 최종 JSON에 기록해 평가에서 루프 비용 상시 계측

## 리스크와 완화책 (요약)

- Ollama 비전 한계(F4)로 고DPI 무효 → P0-A 조기 판정, 타일링 우회, 무효 시 MinerU 2.5 텍스트화가 주 대응
- 리랭커 한국어/도메인(시리얼·지역명) 미검증 → P0-D 선검증, 대안 Qwen3-Reranker-0.6B
- 게이트 제거로 무관거절 회귀 → τ_low를 무관+정상 문항 분포로 캘리브레이션, dense 하한 병행
- 반복 서식 문서 페이지 특정 실패 지속 → 표 단독 청크화로 페이지 고유 앵커(지역명) 보존, 실패 시 인접 페이지 확장 검색
- MinerU 2.5가 캐시/스키마 파괴 → `cache/parsed_v25/` 병행 생성 후 diff 확인·전환
- 소표본 과적합 → 기존 36문항 회귀셋 동결, 신규 ~14문항 홀드아웃

## 정량 목표 요약

| 지표 | 현행 (test_2) | 목표 (test_3) |
|---|---|---|
| page_hit@3 | 55.6% | **≥ 75%** |
| doc_hit | 0.923 (no_catalog) | **≥ 0.92 유지** |
| 근거존재-생성실패율 | 33.3% (catalog) | **0%** |
| vision 오독률(정답 페이지 본 경우) | 86% | **≤ 30%** |
| 무관 질문 거절 | 100% | **100% 유지** |
| 문항당 평균(text/vision/롤백) | 49.5s / 66.3s / — | ≤60s / ≤90s / ≤150s |
| 평균 모델 호출 | 2회 | ≤ 3.5회 |

## 실행 순서 (승인 후)

1. 이 계획을 `test_3/RAG3_설계계획.md`로 저장 (팀 참조용)
2. `test_3/rag3/` 스캐폴드 + rag_test/test_2/rag2에서 재사용 파일 복사 (원본 불변)
3. Phase 0 프로브 4종 구현·실행 → `test_3/probes/results/`에 판정 리포트 저장
4. Phase 0 판정 결과를 사용자에게 보고 후 Phase 1 착수

## Verification (각 Phase 공통)

- 실행: `cmd /c conda activate intern_chatbot && python -m test_3.rag3 evaluate --eval-file ...` (36문항 회귀셋)
- Phase 통과 기준표의 지표를 evaluate.py가 JSON+리포트로 산출, 이전 Phase 결과와 diff
- 평가 전 Chroma 컬렉션 count 확인(기존 B6 교훈: 인프라 손상이 가설로 위장), `ollama ps`로 상주 모델 확인
- 매 Phase ablation은 config 토글 1개씩만 변경해 동일 데이터셋 재실행

## 참고 자료 (리서치 근거)

- Contextual Retrieval: https://www.anthropic.com/engineering/contextual-retrieval (top-20 검색실패 49%↓)
- CRAG: https://openreview.net/forum?id=JnWJbrnaUE / Self-RAG 조합 패턴: https://www.sotaaz.com/post/agentic-rag-part2-en
- 리랭커 비교: https://futureagi.com/blog/best-rerankers-for-rag-2026/ (bge-reranker-v2-m3 기본 추천, Qwen3-Reranker 대안)
- Ollama Gemma 비전 한계: https://github.com/ollama/ollama/issues/10392 (pan&scan 미구현), https://github.com/ollama/ollama/issues/15626 (max_soft_tokens=280 하드코딩)
- Gemma 비전 인코더 896×896 고정·Pan&Scan: https://ai.google.dev/gemma/docs/capabilities/vision , https://arxiv.org/html/2503.19786v1
- MinerU 2.5 (표 TEDS +5.5, 1.2B): https://arxiv.org/pdf/2509.22186 , https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B
- 고해상도 크롭/타일링 연구: https://arxiv.org/pdf/2603.16932 , https://arxiv.org/html/2512.11167v1
- Groundedness/NLI 검증 패턴: https://qaskills.sh/blog/trulens-rag-triad-groundedness-context-relevance-2026
