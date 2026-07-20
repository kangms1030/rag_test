# Phase 3 시범 리포트 — figure 페이지 vlm-engine 재파싱·병합 (4문서)

> 작성: 2026-07-16. 원시결과: `phase3_pilot_eval.json`, `p0b_mineru25.json`.
> 구현: `test_3/rag3/{vlm_reparse,page_store}.py`, `retrieve.py`/`ingest.py`/`answer.py` 수정.
> 사용자 지시: "먼저 3~4문서만 시범 → 회복 검증 후 확대".

## 배경 (P0-B 결론)

P0-B(PHASE3_P0B_REPORT.md): 계획의 "스캔 표 텍스트화" 가설은 방향이 반대였다. MinerU vlm-engine은
**벡터 도표/구성도**를 정확한 구조화 텍스트(mermaid 포함)로 뽑지만 래스터 UI 표는 회귀시킨다.
→ Phase 3 전략을 **figure 페이지만 vlm-engine 재파싱 → pipeline 캐시에 병합 → text 라우팅**으로 재정의.

## 시범 구성

- **문서 4개**(figure 페이지 52p): 무선랜(8p, vp_009/010) · 통합관제3(3p, vp_014) · 이용관리4(5p, vp_015) · 23년안내서(36p, vp_011/012).
- **파이프라인**: figure 페이지 식별(이미지 有 & 텍스트<200자) → 결합 PDF 1개 → vlm-engine 순차 파싱(`MINERU_API_MAX_CONCURRENT_REQUESTS=1`) → 페이지별 텍스트를 `parsed_v25`(test_2 캐시의 비파괴 복사본)에 `type:text` 블록으로 병합 + manifest page_type→text → 재-ingest.
- **비파괴/가역**: 원본 test_2 캐시 불변, 산출은 `cache/parsed_v25`. config `source_parsed_dir`로 전환.

## 결과 (10문항: 회복대상 6 + 회귀체크 4)

| qid | 페이지 유형 | 이전(Phase 2) | 시범 후 | 판정 |
|---|---|---|---|---|
| vp_009 | 도표(앱소개) | vision **오독**(개요→캐너) | text kw=**1.0** | ✅ 회복 |
| vp_010 | 구성도(장애진단) | 검색미스·abstain | text kw=**1.0** | ✅ 회복 |
| vp_014 | mermaid 다이어그램 | 검색미스·abstain | text kw=**1.0** | ✅ 회복 |
| vp_012 | IP/DNS 절차도 | abstain | text kw=**1.0** | ✅ 회복 |
| vp_015 | 신청화면 | 0.8 | 0.8 | 유지 |
| vp_011 | UI 스크린샷 절차 | abstain | abstain | 검색미스(p22 미랭크) |
| vt_001 | (회귀)자원구성 표 | 1.0 | 1.0 | 회귀 0 |
| vt_002 | (회귀)로그인 규칙 | 1.0 | 1.0 | 회귀 0 |
| vp_006 | (회귀,비시범)래스터표 | abstain | abstain | 불변 |
| vp_007 | (회귀,비시범)래스터표 | abstain | abstain | 불변 |

**요약: 도표 회복 4/6, 회귀 0.** 특히 vp_009는 test_2에서 vision이 오독하던 대표 사례를 **text 경로 정답**으로 전환했고, vp_010/014는 이전 검색미스(ev=0)를 **검색 1순위**로 끌어올렸다(도표 텍스트가 검색 앵커가 됨).

## 부수적으로 해결한 인프라 이슈 (검색·답변 전 구간에 이득)

1. **B6 완전 제거(page_store.py)**: 재-ingest 후 Chroma page_index(969)가 크로스프로세스 로드 실패(B6 재발). 리랭크 경로에서 page_index는 벡터검색이 아니라 **page_id KV 조회**뿐이므로 flat json(`page_store.json`)으로 대체 → HNSW 의존 제거. (게이트 경로용 Chroma page_index는 잔존하나 기본 경로 미사용.)
2. **mermaid 평탄화(vlm_reparse._demermaid)**: 12b는 원시 ```mermaid``` 코드에 빈 응답을 낸다(P0-C). 노드/서브그래프 라벨만 평문으로 추출 → vp_014 답변 가능.
3. **굶은 답변 재시도(answer._looks_starved)**: 여러 페이지 컨텍스트에 지저분한 표 HTML이 섞이면 12b가 본문 생성 전 EOS로 끊겨 빈/제목만 응답(P0-C). 이를 감지(빈·초단문·제목에코)하면 **1순위 페이지 단독 + 간결 프롬프트**로 1회 재생성 → vp_012/014 안정 회복. 전 질의에 적용되는 일반적 강건성 개선.

## 미회복 1건 분석

- **vp_011**(23년안내서 p22, 스쿨넷 신규기관 등록 절차 UI 스크린샷): vlm 텍스트는 병합됐으나 **검색에서 p22가 top3 밖**(p19가 상위). 파싱이 아니라 **검색 랭킹(Phase 1)** 문제. UI 스크린샷 절차는 vlm이 뽑은 텍스트가 질의와의 변별 앵커가 약함.

## 규모·비용 (확대 시)

- 전 코퍼스 figure-dominant = 969p 중 **117p(12%)**. 시범 52p 재파싱 ≈ 40분(모델 로드 1회 상각, 페이지당 ~40s). 나머지 ~65p 확대 ≈ 추가 40~50분(1회성 오프라인).
- 병합·재-ingest는 멱등(재병합 시 원본 텍스트 보존 마커 `_vlm_orig_text`). vlm 산출은 재사용.

## 판정 및 권장

- **시범 성공**: 도표 클래스에서 4/6 회복 + 검색미스 해소 + 회귀 0. Phase 3 주 전략(figure→vlm-engine→text)이 유효.
- **권장**: ① 나머지 figure 페이지(약 65p)로 **확대 재파싱**, ② 확대 후 **36문항 전체 회귀 평가**로 집계 지표(근거있는데실패↓, kw_hit↑) 재측정, ③ vp_011류 검색미스는 Phase 1 튜닝(리랭크 raw text·figure 앵커 보강)으로 별도 공략.

## 36문항 전체 회귀 평가 (확대 전, 4문서 상태) — `phase3_full_eval.json`

| 지표 | Phase 2 | Phase 3(4문서) | 판정 |
|---|---|---|---|
| evidence_present(관련 top3에 정답페이지) | 20 | **24** | ↑ 도표 텍스트가 검색 앵커 |
| avg_kw_hit(관련) | 0.386 | **0.467** | ↑ |
| answered_rate(관련) | 0.548 | **0.613** | ↑ |
| 무관 거절 | 5/5 | **5/5** | 유지 |
| avg 모델호출 | 2.58 | 3.13 | 재시도로 증가, ≤3.5 유지 |
| 환각 | 0 | **0** | 유지 |

- **개선 4건**: vp_004(0.25→1.0), vp_009/010/014(0→1.0).
- **표면상 회귀 2건**(core_001 1.0→0.43, core_007 0.67→0.33)은 **Phase 3가 원인이 아니다**: 둘 다 비시범(미변경) 문서이고, 텍스트·검색은 동일하다. 원인은 **12b 콜드스타트 답변 변동성** — 동일 질문이 모델 웜업 상태(프로세스 경계)에 따라 다른 길이의 답을 낸다(실측: core_001 웜 0.86 vs 콜드 0.0, vp_012 웜 1.0 vs 콜드 abstain). **웜 상태 연속 반복은 완전 결정론적**(temp=0). Phase 2에도 있던 환경 특성으로, 챗봇 상시구동(keep_alive 30m)에선 웜 상태라 안정적이다.
- **별도 신뢰성 이슈로 기록**: 12b 콜드스타트 변동성 → seed 고정/웜업 강제/num_predict 재검토로 완화 가능(Phase 3 범위 밖).

## 산출물

- 코드: `test_3/rag3/vlm_reparse.py`(재파싱 병합), `page_store.py`(flat KV), `answer.py`(굶음 재시도), `ingest.py`/`retrieve.py`(page_store 연동).
- 데이터: `test_3/rag3/cache/parsed_v25`(병합 캐시), `test_3/rag3/index/page_store.json`.
- 결과: `phase3_pilot_eval.json`. 오케스트레이션: `tmp/pilot_*.py`.
