# Phase 3 최종 리포트 — vision 개선 (figure→text 전환)

> 작성: 2026-07-16. 상세 근거: `PHASE3_P0B_REPORT.md`(백엔드 판정), `PHASE3_PILOT_REPORT.md`(시범·인프라).
> 원시결과: `phase3_final_eval.json`(확대 후 36문항), `p0b_mineru25.json`.

## 메인 플롯 대비 결과

계획 Phase 3 = **vision 오독을 줄여 챗봇 신뢰성 확보**. P0-B로 전략 확정 후 실행:
**figure/도표 페이지를 MinerU vlm-engine으로 텍스트화 → pipeline 캐시에 병합 → text 경로로 답변**(vision 회피).

| 계획 정량목표 (Phase 3) | 기준 | 결과 | 판정 |
|---|---|---|---|
| vision 오독률(정답 페이지 본 경우) | 86% → ≤30% | **0%** (하드 7종 오독 0건) | ✅ 초과달성 |
| vision 경로 지연 | ≤90s | 해당 케이스가 text로 전환됨(≤60s) | ✅ |

**test_2에서 vision이 오독하던 도표(vp_009 개요→캐너 등)가 이제 text 경로로 정답.** 오독 대신 정답 또는 정직한 abstain — 확신하며 틀리는 케이스 0.

## 구현 범위 (확대 완료)

- **figure-dominant 페이지 117개 중 115p 텍스트화**(시범 52p + 확대 63p). 미처리 1p = 현황분석서 조밀 스캔표(vlm-engine이 폭주·회귀시키는 래스터 표라 무해).
- 문서별 개별 PDF로 vlm-engine 순차 파싱(`MINERU_API_MAX_CONCURRENT_REQUESTS=1`) → 결합 1PDF의 window 스톨/타임아웃을 회피(격리·부분산출).
- `parsed_v25`(test_2 캐시 비파괴 복사본)에 병합, config `source_parsed_dir`로 전환. 원본 불변(가역).

## 36문항 전체 회귀 (확대 후, vs Phase 2)

| 지표 | Phase 2 | Phase 3(확대) | |
|---|---|---|---|
| evidence_present(정답페이지 top3) | 20 | **24** | ↑ 도표 텍스트가 검색 앵커 |
| avg_kw_hit(관련) | 0.386 | **0.443** | ↑ |
| answered_rate(관련) | 0.548 | **0.613** | ↑ |
| **vision 오독** | 다수 | **0** | ✅ |
| 무관 거절 | 5/5 | **5/5** | 유지 |
| 환각 | 0 | **0** | 유지 |
| avg 모델호출 | 2.58 | 3.06 | 굶음재시도로 증가(≤3.5) |

- **개선(0→1.0)**: vp_009, vp_010, vp_014 (도표 오독/검색미스 해소).
- **표면 회귀(core_001, core_007)**: 비시범 미변경 문서. 원인은 **12b 콜드스타트 답변 변동성**(웜 반복은 결정론적 temp=0; core_001 웜 0.86/콜드 0.0). Phase 3가 만든 회귀가 아니며, 상시구동(keep_alive) 환경에선 웜이라 안정적. → 별도 신뢰성 이슈.

## 부수 인프라 개선(전 질의 이득)

1. **B6 완전 제거**(`page_store.py` flat KV) — 재-ingest 후 Chroma page_index HNSW 크로스프로세스 실패 재발을 KV용도에서 json으로 대체해 원천 차단.
2. **mermaid 평탄화**(`vlm_reparse._demermaid`) — 도표를 12b가 읽는 평문으로.
3. **굶은 답변 재시도**(`answer._looks_starved`) — 노이즈 표 컨텍스트로 12b가 빈/제목만 응답 시 1순위 페이지 단독+간결프롬프트로 재생성(P0-C 완화).

## 남은 한계 (정직한 기록)

- **래스터 UI 스크린샷 표/카드**(vp_006/007/013): vlm-engine·pipeline 모두 한계. text로 못 읽으면 abstain(환각 0). 파싱/코퍼스 한계.
- **일부 검색 미스**(vp_011): 도표 텍스트화로 대부분 해소됐으나 UI 절차 스크린샷은 검색 앵커가 약함 → Phase 1 리랭크 튜닝 영역.
- **12b 콜드스타트 변동성**: Phase 3 범위 밖. seed 고정/웜업 강제/num_predict 재검토로 별도 완화 가능.

## 결론

Phase 3의 계획 목표(**vision 오독 ≤30%**)를 **0%로 초과 달성**했고, 집계 지표(kw_hit·answered_rate·evidence)도 순증했다. 도표를 text로 전환하는 전략(P0-B에서 확정)이 유효했으며, 환각 0·무관거절 5/5의 챗봇 신뢰성 요건을 유지한다.
