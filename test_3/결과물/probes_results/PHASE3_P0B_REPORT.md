# Phase 3 착수 진단 — P0-B (MinerU VLM 재파싱) 결과 리포트

> 작성: 2026-07-15. 원시결과: `p0b_mineru25.json`. 프로브: `tmp/p0b_*.py`(추출·채점).
> 환경: MinerU **3.4.4**(2.5의 후속, `vlm-engine`=MinerU2.5-Pro-1.2B 내장), transformers 추론(Windows, vLLM 미사용), RTX 5060 Ti 16GB.
> P0-B는 Phase 0에서 MinerU 미설치로 **보류**됐던 프로브. Phase 3 착수 시 최우선 실행 대상이었다.

## 목적

계획 Phase 3의 주 전략 가설: **"MinerU 2.5로 스캔 표를 텍스트화해 vision 의존을 줄인다."**
이 가설을 test_2가 오독했던 하드 케이스 6건에서 실측 검증한다. 대상은 타일링(P0-A)으로도 못 고친 조밀 표·도표들.

## 방법

오독 6페이지만 1장 PDF로 잘라(전체 문서 재파싱은 969p×VLM≈수십 시간이라 회피) 3개 백엔드로 파싱, gold 토큰 포함 여부 채점:
- **pipeline**(현행 test_2 캐시): OCR 기반
- **vlm-engine**: MinerU2.5-Pro-1.2B VLM 단독
- **hybrid-engine**: pipeline+VLM 결합(3.4.4 기본값)

## 결과 (핵심)

| qid | 페이지 유형 | pipeline | vlm-engine | hybrid |
|---|---|---|---|---|
| vp_006 | 래스터 UI 표(대여이력 화면캡처) | **O** 4/4 | X 0/4 | X 3/4 |
| vp_007 | 래스터 UI 표(CSV 업로드 샘플) | X 1/2 | X 0/2 | X 1/2 |
| vp_013 | 래스터 카드 UI(단말현황 카드) | X 0/2 | X 0/2 | X 0/2 |
| **vp_009** | 벡터 도표(앱 소개 그림) | X 0/2 | **O 2/2** | O 2/2 |
| **vp_010** | 벡터 구성도(장애진단 4단계) | X 0/3 | **O 3/3** | X 0/3 |
| **vp_014** | 벡터 다이어그램(개선과제 영역) | X 0/3 | **O 3/3** | X 1/3 |
| **합계** | | **1/6 · 5/16토큰** | **3/6 · 8/16토큰** | 1/6 · 7/16토큰 |

## 판정: 계획 가설은 **방향이 반대**였다

**단일 백엔드 승자는 없다. 페이지 유형별로 최적 백엔드가 갈린다:**

1. **벡터 도표/구성도/다이어그램 → vlm-engine 압승.** 현행 pipeline OCR은 이 페이지들에서 84~176자·gold 0개(사실상 텍스트 없음)라 검색·답변이 원천 불가능했다. vlm-engine은 라벨을 **정확한 구조화 텍스트**로 추출한다:
   - vp_010: "방화벽 → 무선 집선 스위치 → PoE 스위치" 순서 + IP 예시까지 텍스트화.
   - vp_014: **mermaid 다이어그램**으로 구조화(`공통플랫폼{수집/분석,빅데이터,AI,인증}`, `네트워크{노후 네트워크 장비 대개체, 망중계 장치 대개체}`) — gold 정답과 정확히 일치.
   → 이 클래스는 그동안 vision 경로에서 오독되던 주 대상. **텍스트화하면 vision 오독 문제 자체가 사라진다.**

2. **래스터 UI 스크린샷 표 → vlm-engine 회귀.** 소프트웨어 화면을 캡처해 이미지로 삽입한 페이지(vp_006/007)는 VLM이 전사 대신 **영어 캡션**("a list of user IDs...")을 달아 데이터를 잃는다. 현행 pipeline OCR이 셀 토큰은 더 잘 잡는다(단 rowspan/colspan 구조가 붕괴돼 관계 복원이 어렵고, 시리얼 일부 오독 `R54KB00TXWF→RS4KB0OTXWF`). hybrid는 vp_006 토큰(3/4)은 회복하나 도표에선 vlm-engine의 풍부함을 잃어 불안정.

3. **래스터 카드 UI(vp_013)** — 작은 카드 안 시리얼/MAC은 세 백엔드 모두 실패. **복구 불가 잔여**.

**즉 계획이 겨눈 "스캔 표"는 vlm-engine이 오히려 회귀시키고, 실제로 vlm-engine이 텍스트화하는 것은 "벡터 도표"다.** Phase 3 전략을 데이터에 맞게 재정의한다.

## Phase 2 실패와의 교차 대조 (근본 원인 확정)

phase2_eval.json 대조로 6건의 실패 메커니즘이 정확히 분리됐다:

| qid | Phase2 경로/결과 | 근거 top3 | 근본 원인 | Phase 3 처방 |
|---|---|---|---|---|
| vp_006 | text/abstain | 있음 | OCR 구조 붕괴 → 12b 정당 abstain | 래스터 잔여(부분) |
| vp_007 | text/abstain | 있음 | 동일 | 래스터 잔여(부분) |
| vp_009 | vision/**오독**(개요→캐너) | 있음 | vision 오독 | **vlm 재파싱→text** |
| vp_010 | text/abstain | **없음** | 검색 미스 + 도표 텍스트 부재 | **vlm 재파싱→text**(+Phase1) |
| vp_013 | text/abstain | 있음(OCR 공백) | 카드 UI, 데이터 없음 | 복구 불가 |
| vp_014 | text/abstain | **없음** | 검색 미스 + 도표 텍스트 부재 | **vlm 재파싱→text**(+Phase1) |

- **환각은 여전히 0.** vp_006/007/010/013/014는 못 읽으면 abstain(정직한 거절), 유일한 실제 오독은 vp_009(vision).
- vp_010/014는 **검색 미스(ev=0)**가 겹쳐 있다 — 도표를 텍스트화하면 검색 앵커가 생겨 Phase 1 page_hit도 동반 개선된다(부수 효과).

## 규모·비용 (코퍼스 실측)

- 969페이지 중 **figure-dominant(이미지 블록 有 & 텍스트<200자) = 117p(12%)**. 상위: 23년안내서 36p, 이용관리 20/19/10p, 무선랜 8p 등.
- vlm-engine transformers 속도 ≈ 페이지당 1~2분(모델 로드 1회 ~230s는 상각). **targeted(117p figure만) ≈ 2~4시간**, 전체 969p ≈ 16~32시간(모두 1회성 오프라인).
- 모델: MinerU2.5-Pro-1.2B, VRAM은 12b 축출 후 단독 실행(동시 GPU 프로세스 금지 교훈 준수). 다운로드 완료(HF 캐시).
- **동시성 버그**: vlm-engine 배치 동시성(기본 3)에서 텐서 크기 오류로 전멸 → `MINERU_API_MAX_CONCURRENT_REQUESTS=1`로 순차 처리해 해결(재파싱 스크립트에 고정 필요).

## Phase 3 수정 전략 (데이터 기반)

**주 전략: figure 페이지 vlm-engine 재파싱 → pipeline 캐시에 병합(page-type routing).**
1. pipeline이 이미 파싱한 content_list에서 **figure-dominant 페이지(117p)를 식별**.
2. 그 페이지만 vlm-engine으로 재파싱(순차, concurrency=1), 산출 텍스트/mermaid를 해당 page_idx의 text 블록으로 **병합**(래스터 표 페이지는 건드리지 않아 회귀 없음).
3. chunking·ingest 재실행 → 도표 페이지가 검색·답변 가능한 텍스트를 획득, 라우팅은 **text**(vision 오독 제거).
4. **래스터 UI 표(vp_006/007)**: pipeline OCR 유지 + (선택) ask-time 타일링 vision을 저신뢰 보조로. **카드 UI(vp_013)**: 복구 불가로 abstain 수용.

**부차: render.py 타일링**은 P0-A 확증대로 도표 vision의 잔여 보조로만(재파싱으로 대부분 흡수되므로 우선순위 하향).

## 산출물

- 원시결과: `test_3/probes/results/p0b_mineru25.json`
- 프로브: `tmp/p0b_extract_pages.py`, `tmp/p0b_consolidate.py`, `tmp/p0b_out*/`(파싱 산출)
- vlm-engine 도표 추출 예시(mermaid): `tmp/p0b_out/vp_014_p71/vlm/vp_014_p71.md`
