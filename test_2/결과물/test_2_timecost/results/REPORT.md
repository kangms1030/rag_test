# test_2 카탈로그 유무 비교 실험 (catalog vs no_catalog)

- 생성 시각: 2026-07-13T16:31:07.460215+00:00
- 평가 문항 수: 16 (`rag_eval_dataset.json`, core 13 + irrelevant 3)
- 대상 파이프라인: `test_2/rag2` (수정 없이 import), 인덱스/캐시는 기존 ingest 결과 재사용

## 요약 비교표

| mode | 문항 | doc_hit | page_hit | kw_hit | 무관거절 | 경로(text/vision/none) | avg_docs | avg_모델호출 | avg_초 | 총_초 |
|---|---|---|---|---|---|---|---|---|---|---|
| catalog | 16 | 0.692 | 0.538 | 0.423 | 1.000 | 11/0/5 | 1.375 | 1.688 | 35.339 | 565.420 |
| no_catalog | 16 | 0.923 | 0.769 | 0.705 | 1.000 | 12/1/3 | 1.312 | 1.812 | 30.491 | 487.860 |

## 문항별 상세 (doc_match / page_match / answer_path / 소요초)

| id | catalog.doc | no_catalog.doc | catalog.page | no_catalog.page | catalog.path | no_catalog.path | catalog.sec | no_catalog.sec |
|---|---|---|---|---|---|---|---|---|
| core_001 | True | True | True | True | text | text | 40.5 | 29.3 |
| core_002 | False | True | False | False | none | vision | 0.2 | 28.8 |
| core_003 | False | True | False | True | text | text | 140.1 | 18.2 |
| core_004 | True | True | True | True | text | text | 24.5 | 48.7 |
| core_005 | True | True | False | True | text | text | 51.4 | 50.5 |
| core_006 | True | True | True | True | text | text | 29.9 | 31.3 |
| core_007 | False | False | False | False | text | text | 26.7 | 25.9 |
| core_008 | True | True | True | True | text | text | 46.1 | 69.1 |
| core_009 | True | True | False | True | text | text | 121.2 | 22.8 |
| core_010 | True | True | True | True | text | text | 36.0 | 33.2 |
| core_011 | True | True | True | True | text | text | 15.3 | 20.3 |
| core_012 | False | True | False | False | none | text | 0.2 | 53.4 |
| core_013 | True | True | True | True | text | text | 32.5 | 46.4 |
| irrelevant_001 | - | - | - | - | none | none | 0.2 | 3.3 |
| irrelevant_002 | - | - | - | - | none | none | 0.2 | 3.3 |
| irrelevant_003 | - | - | - | - | none | none | 0.2 | 3.4 |

## baseline 정합성 대조

catalog 모드는 이 실험 코드가 아니라 `rag2.retrieve.run_retrieval`을 그대로 호출한 것이므로,
`test_2/rag2/outputs/evaluation_20260713T100344Z.json`(기존 baseline 평가)과 완전히 같은
경로를 타야 한다. 실측 대조:

| 지표 | baseline | 이 실험(catalog) |
|---|---|---|
| doc_hit_rate | 0.692 | 0.692 |
| page_hit_rate | 0.538 | 0.538 |
| avg_keyword_hit_rate | 0.423 | 0.423 |
| irrelevant_correctly_rejected_rate | 1.0 | 1.0 |
| answer_path_counts | text 11/vision 0/none 5 | text 11/vision 0/none 5 |
| avg_elapsed_seconds | 35.47 | 35.34 |

정확도 지표는 **완전히 일치**(같은 결정론적 검색+게이트 로직이므로 당연), 소요시간만 실행마다
자연 변동하는 수준(35.47 vs 35.34초) — 실험 배선이 기존 프로덕션 경로와 동일함을 확인했다.

## 해석 — 카탈로그가 test_2에서는 왜 다른 결과를 냈는가

test_1차(§10, VLM 기반 query-time 파이프라인)에서는 카탈로그가 정확도·비용 양쪽에서
뚜렷이 우세했다. test_2(rag2)에서는 **반대로 no_catalog가 doc_hit(0.923 vs 0.692)·
page_hit(0.769 vs 0.538)·키워드 recall(0.705 vs 0.423) 전부에서 우세했고, 평균 소요시간도
오히려 더 짧았다(30.5초 vs 35.3초)**. 무관 질문 거절은 두 모드 모두 완벽했다(3/3).

**원인 1 — 비용 구조 자체가 다르다.** test_1의 no_catalog가 비쌌던 이유는 문서 범위를
좁히지 않으면 다수 페이지에 대해 **query-time에 VLM 요약을 새로 호출**해야 했기 때문이다.
test_2는 ingest 단계에서 969페이지 전부를 MinerU로 이미 파싱해 `page_index`에 넣어뒀으므로,
카탈로그로 문서를 먼저 좁히든 안 좁히든 **페이지 검색 자체의 비용은 동일**하다(BM25+dense는
969페이지 전체를 훑어도 임베딩 1회 수준). 두 모드의 실질적 비용 차이는 마지막 답변 호출
1회(text/vision)뿐이라, 카탈로그가 없앨 수 있는 비용이 test_1보다 훨씬 작다.

**원인 2 — 카탈로그 게이트가 정답 문서를 놓친 사례가 실제로 존재.** 문항별 표에서 catalog가
틀리고 no_catalog가 맞춘 3건(core_002/003/012)을 보면:
- **core_002** (UTM 장비 제조사) — catalog는 게이트 자체에서 거절(`dense_similarity 0.342 <
  0.35`), 즉 카탈로그 설명문에 "UTM"이라는 단어가 없어 후보에도 못 들었다. no_catalog는
  페이지 본문 텍스트에 "UTM"이 그대로 있어 바로 찾았다.
- **core_003** (사용자 요구사항 5가지) — catalog가 **완전히 다른 문서**(현황분석서·이용관리
  매뉴얼)를 선택한 반면, no_catalog는 정답 문서(통합관제시스템 3번)와 그 형제 문서(이용관리
  시스템 4번)를 함께 찾아 정답을 포함시켰다.
- **core_012** (최고관리자 대시보드 Traffic TopN) — catalog는 게이트에서 완전히 거절했지만,
  no_catalog는 통합관제(사용자)·통합관제(최고관리자) 형제 문서 둘 다 찾아 정답 문서를 포함시켰다.

세 건 모두 test_1차 REPORT.md §11.4가 이미 지목한 한계(**카탈로그 설명문은 "문서가 전반적으로
무엇을 다루는가"만 담고, 문서 내부의 특정 세부 용어·수치까지는 반영하지 못함**, "형제 문서"
혼동)와 정확히 같은 패턴이다. 다만 test_2는 문서 선정을 카탈로그 설명문이 아니라 **문서
원문 텍스트 자체**로 하는 `page_index`가 이미 있어서, 카탈로그가 놓치는 세부 사항을
no_catalog가 원문 검색으로 메울 수 있었다 — test_1에는 이런 "저비용 전역 원문 검색" 대안이
없었다(있었다면 VLM 요약 비용이 폭증했을 것).

**결론**: "카탈로그가 있으면 무조건 유리하다"는 test_1의 결론은 **파이프라인 구조에
의존적**이다. query-time에 비싼 모델 호출(VLM 요약 등)이 필요한 구조에서는 카탈로그로 검색
범위를 좁히는 것 자체가 비용을 크게 줄인다. 반면 test_2처럼 **모든 페이지를 이미 저비용으로
사전 색인**해 둔 구조에서는, 카탈로그가 오히려 (a) 설명문에 없는 세부 키워드 질문을 조기에
거절하거나 (b) 형제 문서를 하나로 좁혀 오답을 유발하는 손해가, VLM 비용 절감이라는 이점보다
커질 수 있다. test_2 규모(13문서·969페이지)에서는 그냥 페이지 전역 검색이 더 정확하고
더 빠르다.

## 한계

- 문항 16개(irrelevant 3개 포함)로 표본이 작다 — core_002/003/012의 방향성은 명확하지만
  통계적으로 유의하다고 주장하기는 어렵다.
- catalog/no_catalog를 순차 실행했다(캐시 문제는 없음 — page_index는 이미 완전히
  사전연산되어 있어 실행 순서가 결과에 영향을 주지 않는다). 다만 Ollama 응답 시간 자체의
  실행별 변동(core_003: catalog 140.1초 vs no_catalog 18.2초)이 커서, 개별 문항의 초 단위
  비교보다는 `avg_elapsed_seconds`/`total_elapsed_seconds` 총합으로 판단하는 것이 안전하다.
- no_catalog도 page_match 기준으로는 아직 완벽하지 않다(0.769) — 정답 페이지 자체를
  더 정밀하게 못 맞추는 문항이 남아 있다(core_002/005/009/012 등, 문서는 맞혔지만 페이지는
  못 맞춘 경우 다수).
- 카탈로그 13행이라는 소규모 자체가 원인일 수 있다 — 문서 수가 훨씬 많아지면(예: 249행
  전체) 카탈로그의 "문서 범위 좁히기" 효과가 다시 우세해질 가능성이 있다(전역 페이지 검색은
  문서 수에 비례해 후보가 희석되므로).
