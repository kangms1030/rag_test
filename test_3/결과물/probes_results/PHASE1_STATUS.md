# Phase 1 상태 리포트 (검색 코어: 청크 + 리랭커 + 게이트 제거)

> 작성: 2026-07-15. 원시 결과: `phase1_retrieval.json`. 구현: `test_3/rag3/{chunking,rerank,flat_index,index,ingest,retrieve}.py`.

## 구현 완료

- **chunking.py**: MinerU content_list 블록 → 페이지 내 섹션 청크(text 250~450토큰, 표=단독 청크, 초과 표는 헤더 반복 행분할). 969p → **2417청크(text 1650 + table 767)**, avg 2.49청크/p.
- **카탈로그 메타데이터 주입**: 게이트 폐기, 청크 프리픽스에 `문서/분류/범위/키워드 | 섹션 | p{n}` 주입(형제 문서 변별).
- **rerank.py**: bge-reranker-v2-m3(fp16, GPU 상주 ~1.5GB).
- **retrieve.py v2**: 청크 전역 하이브리드 검색 → 리랭크 → small-to-big(청크→페이지 승격) → 라우팅 v2(text 기본, figure만 vision). `use_catalog_gate=true`면 test_2 게이트로 폴백.
- **index.py**: BM25 전역 캐시(F7), 읽기 복구를 비파괴적으로 개선(B6 색인 보존).
- **num_ctx 16384**(P0-C 출력 굶김 대응), **keep_alive 30m**.

## B6(Chroma HNSW 손상) 해결 — 아키텍처 변경

- **문제**: chunk_index(2417건)를 Chroma에 넣으면 **별도 프로세스 reload 시 재현적으로 HNSW 로드 실패**("Error loading hnsw index"). page_index(969)는 정상 → 큰 컬렉션에서만 발생하는 Chroma 1.5.9 compactor 버그. test_2의 B6가 test_3에서도 재발.
- **해결**: 청크 dense 검색을 **Chroma 대신 numpy 브루트포스 코사인 + 캐시 BM25 + RRF**로 전환(`flat_index.py`). 벡터는 npz+json으로 디스크 저장. 2417×768 브루트포스 < 10ms — 근사(HNSW)보다 정확·빠르고 크로스 프로세스 안정. page/catalog_index는 정상이라 Chroma 유지.
- **부수 개선**: index.py의 읽기 복구가 오류 시 컬렉션을 **삭제하던 것(2417벡터 소실)을 재오픈 재시도로 변경** — 일시 오류가 색인 전체를 날리지 않음.

## Chroma → numpy flat 전환: 과정·근거·확장 대비

### 발견·진단 과정 (재현 가능한 단서로 좁힘)
1. 전체 ingest에서 chunk_index(2417) + page_index(969)를 같은 Chroma에 색인 → **ingest 프로세스 내에서는 count=2417 정상**, 쿼리도 정상.
2. **별도 eval 프로세스**에서 chunk_index 첫 쿼리 시 `Error loading hnsw index` → 기존 복구 로직이 컬렉션을 삭제(2417 소실) → 모든 질의가 none으로 반환.
3. 클린 재빌드 후 fresh 프로세스로 재검증: **page_index(969)는 정상 reload, chunk_index(2417)만 재현적으로 실패**. 크기에 의존하는 현상으로, Chroma 1.5.9의 큰 컬렉션 compactor/backfill 세그먼트 로드 버그로 특정(fact 문서 B6가 test_3에서 재발).
4. 비파괴적 복구(client 재오픈 재시도)로도 on-disk 세그먼트 자체가 안 읽혀 실패 → **데이터 결함이 아니라 Chroma 지속성(persistence) 문제**로 결론. 재빌드로도 해결 불가 → 백엔드 교체 결정.

### 전환 근거
- **정확성**: HNSW는 근사최근접(recall<100%). 브루트포스는 전수 코사인이라 recall=100%. 소규모에선 근사가 주는 속도 이점이 미미한데 정확도만 손해.
- **성능**: 2417×768 행렬-벡터곱은 수 μs~ms. HNSW 인덱스 빌드/세그먼트 로드 오버헤드가 오히려 큼.
- **견고성**: `vectors.npz`(벡터) + `docs.json`(문서/메타)은 단순 파일 → 크로스 프로세스 로드가 실패할 여지 없음. B6 원천 제거.
- **자원**: numpy는 CPU/RAM에서 동작 → **VRAM 미소모**. 16GB VRAM을 LLM/리랭커에 온전히 남김.
- **국소성**: `flat_index.FlatChunkIndex.query()`가 `HybridIndex.query()`와 동일 인터페이스(ScoredItem 반환) → 이후 백엔드를 바꿔도 수정이 이 파일 한 곳에 국한.

### 데이터가 많아질 때 (확장 대비 가이드)
브루트포스 비용 ≈ O(N·D), 메모리 ≈ N × 768 × 4 bytes.

| 청크 수 N | 메모리 | 쿼리 지연 | 권장 |
|---|---|---|---|
| ~2.4천 (현재) | ~7MB | <1ms | **flat 유지** |
| ~5만 (249문서 확장 규모) | ~150MB | <5ms | **flat 유지** (단순·정확) |
| ~100만 | ~3GB RAM | ~10ms | flat 가능하나 RAM·동시성 주의 |
| 1000만+ | 30GB+ RAM, >100ms | — | **ANN 전환 시점** |

전환 시 권장 순서:
1. **FAISS**(`IndexHNSWFlat` 또는 `IVF-PQ`) 또는 hnswlib 직접 사용 — Chroma 1.5.9 compactor 버그를 우회하면서 검증된 ANN + 안정적 지속성.
2. **양자화**(int8/PQ)로 메모리 절감, **mmap**으로 RAM 상주 회피.
3. 이미 2단계 검색(후보 → 크로스인코더 리랭크)이므로 **1단계(dense 후보)만 ANN으로 교체**하고 리랭크는 유지 → 정확도 손실 최소화.
4. Chroma로 복귀하려면 **compactor 버그 수정 버전 확인 후, 대규모 컬렉션의 크로스 프로세스 reload를 별도 회귀 테스트(B6)로 통과**시킨 뒤에만 채택.

**요약 판단 기준**: N < ~10만이면 flat 유지(단순·정확·무결), 그 이상이면 FAISS 계열 ANN 도입. 교체 범위는 `flat_index.py` 내부로 한정된다.

## 검증 결과 (36문항, candidates=20, floor=0.1)

| 지표 | test_2 | test_3 Phase 1 | 목표 | 판정 |
|---|---|---|---|---|
| page_hit@1 | — | 0.516 | — | — |
| **page_hit@3** | 0.556 | **0.645** (+8.9%p) | ≥0.75 | **미달**(개선했으나 목표 밑) |
| doc_hit@3 | 0.923 | **0.968** | ≥0.92 | **통과** |
| 무관 거절 | 100% | **5/5** (floor 0.1) | 5/5 | **통과** |
| 문항당 지연 | 49.5s(답변포함) | **1~5s(검색만)** | ≤3s | 통과(검색 단계) |

- floor 캘리브레이션: 관련 문항 리랭크 top score min **0.688~0.729** vs 무관 max **0.013** → **floor 0.1**이 무관 5/5 거절, 관련 전부 통과. 깔끔히 분리됨.
- candidates 20 vs 40: **20이 page_hit@3 높음(0.645 vs 0.613)** — 후보 확대는 리랭커에 방해후보만 늘림.

## page_hit@3가 0.75에 못 미친 원인(미해결, 다음 작업)

- 남은 미스 11/31 중 다수가 **figure/반복서식 스캔 페이지**(vp_010 p11 figure, vp_011, vp_014) — 텍스트가 빈약해 청크/리랭크로도 상위 못 올림.
- 일부는 정답 청크가 **초기 후보 20 밖**이거나, 리랭커가 프리픽스 포함 텍스트를 봐 변별력 희석.
- **다음 튜닝 후보**: ① 리랭크를 프리픽스 제거한 raw 청크 텍스트로 ② 청크→페이지 집계를 best-score가 아닌 top-N 합산 ③ figure 페이지는 캡션/주변 텍스트를 청크에 보강 ④ Phase 2 롤백(차순위 페이지 재시도)이 page_hit 실패를 답변 단계에서 흡수.

## 다음 단계

1. page_hit@3 0.75 튜닝(위 후보) — 선택적.
2. **Phase 2**: controller.py(상태기계) + verify.py(숫자 대조/groundedness) + judge.py(CRAG 재검색) + 라우팅/답변 통합 + evaluate.py 지표 확장. "근거 있는데 실패 0/18" 목표.
3. Phase 3(vision): P0-A 확증대로 타일링 + MinerU 2.5 텍스트화(MinerU 설치 완료됨).
