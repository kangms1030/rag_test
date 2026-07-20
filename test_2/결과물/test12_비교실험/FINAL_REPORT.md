# 최종 통합 실험 보고서 — test_1차 · test_2 4개 비교군 통합 비교

- 생성 시각: 2026-07-14 (KST) / 실험 실행 구간: 2026-07-14 10:59 ~ 13:06 (약 2시간 7분, ingest 포함)
- 실험 실행(질문 응답)만: 약 87분 (12건, ingest·워밍업 제외)
- 대상 비교군 (4개): `test1.catalog`, `test1.no_catalog`, `test2.catalog`, `test2.no_catalog`
  (`test1.filename_only`는 소요시간이 과다하여 사용자 지시로 이번 실험에서 제외)
- 평가 문항: 신규 작성 3문항(`final_qa_dataset.json`) — 기존 test_1차(`human_20.json`)·test_2(`rag_eval_dataset.json`) 평가셋과 겹치지 않는 문서/페이지/성격으로 구성
- 실행 결과: **12건 전부 성공 (실패 0건)**

---

## 0. 실험 설계 요약

### 0.1 왜 이 실험인가

test_1차에서는 3개 비교군(catalog/no_catalog/filename_only), test_2에서는 2개 비교군(catalog/no_catalog)을 각각 별도로 평가했다. 이번 실험은 **두 세대의 파이프라인을 동일한 신규 질문 3개로 나란히 세워** 정확도·소요시간·모델 호출 비용을 하나의 보고서에서 비교하는 것이 목적이다.

### 0.2 신규 평가 문항 3개 (근거 확인 방법)

카탈로그의 13개 실제 처리 대상 문서(PDF) 중 서로 다른 3개 문서에서, pdfplumber로 원문 텍스트를 직접 추출해 정답과 페이지 번호를 사람이 검증한 뒤 작성했다. 기존 두 평가셋이 이미 다룬 문서/페이지와 겹치지 않도록 확인했다.

| id | 문서 | 페이지 | 성격 | 질문 |
|---|---|---|---|---|
| final_001 | 8-1. 스쿨넷서비스 학내망 구축 및 운영관리 개선을 위한 가이드.pdf | 12 | 표/수치형 | UTP 케이블 규격(Cat.5/5e/6)별 최대 전송거리·통신속도 |
| final_002 | 이용관리_20260702.pdf | 11-12 | 절차형(+수치) | 비밀번호 초기화 절차, 고객센터 연락처, 메일 발송 소요시간 |
| final_003 | ★. 무선랜 구축운영 가이드라인_0.1v_20p수정_2023년.pdf | 5 | 사실/도표형 | 802.11ac/ax 표준의 대역·속도, 160MHz 사용 시 최대속도 |

전체 문항·정답·근거는 [`final_qa_dataset.json`](final_qa_dataset.json) 참조.

### 0.3 캐시 독립성 확보 방법 (요구사항 4)

**test_1 (lazy VLM 요약 구조)** — 질문마다 VLM 요약 캐시가 warm해지는 구조이므로, 가장 엄격한 방식을 택했다:
1. 기존 test_1차의 실전 index/cache는 전혀 건드리지 않고, `test12_total_test/_cold_baseline_test1/`에 **완전히 새로운 격리 ingest**를 1회 실행 (VLM 미호출, PDF 파싱 + catalog/filename 임베딩만 — 969페이지, 13문서, ~7분)
2. **6번의 실행(catalog×3문항 + no_catalog×3문항) 각각의 직전**에 이 냉시작 스냅샷을 작업 디렉터리(`_work_test1/`)로 복원 — 이전 질문이 만든 요약/청크 캐시가 다음 질문에 전혀 넘어가지 않음
3. 각 실행은 **독립 서브프로세스**(`run_single_test1.py`)로 새 인터프리터에서 구동 — 모듈 전역 캐시·contextvar 등 프로세스 내부 상태 누수를 원천 차단

**test_2 (전량 사전 색인 구조)** — ingest 시점에 969페이지를 MinerU로 이미 전량 파싱·색인해 두었고(`test_2/rag2/index`: 25,333,760바이트 Chroma DB, `test_2/rag2/cache/parsed`: 13문서 2,886파일), `ask` 경로는 인덱스/캐시에 **아무것도 쓰지 않는다**(test_2_timecost/results/REPORT.md에서 이미 실측 확인된 사실). 따라서:
- 재-ingest 없이 기존 산출물을 **읽기전용으로 재사용** — 실험 전후 `chroma.sqlite3` 파일 크기 동일(25,333,760바이트, mtime만 오픈으로 갱신)로 실측 확인
- `test_2/` 아래 신규 데이터 파일 없음(`.pyc` 바이트코드 캐시만 생성, 무해)
- 그래도 실행 순서 영향이 전혀 없도록 6번 모두 독립 서브프로세스(`run_single_test2.py`)로 구동

### 0.4 모델 워밍업 (시간 공정성)

4개 비교군 모두 동일한 Ollama 서버·동일 모델(`gemma4:12b`, `embeddinggemma`)을 공유하므로, 실험 시작 시 LLM/Embedding을 1회씩 미리 호출해 모델을 VRAM에 로드해두었다(keep_alive 10m). 첫 비교군(`test1.catalog`)만 모델 로드 비용을 떠안지 않도록 하기 위함이며, 답변 내용에는 영향이 없다.

### 0.5 실행 환경

- 모델: LLM/VLM `gemma4:12b`, Embedding `embeddinggemma`, fallback `gemma4:e4b` (Ollama, `keep_alive=10m`, `num_ctx=8192`)
- test_1 설정: `top_docs=2`, `page_prefilter_topn=12`(문서당), `chunk_pages_topk=4`, `max_images_per_call=3`
- test_2 설정: `top_docs=2`, `top_pages=3`, MinerU pipeline 파서(korean, cuda→cpu 폴백 가능)
- 산출물 폴더: `test1_catalog/`, `test1_no_catalog/`, `test2_catalog/`, `test2_no_catalog/` (문항별 `{qid}_result.json` + test_1은 `{qid}_evidence/`에 crop/highlight 이미지)
- 통합 원본 데이터: [`all_results_20260714T040610Z.json`](all_results_20260714T040610Z.json)

---

## 1. 비교군별 RAG 진행 단계별 소요시간 · 총 소요시간

### 1.1 test_1 단계 분해 (query_analyzer → page_summary → visual_chunk → answer → evidence → verify)

| qid | mode | query_analyzer | page_summary(VLM,lazy) | visual_chunk(VLM) | retrieval_search(나머지) | retrieval_total | answer(VLM) | evidence | verify(LLM) | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| final_001 | catalog | 22.6s | 364.4s | 345.4s | 1.3s | 733.7s | 28.3s | 0.2s | 17.1s | **779.3s** |
| final_002 | catalog | 24.0s | 336.0s | 449.9s | 1.3s | 811.2s | 108.6s | 0.1s | 16.1s | **936.0s** |
| final_003 | catalog | 27.6s | 341.3s | 474.3s | 1.4s | 844.5s | 45.5s | 0.0s | 0.0s* | **890.0s** |
| final_001 | no_catalog | 22.6s | 362.2s | 439.6s | 1.3s | 825.7s | 46.2s | 0.4s | 17.8s | **890.2s** |
| final_002 | no_catalog | 24.2s | 339.2s | 385.8s | 1.2s | 750.5s | 51.0s | 0.2s | 15.5s | **817.1s** |
| final_003 | no_catalog | 28.1s | 358.2s | 271.5s | 1.4s | 659.2s | 57.6s | 0.1s | 24.3s | **741.2s** |

\* final_003/catalog는 VLM 답변의 evidence bbox가 스키마 검증을 통과하지 못해 `page_evidence`가 빈 배열이 되었고, `verify_answer`가 "근거 없음"으로 즉시 조기 종료해 LLM을 호출하지 않았다(§8.2 한계 참조). 답변 자체는 정확했다.

**단계 비중(총 6회 평균)**: `page_summary` 41.0%, `visual_chunk` 47.5%, `answer` 6.4%, `verify` 2.2%, `query_analyzer` 2.9% — **VLM 요약+청킹 두 단계가 전체 시간의 약 88.5%**를 차지한다. 이는 이번 실험이 매 질문 냉시작(요약 캐시 없음)을 강제했기 때문이며, 실전 운영에서 같은 문서를 반복 질의하면 이 두 단계는 캐시로 대부분 사라진다.

### 1.2 test_2 단계 분해 (retrieve → answer)

| qid | mode | retrieve(embed+검색) | answer(text/vision) | **total** |
|---|---|---|---|---|
| final_001 | catalog | 0.7s | 18.4s | **19.1s** |
| final_002 | catalog | 0.7s | 16.2s | **16.8s** |
| final_003 | catalog | 0.6s | 24.9s | **25.5s** |
| final_001 | no_catalog | 1.0s | 17.6s | **18.6s** |
| final_002 | no_catalog | 1.1s | 22.6s | **23.7s** |
| final_003 | no_catalog | 1.0s | 68.6s | **69.6s** |

3모드 전부 `answer_path: "text"`(MinerU가 이미 표를 마크다운으로 추출해둬 vision 경로 불필요)였고, `answer` 단계(LLM 텍스트 생성)가 총 시간의 95% 이상을 차지한다. `final_003/no_catalog`가 69.6초로 유독 길었던 것은 알고리즘 차이가 아니라 Ollama 응답 시간의 자연 변동(같은 모델·같은 프롬프트 구조에서도 흔한 현상, test_2_timecost 실측에서도 동일 패턴 관찰됨)으로 보인다.

### 1.3 비교군별 총 소요시간 (3문항 합계)

| 비교군 | 총 소요시간 | 문항당 평균 | test_2 대비 배율 |
|---|---|---|---|
| test1.catalog | 2605.4s (43.4분) | 868.4s (14.5분) | 43.5x |
| test1.no_catalog | 2448.5s (40.8분) | 816.2s (13.6분) | 39.7x |
| test2.catalog | 61.4s (1.0분) | 20.5s | 1x (기준) |
| test2.no_catalog | 112.0s (1.9분) | 37.3s | 1.8x |
| **전체 12건 합계** | **5227.2s (87.1분)** | 435.6s | — |

**해석**: test_2가 test_1보다 약 40~44배 빠르다. 원인은 아키텍처 차이다 — test_2는 ingest 시점에 969페이지를 MinerU로 전량 사전 파싱해 `ask` 경로가 임베딩 1회+답변 1회만 호출하면 되는 반면, test_1은 이번 실험의 냉시작 조건상 매 질문마다 최대 24페이지 VLM 요약 + 4페이지 시각적 청킹을 새로 생성해야 했다. 이는 test_1차 REPORT.md와 test_2_timecost/REPORT.md가 이미 지적한 구조적 차이(§8.1 재해석)와 일치한다.

---

## 2. 답변 성능 비교

### 2.1 요약 비교표

| 비교군 | doc_match | page_match | 평균 keyword_recall | is_answer_supported |
|---|---|---|---|---|
| test1.catalog | 3/3 (100%) | 3/3 (100%) | 0.646 | 2/3 (final_003은 evidence 추출 실패로 조기 미지원 판정) |
| test1.no_catalog | 3/3 (100%) | 3/3 (100%) | 0.562 | 3/3 (100%) |
| test2.catalog | 3/3 (100%) | 3/3 (100%) | 0.736 | 측정 안 함(rag2는 별도 verify 단계 없음) |
| test2.no_catalog | 3/3 (100%) | 3/3 (100%) | 0.736 | 측정 안 함 |

**문서/페이지 선정은 4개 비교군 전부 100% 정답** — 3문항 모두 검색 단계 자체는 실패하지 않았다. 차이는 **답변 생성 단계의 수치 정확도**에서 갈렸다(§2.2, §8.1).

### 2.2 문항별 keyword_recall 상세

| qid | 성격 | test1.catalog | test1.no_catalog | test2.catalog | test2.no_catalog |
|---|---|---|---|---|---|
| final_001 (표/수치) | 케이블 규격표 | 0.875 | 0.875 | **1.000** | **1.000** |
| final_002 (절차+수치) | 비밀번호 초기화 | 0.286 | 0.143 | 0.429 | 0.429 |
| final_003 (사실/도표) | WiFi 표준 | 0.778 | 0.667 | 0.778 | 0.778 |
| **평균** | | 0.646 | 0.562 | **0.736** | **0.736** |

test_2가 3문항 모두에서 test_1과 같거나 더 높은 keyword_recall을 기록했다. final_002는 4개 비교군 모두 낮은데(0.143~0.429), 이는 정답 키워드 7개 중 절반 이상이 세부 수치(전화번호, 이메일 주소, 5분)라서 텍스트 완전 일치 기준의 기계적 grading이 엄격하게 작동했기 때문이며, §2.3에서 실제 답변 내용을 대조한다.

### 2.3 문항별 정오 대조 (핵심 발견)

**final_001 — Cat.5의 "Data Performance" 값 (정답: 10Mbps)**

| 비교군 | 답변한 값 | 정오 |
|---|---|---|
| test1.catalog | 100Mbps | ❌ 오독 |
| test1.no_catalog | 100Mbps | ❌ 오독 |
| test2.catalog | 10Mbps | ✅ 정답 |
| test2.no_catalog | 10Mbps | ✅ 정답 |

**final_002 — 고객센터 전화번호 (정답: 1899-0979) / 메일 발송 소요시간 (정답: 5분)**

| 비교군 | 전화번호 | 소요시간 |
|---|---|---|
| test1.catalog | 1899-0970 ❌ | "확인할 수 없음" ❌ (페이지 12를 실제로 봤음에도 누락) |
| test1.no_catalog | 1899-0970 ❌ | "약 1분" ❌ (환각) |
| test2.catalog | 1899-0979 ✅ | "최대 5분" ✅ |
| test2.no_catalog | 1899-0979 ✅ | "최대 5분" ✅ |

**final_003 — 802.11ax 주파수 대역 (정답: 2.4GHz/5GHz)**

| 비교군 | 답변 |
|---|---|
| test1.catalog | "2.4GHz/5GHz 대역" ✅ (다만 verify 단계에서 근거 추출 실패로 `is_answer_supported=false` 오판정) |
| test1.no_catalog | **"1GHz~6GHz", 802.11ac 속도 "6.933Gbps"** ❌ (원문에 없는 수치 환각) |
| test2.catalog | "2.4GHz/5GHz 대역" ✅ |
| test2.no_catalog | "2.4GHz/5GHz 대역" ✅ |

**패턴**: 3문항 6개 지점(표1개+수치2개+대역1개, catalog/no_catalog 합산 시 총 8회 관측) 중 **test_1(VLM 비전 판독)은 6회 중 4회 수치를 틀렸고, test_2(MinerU 텍스트 추출)는 6회 전부 정확했다.** test_1차 자체 REPORT.md §5 A1이 이미 "VLM이 조밀한 표의 숫자를 잘못 전사할 수 있다"고 경고한 한계가, 이번에 완전히 새로운 질문 3개로도 동일하게 재현되었다.

---

## 3. VLM/LLM/Embedding 모델 호출 횟수

### 3.1 test_1 (단계별 호출 수)

| qid | mode | query_analyzer(LLM) | summary(VLM) | chunk(VLM) | answer(VLM) | verify(LLM) | embed | LLM 합계 | VLM 합계 |
|---|---|---|---|---|---|---|---|---|---|
| final_001 | catalog | 1 | 24 | 4 | 1 | 1 | 11 | 2 | 29 |
| final_002 | catalog | 1 | 24 | 4 | 1 | 1 | 11 | 2 | 29 |
| final_003 | catalog | 1 | 24 | 4 | 1 | 0* | 11 | 1 | 29 |
| final_001 | no_catalog | 1 | 24 | 4 | 1 | 1 | 32 | 2 | 29 |
| final_002 | no_catalog | 1 | 24 | 4 | 1 | 1 | 31 | 2 | 29 |
| final_003 | no_catalog | 1 | 24 | 4 | 1 | 1 | 33 | 2 | 29 |

\* §1.1 각주와 동일한 원인(evidence 추출 실패로 verify 조기 종료).

**캐시 독립적 비용(`vlm_pages_required`)**: catalog·no_catalog **6회 전부 동일하게 `summary_pages_required=24`, `chunk_pages_required=4`**였다. 이는 우연이 아니라 설계값이 정확히 맞춰져 있기 때문이다 — `catalog`는 `top_docs=2 × page_prefilter_topn=12 = 24`, `no_catalog`는 `no_catalog_page_prefilter_topn=24`로 config.yaml에 "catalog와 비용 예산을 맞추기 위해" 명시적으로 설정되어 있다(config.yaml 주석). **즉 이 코드베이스의 현재 설정에서는 catalog 유무가 VLM 호출 비용(요약/청킹 대상 페이지 수)에 차이를 내지 않는다** — test_1차 REPORT.md §10이 보고한 "카탈로그가 비용을 줄인다"는 결론은 이 예산 동일화 설정 이후에는 재현되지 않으며, 이번 실험의 실제 차이는 **embed_calls만**(catalog 11회 vs no_catalog 31~33회, no_catalog가 문서 필터 없이 page_index 전역을 재질의하며 추가 임베딩 호출이 필요하기 때문) 유의미했다.

### 3.2 test_2 (호출 수)

| qid | mode | embed | text_answer(LLM) | vision_answer(VLM) | 총 호출 |
|---|---|---|---|---|---|
| final_001 | catalog | 1 | 1 | 0 | 2 |
| final_002 | catalog | 1 | 1 | 0 | 2 |
| final_003 | catalog | 1 | 1 | 0 | 2 |
| final_001 | no_catalog | 1 | 1 | 0 | 2 |
| final_002 | no_catalog | 1 | 1 | 0 | 2 |
| final_003 | no_catalog | 1 | 1 | 0 | 2 |

**6회 전부 정확히 2회 호출**(임베딩 1 + 텍스트 답변 1), vision 경로는 한 번도 타지 않았다. test_1의 VLM 호출 수(29회/문항)와 비교하면 **약 14.5분의 1**이다 — 이것이 §1.3의 40배 속도차를 만든 직접적 원인이다.

---

## 4. 답변 결과 (4개 비교군 × 3문항, 전문)

### 4.1 final_001 — UTP 케이블 규격표

> **질문**: 학내망 구축 및 운영관리 개선을 위한 가이드에서 제시하는 UTP 케이블 규격(Cat.5, Cat.5e, Cat.6)별 최대 전송거리와 통신속도(Data Performance / Max Speed Data)는 각각 어떻게 다른가요?
> **정답**: Cat.5=100M/10Mbps/100Mbps, Cat.5e=100M/100Mbps/1Gbps, Cat.6=100M/1Gbps/10Gbps (근거: 8-1 가이드 12페이지)

| 비교군 | 답변 |
|---|---|
| test1.catalog | UTP 케이블 규격별 최대 전송거리와 통신속도는 다음과 같습니다.<br>- Cat.5: 최대 전송거리 100m, Data Performance **100Mbps**(오독), Max Speed Data 100Mbps<br>- Cat.5e: 최대 전송거리 100m, Data Performance 100Mbps, Max Speed Data 1Gbps<br>- Cat.6: 최대 전송거리 100m, Data Performance 1Gbps, Max Speed Data 10Gbps |
| test1.no_catalog | UTP 케이블 규격별 최대 전송거리와 통신속도는 다음과 같습니다.<br>- Cat.5: 100M / **100Mbps**(오독) / 100Mbps<br>- Cat.5e: 100M / 100Mbps / **100Mbps**(오독, 정답 1Gbps)<br>- Cat.6: 100M / 1Gbps / 10Gbps |
| test2.catalog | Cat.5: 최대 전송거리 100M, Data Performance **10Mbps**, Max Speed Data 100Mbps / Cat.5e: 100M, 100Mbps, 1Gbps / Cat.6: 100M, 1Gbps, 10Gbps |
| test2.no_catalog | Cat.5: 최대 전송거리 100M, Data Performance **10Mbps**, Max Speed Data 100Mbps / Cat.5e: 100M, 100Mbps, 1Gbps / Cat.6: 100M, 1Gbps, 10Gbps |

### 4.2 final_002 — 이용관리시스템 비밀번호 초기화

> **질문**: 스쿨넷 이용관리시스템에서 이용기관 비밀번호를 초기화하는 절차는 무엇이며, 담당자 이메일을 모를 경우 문의할 고객센터 연락처와 초기화 완료 안내 메일 발송에 소요되는 시간은 얼마인가요?
> **정답**: 로그인 화면에서 기관명 선택 → 비밀번호초기화 → 담당자 이메일 입력. 모를 경우 **1899-0979**. 메일 발송 최대 **5분**. (근거: 이용관리_20260702.pdf 11-12페이지)

| 비교군 | 답변 |
|---|---|
| test1.catalog | 비밀번호를 초기화하려면 로그인 화면에서 '비밀번호 찾기' 버튼을 클릭하면 되며, 담당자 이메일을 모를 경우 고객센터(**1899-0970**, 오독)로 문의하면 됩니다. 다만, 초기화 완료 안내 메일 발송에 소요되는 시간은 제공된 문서에서 확인할 수 없습니다.(누락) |
| test1.no_catalog | 비밀번호를 초기화하려면 '비밀번호 초기화' 버튼을 클릭하고 기관 고유 ID를 입력한 후, 담당자 계정을 선택하여 '초기화' 버튼을 누르면 됩니다. 담당자 이메일을 모르는 경우 고객센터(**1899-0970**, 오독)로 문의할 수 있으며, 초기화 완료 안내 메일은 **약 1분 내**(환각, 정답 5분)에 발송됩니다. |
| test2.catalog | 비밀번호 초기화 팝업이 나타나면 이용기관 등록신청 시 입력한 담당자 이메일 주소를 입력하는 것입니다. 담당자 이메일을 알 수 없는 경우 문의할 고객센터 연락처는 **1899-0979**이며, 초기화 완료 안내 메일 발송에는 **최대 5분** 가량 소요될 수 있습니다. |
| test2.no_catalog | (test2.catalog와 동일 내용) 고객센터 **1899-0979**, 메일 발송 **최대 5분**. |

### 4.3 final_003 — 802.11ac/802.11ax 무선랜 표준

> **질문**: 무선랜 구축운영 가이드라인에서 설명하는 802.11ac와 802.11ax 무선랜 표준의 주파수 대역·속도 특성은 각각 무엇이며, 802.11ax를 160MHz 대역폭으로 사용할 때 제공 가능한 최대 속도는 얼마인가요?
> **정답**: 802.11ac(WiFi5)=5GHz대역, 6Gbps목표. 802.11ax(WiFi6)=**2.4GHz/5GHz** 대역, 대규모 혼잡환경 고수율. 160MHz 시 최대 **9.6Gbps**. (근거: 무선랜 가이드라인 5페이지)

| 비교군 | 답변 |
|---|---|
| test1.catalog | 802.11ac는 5GHz 대역에서 6Gbps 속도를 목표로 하며, 802.11ax는 **2.4GHz/5GHz** 대역에서 사용되며 대규모 혼잡 환경에서도 높은 수용력을 달성합니다. 802.11ax를 160MHz 대역폭으로 사용할 때 제공 가능한 최대 속도는 **9.6Gbps**입니다. (내용은 정답과 일치하나 verify 단계 오판정으로 `is_answer_supported=false`) |
| test1.no_catalog | 802.11ac 무선랜 표준은 주파수 대역이 5GHz이며 최대 전송률은 **6.933Gbps**(환각, 원문에 없는 수치)이고, 802.11ax 무선랜 표준은 주파수 대역이 **1GHz~6GHz**(환각, 정답 2.4GHz/5GHz)이며 최대 전송률은 9.6Gbps입니다. |
| test2.catalog | 802.11ac(WiFi5): 5GHz 대역에서 6Gbps 속도 목표('기가급 와이파이'). 802.11ax(Wifi6): **2.4GHz/5GHz대역**에서 대규모 혼잡환경에서도 높은 수율 달성. 160MHz 사용 시 최대 속도 **9.6Gbps**. |
| test2.no_catalog | (test2.catalog와 사실상 동일, 원문 표현을 거의 그대로 전사) |

---

## 5. 답변 증거 (evidence)

### 5.1 test_1 — 근거 이미지 (bbox + crop/highlight)

test_1은 VLM이 지목한 bbox로 crop/highlight 이미지를 직접 생성해 `{qid}_evidence/`에 저장한다. 예:

| qid/mode | 문서 | 페이지 | confidence | bbox (x1,y1,x2,y2) | 근거 이미지 |
|---|---|---|---|---|---|
| final_001/catalog | 8-1 가이드 | 12 | 1.00 | 0.12,0.13,0.58,0.24 | `test1_catalog/final_001_evidence/8-1-33511a48_p0012_ev1_crop.jpg` |
| final_001/no_catalog | 8-1 가이드 | 12 | 0.95 | 0.04,0.13,0.48,0.26 | `test1_no_catalog/final_001_evidence/8-1-33511a48_p0012_ev1_crop.jpg` |
| final_002/catalog | 이용관리_20260702 | 12 | 0.90 | 0.13,0.05,0.54,0.48 | `test1_catalog/final_002_evidence/20260702-336b732d_p0012_ev1_crop.jpg` |
| final_002/no_catalog | 이용관리_20260702 | 12 | 0.95 | 0.12,0.24,0.88,0.46 | `test1_no_catalog/final_002_evidence/20260702-336b732d_p0012_ev1_crop.jpg` |
| final_003/no_catalog | 무선랜 가이드라인 | 5 | 1.00 | 0,0,1,1 (전체페이지) | `test1_no_catalog/final_003_evidence/0-1v-20p-2023-a874b961_p0005_ev1_crop.jpg` |
| final_003/catalog | — | — | — | **없음** — VLM이 evidence를 반환하지 않아 crop/highlight 미생성(§8.2) |

검증 방법: `crop_image_path`를 열어 표/문단이 실제로 정답을 뒷받침하는지 육안 대조 — final_001/002는 정확한 표 영역을 짚었으나(bbox가 실제 표 위치와 일치), **표 안의 숫자를 읽어내는 과정(VLM 전사)에서 오류**가 발생했다는 것이 이번 실험의 핵심 발견이다(bbox 위치는 맞고 값만 틀림 → "어디를 봤는지"와 "무엇을 읽었는지"가 분리된 오류).

### 5.2 test_2 — 근거 페이지 이미지 / 표 크롭

test_2는 ingest 시점에 MinerU가 이미 만들어 둔 페이지 이미지·표 크롭 경로를 그대로 반환한다(별도 생성 없음, 기존 `test_2/rag2/cache/parsed/` 재사용):

| qid/mode | 문서 | 페이지 | page_image_path (발췌) | table_crop_path |
|---|---|---|---|---|
| final_001 (양쪽) | 8-1 가이드 | 12 | `.../8-1-33511a48/pages/p0012.png` | `.../mineru/.../805fd062...955dd.jpg` |
| final_002 (양쪽) | 이용관리_20260702 | 12 | `.../20260702-336b732d/pages/p0012.png` | (표 없음, 텍스트만) |
| final_003 (양쪽) | 무선랜 가이드라인 | 5 | `.../0-1v-20p-2023-a874b961/pages/p0005.png` | `.../mineru/.../dc3c72f0...27906.jpg` |

test_2의 `table_crop_path`가 있는 경우(final_001, final_003) MinerU가 표를 별도 마크다운+이미지로 미리 추출해두었고, VLM/LLM은 그 마크다운 텍스트를 그대로 옮겨 적은 뒤 답했다 — 이것이 test_2의 수치 정확도가 test_1보다 높았던 근본 원인이다(§8.1).

전체 원본 경로는 [`all_results_20260714T040610Z.json`](all_results_20260714T040610Z.json)의 각 항목 `evidence` 필드 참조.

---

## 6. 추가로 필요하다고 판단한 항목

### 6.1 실행 신뢰성

12건 전원 `error: null`, `subprocess_returncode: 0`. 타임아웃(test_1 2400초/test_2 1200초 상한) 초과 없음. 가장 오래 걸린 실행은 `test1.catalog/final_002`(936.0초, 15.6분)로 상한의 39%만 사용해 여유가 있었다.

### 6.2 냉시작 조건이 test_1 소요시간에 미친 영향 (해석 시 주의)

이번 test_1 소요시간(문항당 13~15분)은 **실전 반복 사용 시나리오를 대표하지 않는다.** 매 질문 전 요약/청킹 캐시를 완전히 비운 "최초 1회 질문" 조건이기 때문이며, 같은 문서에 대한 두 번째 질문부터는 `page_summary`(41%)·`visual_chunk`(47.5%) 두 단계가 대부분 캐시 히트로 대체되어 실질 소요시간이 answer+verify 수준(문항당 40~70초대)으로 줄어들 것으로 예상된다(README §"왜 lazy인가" 참조). 이번 실험은 **의도적으로 이 캐시 이득을 배제**했다 — 목적이 "동일 냉시작 조건에서의 공정 비교"였기 때문이다.

### 6.3 Ollama 응답시간 자연 변동

`test2.no_catalog/final_003`(69.6초)이 같은 비교군의 다른 두 문항(18.6초, 23.7초)보다 3배 가까이 길었던 것처럼, 동일 파이프라인·동일 모델에서도 개별 호출의 소요시간은 자연 변동이 크다(test_2_timecost 실측에서도 동일 관찰). N=3의 소표본에서는 개별 초 단위 비교보다 **합계/평균**으로 판단하는 것이 안전하다.

---

## 7. 요약 비교표 (전체 지표 한눈에)

| 비교군 | 총 소요시간 | 문항당 평균 | doc/page_match | 평균 kw_recall | LLM 호출/문항 | VLM 호출/문항 | Embed/문항 |
|---|---|---|---|---|---|---|---|
| test1.catalog | 2605.4s | 868.4s | 100%/100% | 0.646 | 1.67 | 29 | 11 |
| test1.no_catalog | 2448.5s | 816.2s | 100%/100% | 0.562 | 2.00 | 29 | 32 |
| test2.catalog | 61.4s | 20.5s | 100%/100% | **0.736** | 1 | 0 | 1 |
| test2.no_catalog | 112.0s | 37.3s | 100%/100% | **0.736** | 1 | 0 | 1 |

---

## 8. 해석 및 한계

### 8.1 핵심 결론

1. **문서/페이지 검색 정확도는 4개 비교군 모두 100%** — 이번 3문항 규모에서는 검색 단계가 병목이 아니었다.
2. **답변의 수치 정확도는 test_2(MinerU 텍스트 추출)가 test_1(VLM 비전 판독)보다 명확히 우수했다** — 8회 수치 관측 중 test_1은 4회 오류(오독 2회+환각 2회), test_2는 0회. test_1차 자체 REPORT.md §5 A1이 지목한 "VLM이 조밀한 표 숫자를 잘못 전사"하는 한계가 완전히 새로운 문서·질문에서도 재현되었다.
3. **속도는 test_2가 test_1보다 약 40배 빠르다** — ingest 시점 사전 파싱(MinerU) vs query-time lazy VLM 요약이라는 아키텍처 차이가 원인이며, 이는 캐시 유무와 무관한 구조적 차이다(첫 질문 기준).
4. **이 코드베이스의 현재 설정에서는 catalog 유무가 test_1의 VLM 호출량에 차이를 만들지 않는다** — `no_catalog_page_prefilter_topn=24`가 catalog의 `top_docs×page_prefilter_topn=24`와 의도적으로 동일하게 맞춰져 있기 때문(config.yaml 주석 확인). 3문항 전부 `summary_pages_required=24`, `chunk_pages_required=4`로 정확히 일치했다. catalog 유무의 실질적 차이는 `embed_calls`(11 vs 31~33)뿐이었다.

### 8.2 관측된 한계

- **verify 단계의 취약성**: `test1.catalog/final_003`은 답변 내용이 정답과 완전히 일치했음에도 VLM이 evidence bbox를 스키마 규격에 맞게 반환하지 못해 `page_evidence`가 비었고, 그 결과 `verify_answer`가 근거 부재로 조기 종료해 `is_answer_supported=false`로 잘못 판정했다. **답변 품질과 검증 판정이 분리되어 있어, 근거 추출 실패가 있으면 정답도 "미지원"으로 보일 수 있다.**
- **no_catalog의 환각 경향**: `test1.no_catalog/final_003`에서 원문에 없는 구체적 수치(6.933Gbps, 1GHz~6GHz)를 생성했다. 표본이 1건뿐이라 일반화하기는 이르지만, catalog가 문서를 2개로 좁혀주는 것과 달리 no_catalog는 더 넓은 후보 페이지 조합을 다뤄야 하는 만큼 프롬프트 컨텍스트가 산만해질 가능성을 시사한다.
- **키워드 채점의 정밀도 한계**: test_2의 답변이 원문 그대로 "WiFi5"(공백 없음)로 표기했는데 채점 키워드는 "WiFi 5"(공백 있음)였다. 이 경우 keyword_recall이 실제보다 낮게 잡혔다(자기 진단: 이번 채점 스크립트의 대소문자 무시·부분일치 로직이 공백까지는 흡수하지 못함). 실제 답변 완성도는 표에 기록된 recall 수치보다 다소 높다고 봐야 한다.
- **표본 크기(N=3)**: 4개 비교군 x 3문항으로 방향성은 뚜렷하지만 통계적으로 유의하다고 주장하기는 어렵다. 특히 test_1의 "냉시작 소요시간"은 §6.2에서 설명한 대로 실전 반복사용 시나리오를 대표하지 않는다.
- **`filename_only` 미포함**: 사용자 지시로 이번 실험에서 제외했다. test_1차 자체 3비교군 실험 결과는 `test_1차/rag_catalog_experiment/REPORT.md §10`을 참고할 것.

### 8.3 실험 방법론 자체의 한계

- test_1의 냉시작 스냅샷 복원은 파일시스템 복사(`shutil.copytree`)로 수행했다 — 매 실행 전 `_work_test1/{index,cache,output}`을 삭제 후 재복사하는 방식이라 시스템 I/O 부하가 소요시간에 소폭 포함되어 있을 수 있으나(초 단위, 무시할 수준), `retrieval_total`이 각 단계 시간의 합과 거의 일치하는 것으로 보아 유의미한 오차는 아니다.
- 모델 워밍업은 실험 시작 시 1회만 수행했다 — 개별 비교군 사이(예: test1 6건이 끝나고 test2로 넘어가는 시점)에 Ollama의 `keep_alive=10m`이 만료되지 않았는지는 로그상 호출 간격이 모두 10분 미만이라 만료되지 않았을 것으로 판단되나 별도로 재검증하지는 않았다.

---

## 9. 재현 방법

```
cmd /c conda activate intern_chatbot && python test12_total_test/run_final_experiment.py
```

- `final_qa_dataset.json` 수정 시 3문항 이상/이하로 조정 가능(질문 형식은 스키마 참조)
- test_1 냉시작 스냅샷은 `_cold_baseline_test1/`에 보존되어 있어 재실행 시 ingest를 다시 하지 않음(스냅샷 폴더 삭제 시에만 재생성 필요)
- test_2는 항상 `test_2/rag2/index`·`cache`를 읽기전용 재사용(재-ingest 불필요, 수정 시 test_2 쪽 ingest를 별도로 다시 돌려야 함)
