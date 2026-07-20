# rag3 설정 파라미터 가이드 (하이퍼파라미터 사전)

> 딥러닝의 learning rate처럼 **코드는 건드리지 않고 값만 바꿔** 동작을 조정하는 표면 정리.
> 그 표면이 바로 [`rag3/config.yaml`](rag3/config.yaml)이다. 여기 값을 바꾸는 것은 동결된 rag3 로직 수정이 아니라 **설정(데이터)** 변경이므로 "완성 코드 무수정" 원칙에 걸리지 않는다.

## 파라미터의 두 종류 (중요)

| 표기 | 의미 | 반영 방법 | DL 비유 |
|---|---|---|---|
| 🟢 **[즉시]** | ask(질의) 시점에만 작동 | yaml 저장 → 다시 질문하면 바로 반영 | inference에서 lr/threshold 조정 |
| 🔴 **[재색인]** | ingest(색인 구축) 시점에 작동 | 바꾸면 **인덱스를 다시 만들어야** 반영 | 모델 구조 변경 후 재학습 |

---

## 1. 청킹 — 문서를 어떤 크기로 자르나  🔴 [재색인]

| 파라미터 | 현재값 | 무엇을 건드나 | ↑ 올리면 / ↓ 내리면 |
|---|---|---|---|
| `chunk_target_chars` | 700 | 텍스트 청크 목표 길이 | ↑ 문맥 풍부·검색 정밀도↓(뭉뚱그림) / ↓ 정밀하나 문맥 파편화 |
| `chunk_max_chars` | 1200 | 강제 분할 상한 | 청크 최대 크기 |
| `chunk_table_split_chars` | 2500 | 큰 표를 헤더 반복하며 쪼개는 기준 | 표 청크 크기 |
| `chunk_min_chars` | 40 | 이보다 작은 조각은 인접 청크에 병합(노이즈 제거) | ↑ 노이즈↓·짧은 정보 손실 위험 |

> DL 비유: **입력 patch/window 크기**. 바꾸면 색인을 다시 구축해야 하는 "재학습"급 작업.

## 2. 하이브리드 검색 후보 풀  🟢 [즉시]

| 파라미터 | 현재값 | 무엇을 건드나 | 효과 |
|---|---|---|---|
| `retrieve_candidates` | 20 | 리랭커에 넣을 후보 청크 수(top-K) | DL의 **beam/top-k**. 실측상 20>40 (후보 늘리면 방해후보만 늘어 손해) |
| `rrf_k` | 60 | BM25+dense 순위 융합(RRF) 상수 | 두 검색 결합 방식. 클수록 순위 차이 완만 반영 |

## 3. 리랭커 — 관련성 재점수 + 거절 문턱  🟢 [즉시]

| 파라미터 | 현재값 | 무엇을 건드나 | ↑ 올리면 / ↓ 내리면 |
|---|---|---|---|
| **`rerank_score_floor`** | 0.1 | **무관질문 거절 문턱**(미만이면 "확인 불가") | ↑ 엄격(거절↑, 엉뚱한 답↓) / ↓ 관대(답변↑, 무관질문 헛답 위험). 캘리브레이션: 관련 min 0.688 vs 무관 max 0.013 → 0.1이 깔끔히 분리 |
| `rerank_max_length` | 2048 | 리랭커가 읽는 최대 토큰 | ↑ 긴 청크 끝까지 봄·느려짐 |

> `rerank_score_floor`가 코드에서 **가장 lr에 가까운 "결정 문턱"**. 분류기의 판정 컷오프와 동일 역할.

## 4. small-to-big 집계 — 몇 페이지를 근거로 올리나  🟢 [즉시]

| 파라미터 | 현재값 | 무엇을 건드나 | 효과 |
|---|---|---|---|
| `final_pages` | 3 | 답변 컨텍스트로 승격할 **최대 페이지 수** | ↑ 커버리지↑·노이즈↑ / ↓ 집중되나 다중페이지 답 누락. **mdm 다중페이지 문제와 직결** |
| `top_pages` | 3 | 평가/근거 표시용 상위 페이지 수 | 위와 유사 |
| `page_score_agg` | "max" | 청크 점수→페이지 점수 집계(`max`\|`sum_topk`) | Phase 1 실측상 max로 충분 |
| `page_score_topk` / `page_score_decay` | 3 / 0.5 | sum_topk일 때만 사용 | 보조 청크 가중 |

## 5. 라우팅 — text냐 vision이냐  🟢 [즉시]

| 파라미터 | 현재값 | 무엇을 건드나 | 효과 |
|---|---|---|---|
| `figure_area_ratio_threshold` | 0.5 | vision 경로 발동 면적 문턱 | **page_type="figure"인 페이지에만** 적용 |
| `scanned_table_verify` | true | 스캔+표 교차확인 여부 | vision 교차검증 스위치 |

> ⚠️ 이 문턱 **위에 코드 하드코딩 상위 게이트**가 있다 — [parse_mineru.py:26-27](rag3/parse_mineru.py#L26)의 `0.3`(figure 후보 면적)과 `200`(텍스트 상한 char). yaml에 없어 바꾸려면 **코드 수정 + 재파싱** 필요. (p13이 여기서 걸려 text로 라우팅됨: 표 보유 → page_type="table", 텍스트 2118자 → figure 후보 탈락.)

## 6. 답변 LLM 옵션  🟢 [즉시]

| 파라미터 | 현재값 | 무엇을 건드나 | ↑/↓ |
|---|---|---|---|
| `ollama_num_ctx` | 16384 | 컨텍스트 창(입력+KV 캐시) | ↑ 긴 문맥 수용·VRAM/지연↑ |
| `ollama_num_predict` | 1536 | **출력 토큰 상한** | ↑ 긴 답 허용·폭주 시 낭비↑ |
| `context_max_chars` | 10000 | 답변에 넣는 페이지 텍스트 예산(트림) | ↑ 근거 많이·num_ctx 압박 |
| `ollama_keep_alive` | "30m" | 모델을 VRAM에 유지하는 시간 | ↑ 웜 유지(빠름)·VRAM 점유 |
| `ollama_seed` | 0 | 재현성 시드 | temp=0에선 사실상 무효 |

> 💡 **temperature는 config에 없다.** [models.py:67](rag3/models.py#L67)에 `temperature: 0.0` **하드코딩**. 진짜 "창의성 노브"지만 숫자 정확도(그리디 디코딩)를 위해 0으로 잠갔다. 올리면 답 다양성↑·숫자 오류 위험↑라 일부러 config로 안 뺐다.

## 7. 검증·롤백·CRAG — 재시도 정책  🟢 [즉시]

| 파라미터 | 현재값 | 무엇을 건드나 | 효과 |
|---|---|---|---|
| `enable_verify` / `enable_rollback` / `enable_crag` | true | 각 단계 on/off | 끄면 모델 호출↓·정확도 방어↓ |
| **`rollback_rerank_tau_high`** | 0.5 | 이 점수 이상이면 빈답/거절 시 **롤백 재시도** | 학내망 사례에서 gemma 추가 호출 유발한 값. ↑ 하면 재시도 덜 함(비용↓·복구↓) |
| `crag_retry_floor` | 0.02 | 경계 점수에서 질의 재작성 재시도 범위 | 무관거절과 재시도의 경계 |
| `deadline_seconds` | 180 | 문항 wall-clock 상한 | 초과 시 best-effort 반환 |
| `ollama_retry_on_length` / `ollama_max_length_retries` | true / 2 | 콜드 whitespace 런어웨이 방어 재발행 예산 | ↑ 복구↑·지연↑ |
| `verify_model` | "" (=12b 겸용) | groundedness 판정 모델 | e4b 등으로 교체 가능 |

## 8. 현재 비활성 (참고)  🔴

`use_catalog_gate: false`라 아래는 **지금 작동 안 함**(249문서 확장 시 부활): `min_doc_score`, `doc_score_gap_ratio`, `min_dense_similarity`, `top_docs`.
`tokenizer: "kiwi"`도 바꾸면 재색인 필요.

---

## 코드에 숨은(config에 없는) 하이퍼파라미터

yaml로 못 바꾸고 **코드 수정이 필요한** 값들. 참고용.

| 위치 | 값 | 의미 | 바꾸면 |
|---|---|---|---|
| [models.py:67](rag3/models.py#L67) | `temperature=0.0` | 디코딩 온도 | 숫자 정확도 트레이드오프 |
| [parse_mineru.py:26](rag3/parse_mineru.py#L26) | `_FIGURE_TYPE_AREA_RATIO=0.3` | figure 후보 판정 면적 | 재파싱 |
| [parse_mineru.py:27](rag3/parse_mineru.py#L27) | `_FIGURE_TYPE_MAX_CHARS=200` | figure 후보 텍스트 상한 | 재파싱 |
| [vlm_reparse.py:50](rag3/vlm_reparse.py#L50) | `txt_max=200` | "그림 위주 페이지" 텍스트 문턱 | 재색인(그림 텍스트화 대상 변동) |

---

## "lr처럼 바로 만질 수 있는" 핵심 노브 4개

앞서 분석한 문제들과 직접 연결되는 **즉시 반영** 노브:

| 노브 | 현재값 | 연결된 문제 | 방향 |
|---|---|---|---|
| `rerank_score_floor` | 0.1 | 거절을 얼마나 엄격하게 | ↓ 거절 완화(답변↑, 헛답 위험↑) — lr에 가장 가까운 노브 |
| `rollback_rerank_tau_high` | 0.5 | 학내망 사례 gemma 추가호출 | ↑ 재시도 억제(비용↓) |
| `final_pages` | 3 | mdm류 다중페이지 커버리지 | ↑ 커버리지↑(노이즈↑) |
| `ollama_num_predict` | 1536 | 답 길이 상한 | ↑ 긴 답 허용 |

> 이 넷은 yaml 값만 고치고 다시 질문하면 끝이라, **코드 무수정 원칙을 지키며 A/B** 해볼 수 있는 안전한 표면.

## 실측(A/B) 방법

```
# 1) config.yaml에서 노브 1개만 변경 (한 번에 하나씩 — Phase 규칙)
# 2) CLI로 예전 실패 질문 재실행
cmd /c conda activate intern_chatbot && cd test_3 && python ask_cli.py -q "질문"
# 3) 파이프라인 추적(모델 호출수/경로/타이밍)으로 before/after 비교
```
