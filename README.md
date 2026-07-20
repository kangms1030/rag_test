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
├── 최종_결과_보고서.md           3세대 종합 보고서
├── .gitignore
├── .env                        GEMINI_API_KEY (git 제외)
└── CLAUDE.md                   프로젝트 실행 규칙
```

## 버킷 규칙

- **코드/**: 실행 RAG 패키지 통째(내부 빌드 산출 `index/`·`cache/` 포함, 실행상태 보존).
- **사전데이터/**: 원천 입력(PDF·카탈로그 Excel·평가 질문셋·원본 참고 노트북).
- **결과물/**: 보고서(.md/.docx)·평가결과 JSON·근거이미지·외부 비교실험 결과.

## 읽는 순서

1. [최종_결과_보고서.md](최종_결과_보고서.md) — 3세대 전체 조망.
2. 각 세대 `final.md`(파이프라인 단계별 모델·프로그램) → `CHANGES_from_*.md`(세대 전환 이유).
3. 세부 지표: `test_3/결과물/probes_results/` · `test_3/결과물/rag3_참고_실험팩트정리.md`.

> ⚠️ **경로 주의**: 각 RAG 패키지의 `config.yaml`은 사전데이터를 `../데이터…` 상대경로로 참조하므로,
> 사전데이터를 `사전데이터/`로 분리한 현 구조에서 **재-ingest 경로는 조정이 필요**하다. 쿼리·데모는
> test_3의 경우 내부 캐시로 자립 동작한다. (로컬 모델 특성상 실행 검증은 지양, 비교대조로 검증함.)
