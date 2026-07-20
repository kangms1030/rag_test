# test_2 카탈로그 유무 비교 실험 (catalog vs no_catalog)

test_1차(`rag_catalog_experiment/REPORT.md` §10)에서 진행한 "카탈로그 유무가 RAG 정확도·
시간비용에 미치는 영향" 실험을 test_2(`rag2`) 파이프라인에 대해 재현한 실험 폴더다.

- `test_2/rag2`는 **수정하지 않는다** — 여기서는 import만 한다.
- 기존에 ingest된 인덱스(`test_2/rag2/index`)·캐시(`test_2/rag2/cache`)·카탈로그를 그대로
  재사용한다(재-ingest 불필요).
- 산출물은 `results/`에만 쓴다. `test_2/` 아래에는 어떤 파일도 새로 생기지 않는다.

## 구성

- `retrieve_no_catalog.py` — catalog_index 게이트를 건너뛰고 `page_index`를 문서 제한 없이
  전역 검색해 상위 페이지에서 문서를 역산하는 baseline 리트리버.
- `run_experiment.py` — 평가셋(`test_2/rag_eval_dataset.json`, 16문항)을 catalog/no_catalog
  두 모드로 나란히 실행하고 `results/compare_{ts}.json` + `results/REPORT.md`를 생성.

## 실행

```
cmd /c conda activate intern_chatbot && python "test_2/rag2/__main__.py" check --skip-indexes
cmd /c conda activate intern_chatbot && python run_experiment.py
```

(작업 디렉터리는 이 폴더 `test_2_timecost/` 기준)
