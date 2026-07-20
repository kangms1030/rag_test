# MinerU 태깅 확인 보고서

작성일: 2026-07-20

## 확인 대상

- 코드: `test_3/코드/rag3`
- MinerU 파싱 캐시: `test_3/코드/rag3/cache/parsed_v25`
- 확인 파일:
  - `*/mineru/*/auto/*_content_list.json`
  - `*/mineru/*/auto/*_content_list_v2.json`
  - `*/manifest.json`

## 결론

현 `test_3` 코드는 MinerU를 단순 PDF 텍스트 파싱 용도로만 쓰지 않는다.

1. MinerU 기본 파싱 결과에는 페이지 내 객체가 `text`, `table`, `image`, `chart`, `header`, `footer`, `page_number`, `code`, `equation` 등으로 태깅되어 있다.
2. 모든 주요 블록에는 `page_idx`와 `bbox`가 붙어 있어 페이지 내 위치 태깅도 되어 있다.
3. `content_list_v2.json`에는 더 의미론적인 태그인 `title`, `paragraph`, `table`, `image`, `list`, `index`, `page_header`, `page_footer`, `chart`, `code`, `algorithm`, `equation_interline`이 보존되어 있다.
4. figure 위주 페이지 115개에는 MinerU `vlm-engine` 재파싱 결과가 `_vlm_reparse: true`인 `type: "text"` 블록으로 병합되어 있다.
5. 병합된 페이지는 `manifest.json`에서 `page_type`이 `text`로 바뀌어, 질의 시 별도 VLM 호출 없이 텍스트 검색/답변 경로로 들어가도록 설계되어 있다.

따라서 회의에서 들은 “MinerU가 파싱뿐 아니라 이미지 태깅 및 텍스트화까지 해서 이후 VLM을 덜 쓰거나 쓰지 않게 한다”는 설명은 코드와 캐시 기준으로 대체로 맞다. 다만 정확히 말하면, 모든 이미지가 완전하게 의미 태깅된 것은 아니고, figure-dominant 페이지를 선별해 MinerU `vlm-engine` 결과를 텍스트 블록으로 승격/병합하는 방식이다.

## 집계 결과

### 기본 `content_list.json` 태그 수

총 13개 문서 캐시에서 확인한 블록 수:

| 태그 | 개수 |
|---|---:|
| text | 8,405 |
| image | 1,205 |
| header | 705 |
| footer | 643 |
| table | 553 |
| page_number | 510 |
| chart | 57 |
| code | 33 |
| equation | 1 |

`bbox`가 붙은 블록은 11,997개로, 대부분의 레이아웃 객체가 위치 정보까지 가진다.

### `content_list_v2.json` 의미 태그 수

| 태그 | 개수 |
|---|---:|
| paragraph | 6,033 |
| title | 2,185 |
| image | 1,205 |
| page_header | 705 |
| page_footer | 643 |
| table | 553 |
| page_number | 510 |
| chart | 57 |
| index | 40 |
| list | 32 |
| code | 17 |
| algorithm | 16 |
| equation_interline | 1 |

v2에는 `title_content`, `paragraph_content`, `html`, `image_source`, `table_type`, `code_language`, `math_content` 같은 구조화 필드도 들어 있다.

### VLM 재파싱 병합 현황

`_vlm_reparse: true` 블록이 병합된 페이지는 총 115페이지다.

| 문서 slug | 전체 페이지 | VLM 병합 페이지 |
|---|---:|---:|
| 0-1v-20p-2023-a874b961 | 23 | 8 |
| 20260702-336b732d | 133 | 19 |
| 20260702-62c3c925 | 28 | 3 |
| 20260702-6658f5d9 | 165 | 20 |
| 20260702-80cc24f0 | 101 | 10 |
| 23-b3bffb22 | 101 | 36 |
| 5-c9ff266c | 40 | 0 |
| 8-1-33511a48 | 29 | 2 |
| doc-8bc422d9 | 13 | 6 |
| mdm-argos-edu-v1-5-6061726f | 94 | 3 |
| weiss-de-001-3-v0-82-f69ab0fd | 78 | 3 |
| weiss-de-001-4-v0-82-b9c3bac8 | 64 | 5 |
| weiss-ds-001-v0-91-aa83eb15 | 100 | 0 |

`manifest.json` 기준 페이지 타입은 `text` 633페이지, `table` 326페이지, `figure` 10페이지다. 즉 재파싱 병합 후에도 아직 순수 figure로 남은 페이지는 10페이지뿐이다.

## 실제 태깅 예시

### 표 태그

`type: "table"` 블록에는 다음 정보가 들어 있다.

- `page_idx`
- `bbox`
- `img_path`
- `table_caption`
- `table_body`

예시로 무선 AP 설명 표는 HTML table 형태로 보존되어 있으며, 표 이미지 crop 경로도 같이 가진다. 그래서 후속 검색 청크는 표 텍스트를 쓰고, 필요하면 근거 표시에는 crop 이미지를 쓸 수 있다.

### 이미지 태그

`type: "image"` 블록에는 다음 정보가 들어 있다.

- `page_idx`
- `bbox`
- `img_path`
- `image_caption`
- `image_footnote`

기본 파싱의 image caption은 빈 배열인 경우가 많다. 이 때문에 코드에서는 일반 image 블록 자체를 청킹에서 제외하고, figure-dominant 페이지에 대해서만 MinerU `vlm-engine` 결과를 따로 병합한다.

### VLM 재파싱 텍스트 예시

`_vlm_reparse` 블록은 원래 이미지/도표 중심이던 페이지에 다음처럼 텍스트를 추가한다.

- 망분리 구성도: 방화벽, 학생망 집선스위치, 교사망 집선스위치, 연결 계위 구조 등 도표 라벨을 텍스트화
- 통합 플랫폼 화면: 로그인 화면, 계정신청, 사용자 수정, 장비현황, AP/PoE/유선/접속단말기 등 화면 요소를 텍스트화
- 표지/그림 페이지: 문서 제목과 이미지 설명을 텍스트화

즉 이미지 자체를 답변 시점에 다시 VLM으로 읽는 대신, 사전 단계에서 MinerU VLM 결과를 텍스트로 박아두는 방식이다.

## 코드 흐름 확인

- `vlm_reparse.py`는 image/figure가 있고 기존 텍스트가 적은 페이지를 찾는다.
- 해당 페이지의 MinerU `vlm-engine` 결과에서 text/table/image content를 모아 페이지별 텍스트를 만든다.
- 그 텍스트를 원래 `content_list.json`에 `type: "text"`, `_vlm_reparse: true` 블록으로 추가한다.
- 동시에 `manifest.json`의 해당 페이지 `text`, `char_count`, `page_type`을 갱신한다.
- `chunking.py`는 `image` 블록을 직접 청킹하지 않지만, 병합된 VLM 텍스트 블록은 일반 `text`로 청킹한다.
- `retrieve.py`는 기본적으로 text-first 라우팅이며, `page_type == "figure"`이고 그림 비율이 큰 경우에만 vision 경로를 선택한다.

## 주의점

1. `content_list_v2.json`의 의미 태그가 풍부하지만, 현재 검색 청킹 코드는 주로 기존 `content_list.json`를 사용한다.
2. 기본 `image` 태그의 `image_caption`은 빈 경우가 많아, image 태그만으로는 충분한 설명이 되지 않는다.
3. 일부 VLM 텍스트에는 영어식 일반 이미지 설명이나 반복/오독 흔적이 섞여 있다. 예: "Illustration of..." 또는 UI 필드 반복.
4. 그래도 도표/구성도 페이지에서는 방화벽, PoE, 스위치, 망분리, 절차명 같은 검색 앵커가 추가되어 이후 VLM 의존도를 크게 낮춘다.

## 과제 답변용 요약

`test_3`의 MinerU 산출물을 확인한 결과, MinerU는 PDF를 텍스트로 파싱하는 것뿐 아니라 페이지 내부 요소를 표, 이미지, 차트, 제목, 본문, 헤더/푸터 등으로 태깅하고 bbox 위치까지 저장하고 있었다. 또한 figure 중심 페이지 115개에는 MinerU `vlm-engine` 결과가 `_vlm_reparse` 텍스트 블록으로 병합되어 있어, 후속 질의 단계에서 별도 VLM 호출 없이 텍스트 검색과 답변에 활용되도록 구성되어 있다. 다만 모든 이미지가 완전한 의미 설명을 갖는 것은 아니며, 현재 핵심 전략은 figure 페이지를 선별해 VLM 텍스트를 사전 병합하는 방식이다.
