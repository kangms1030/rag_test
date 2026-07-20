# 답변 결과 모음 — 4개 비교군 × 3문항

각 질문에 대해 정답(근거 포함)과 4개 비교군(`test1.catalog`, `test1.no_catalog`, `test2.catalog`, `test2.no_catalog`)이 실제로 생성한 답변 원문을 그대로 모았다. 분석·해석은 [`FINAL_REPORT.md`](FINAL_REPORT.md)를 참조할 것 — 이 문서는 답변 원문 그 자체만 다룬다.

원본 데이터: [`all_results_20260714T040610Z.json`](all_results_20260714T040610Z.json)

---

## final_001 — UTP 케이블 규격표

**질문**: 학내망 구축 및 운영관리 개선을 위한 가이드에서 제시하는 UTP 케이블 규격(Cat.5, Cat.5e, Cat.6)별 최대 전송거리와 통신속도(Data Performance / Max Speed Data)는 각각 어떻게 다른가요?

**정답 (근거: 8-1. 스쿨넷서비스 학내망 구축 및 운영관리 개선을 위한 가이드.pdf, 12페이지)**
> UTP 케이블 규격별 통신 속도 표에 따르면 세 규격 모두 최대 전송거리는 100M로 동일합니다. Cat.5는 Data Performance 10Mbps, Max Speed Data 100Mbps이고, Cat.5e는 Data Performance 100Mbps, Max Speed Data 1Gbps이며, Cat.6은 Data Performance 1Gbps, Max Speed Data 10Gbps입니다.

### test1.catalog
- 선정 문서: 8-1. 스쿨넷서비스 학내망 구축 및 운영관리 개선을 위한 가이드.pdf / ★23년 학교 유무선 운영·관리 안내서_최종.pdf
- doc_match: True / page_match: True / keyword_recall: 0.875 / 소요시간: 779.3초

> UTP 케이블 규격별 최대 전송거리와 통신속도(Data Performance / Max Speed Data)는 다음과 같습니다.
> - Cat.5: 최대 전송거리 100m, Data Performance 100Mbps, Max Speed Data 100Mbps
> - Cat.5e: 최대 전송거리 100m, Data Performance 100Mbps, Max Speed Data 1Gbps
> - Cat.6: 최대 전송거리 100m, Data Performance 1Gbps, Max Speed Data 10Gbps

### test1.no_catalog
- 선정 문서: 8-1. 스쿨넷서비스 학내망 구축 및 운영관리 개선을 위한 가이드.pdf / ★23년 학교 유무선 운영·관리 안내서_최종.pdf
- doc_match: True / page_match: True / keyword_recall: 0.875 / 소요시간: 890.2초

> UTP 케이블 규격별 최대 전송거리와 통신속도(Data Performance, Max Speed Data)는 다음과 같습니다.
> - Cat.5: 100M / 100Mbps / 100Mbps
> - Cat.5e: 100M / 100Mbps / 100Mbps
> - Cat.6: 100M / 1Gbps / 10Gbps

### test2.catalog
- 선정 문서: 8-1. 스쿨넷서비스 학내망 구축 및 운영관리 개선을 위한 가이드.pdf / ★. 무선랜 구축운영 가이드라인_0.1v_20p수정_2023년.pdf
- doc_match: True / page_match: True / keyword_recall: 1.0 / 소요시간: 19.1초

> 학내망 구축 및 운영관리 개선을 위한 가이드에서 제시하는 UTP 케이블 규격별 최대 전송거리와 통신 속도는 다음과 같습니다.
>
> * **Cat.5**: 최대 전송거리 100M, Data Performance 10Mbps, Max Speed Data 100Mbps
> * **Cat.5e**: 최대 전송거리 100M, Data Performance 100Mbps, Max Speed Data 1Gbps
> * **Cat.6**: 최대 전송거리 100M, Data Performance 1Gbps, Max Speed Data 10Gbps

### test2.no_catalog
- 선정 문서: 8-1. 스쿨넷서비스 학내망 구축 및 운영관리 개선을 위한 가이드.pdf
- doc_match: True / page_match: True / keyword_recall: 1.0 / 소요시간: 18.6초

> 학내망 구축 및 운영관리 개선을 위한 가이드에서 제시하는 UTP 케이블 규격별 최대 전송거리와 통신 속도(Data Performance / Max Speed Data)는 다음과 같습니다.
>
> * **Cat.5**: 최대 전송거리 100M, Data Performance 10Mbps, Max Speed Data 100Mbps
> * **Cat.5e**: 최대 전송거리 100M, Data Performance 100Mbps, Max Speed Data 1Gbps
> * **Cat.6**: 최대 전송거리 100M, Data Performance 1Gbps, Max Speed Data 10Gbps

---

## final_002 — 이용관리시스템 비밀번호 초기화

**질문**: 스쿨넷 이용관리시스템에서 이용기관 비밀번호를 초기화하는 절차는 무엇이며, 담당자 이메일을 모를 경우 문의할 고객센터 연락처와 초기화 완료 안내 메일 발송에 소요되는 시간은 얼마인가요?

**정답 (근거: 이용관리_20260702.pdf, 11-12페이지)**
> 이용관리시스템 로그인 화면에서 초기화할 기관명을 선택하고 이용기관등록신청 버튼 하단의 비밀번호초기화를 선택합니다. 비밀번호초기화 팝업이 나타나면 이용기관등록신청 시 입력한 담당자 이메일 주소를 입력해야 하며(필수 입력, 등록 시와 동일한 주소여야 진행됨), 담당자 이메일을 알 수 없는 경우 이용관리시스템 고객센터(1899-0979)로 문의해야 합니다. 처리가 완료되면 school_report@nia.or.kr 주소로 비밀번호 재설정 안내 메일을 발송하며, 메일 발송에는 최대 5분가량 소요될 수 있습니다.

### test1.catalog
- 선정 문서: 이용관리_20260702.pdf / ★23년 학교 유무선 운영·관리 안내서_최종.pdf
- doc_match: True / page_match: True / keyword_recall: 0.286 / 소요시간: 936.0초

> 비밀번호를 초기화하려면 로그인 화면에서 '비밀번호 찾기' 버튼을 클릭하면 되며, 담당자 이메일을 모를 경우 고객센터(1899-0970)로 문의하면 됩니다. 다만, 초기화 완료 안내 메일 발송에 소요되는 시간은 제공된 문서에서 확인할 수 없습니다.

### test1.no_catalog
- 선정 문서: 이용관리_20260702.pdf
- doc_match: True / page_match: True / keyword_recall: 0.143 / 소요시간: 817.1초

> 비밀번호를 초기화하려면 '비밀번호 초기화' 버튼을 클릭하고 기관 고유 ID(예: school_name.kr.or.kr)를 입력한 후, 담당자 계정을 선택하여 '초기화' 버튼을 누르면 됩니다. 담당자 이메일을 모르는 경우 고객센터(1899-0970)로 문의할 수 있으며, 초기화 완료 안내 메일은 약 1분 내에 발송됩니다.

### test2.catalog
- 선정 문서: 이용관리_20260702.pdf / WEISS-DE-001-(컨설팅)완료보고서-4.이용관리시스템-v0.82.pdf
- doc_match: True / page_match: True / keyword_recall: 0.429 / 소요시간: 16.8초

> 스쿨넷 이용관리시스템에서 이용기관 비밀번호를 초기화하는 절차는 비밀번호 초기화 팝업이 나타나면 이용기관 등록신청 시 입력한 담당자 이메일 주소를 입력하는 것입니다.
>
> 담당자 이메일을 알 수 없는 경우 문의할 고객센터 연락처는 1899-0979이며, 초기화 완료 안내 메일 발송에는 최대 5분 가량 소요될 수 있습니다.

### test2.no_catalog
- 선정 문서: 이용관리_20260702.pdf
- doc_match: True / page_match: True / keyword_recall: 0.429 / 소요시간: 23.7초

> 비밀번호 초기화 절차는 비밀번호 초기화 팝업이 나타나면 이용기관 등록신청 시 입력한 담당자 이메일 주소를 입력하는 것입니다. 담당자 이메일을 알 수 없는 경우 문의할 고객센터 연락처는 1899-0979이며, 초기화 완료 안내 메일 발송에는 최대 5분 가량 소요될 수 있습니다.

---

## final_003 — 802.11ac/802.11ax 무선랜 표준

**질문**: 무선랜 구축운영 가이드라인에서 설명하는 802.11ac와 802.11ax 무선랜 표준의 주파수 대역·속도 특성은 각각 무엇이며, 802.11ax를 160MHz 대역폭으로 사용할 때 제공 가능한 최대 속도는 얼마인가요?

**정답 (근거: ★. 무선랜 구축운영 가이드라인_0.1v_20p수정_2023년.pdf, 5페이지)**
> 802.11ac(WiFi 5)는 5GHz 대역에서 6Gbps 속도를 목표로 하며 '기가급 와이파이'로 불립니다. 802.11ax(WiFi 6)는 2.4GHz/5GHz 대역에서 대규모 혼잡 환경에서도 높은 수율을 달성하는 특성이 있습니다. 802.11ax를 160MHz 대역폭으로 사용하면 최대 9.6Gbps 속도를 제공하는 제품이 출시되고 있습니다.

### test1.catalog
- 선정 문서: ★. 무선랜 구축운영 가이드라인_0.1v_20p수정_2023년.pdf / 학교 무선인터넷 자가진단 체크리스트(안).pdf
- doc_match: True / page_match: True / keyword_recall: 0.778 / is_answer_supported: False(근거 추출 실패로 인한 오판정) / 소요시간: 890.0초

> 802.11ac는 5GHz 대역에서 6Gbps 속도를 목표로 하며, 802.11ax는 2.4GHz/5GHz 대역에서 사용되며 대규모 혼잡 환경에서도 높은 수용력을 달성합니다. 또한, 802.11ax를 160MHz 대역폭으로 사용할 때 제공 가능한 최대 속도는 9.6Gbps입니다.

### test1.no_catalog
- 선정 문서: ★. 무선랜 구축운영 가이드라인_0.1v_20p수정_2023년.pdf / ★23년 학교 유무선 운영·관리 안내서_최종.pdf / 학교 무선인터넷 자가진단 체크리스트(안).pdf
- doc_match: True / page_match: True / keyword_recall: 0.667 / 소요시간: 741.2초

> 802.11ac 무선랜 표준은 주파수 대역이 5GHz이며 최대 전송률은 6.933Gbps이고, 802.11ax 무선랜 표준은 주파수 대역이 1GHz ~ 6GHz이며 최대 전송률은 9.6Gbps입니다. 또한 802.11ax를 160MHz 대역폭으로 사용할 때 제공 가능한 최대 속도는 9.6Gbps입니다.

### test2.catalog
- 선정 문서: ★. 무선랜 구축운영 가이드라인_0.1v_20p수정_2023년.pdf / 학교 무선인터넷 자가진단 체크리스트(안).pdf
- doc_match: True / page_match: True / keyword_recall: 0.778 / 소요시간: 25.5초

> 무선랜 구축운영 가이드라인에 따르면, 802.11ac와 802.11ax 무선랜 표준의 주파수 대역 및 속도 특성은 다음과 같습니다.
>
> * **802.11ac(WiFi5):** 5GHz 대역에서 6Gbps 속도 목표 (일명 '기가급 와이파이')
> * **802.11ax(Wifi6):** 2.4GHz/5GHz대역에서, 대규모 혼잡환경에서도 높은 수율 달성
>
> 또한, **802.11ax를 160MHz 대역폭으로 사용할 때 제공 가능한 최대 속도는 9.6Gbps**입니다.

### test2.no_catalog
- 선정 문서: ★. 무선랜 구축운영 가이드라인_0.1v_20p수정_2023년.pdf / ★23년 학교 유무선 운영·관리 안내서_최종.pdf / 학교 무선인터넷 자가진단 체크리스트(안).pdf
- doc_match: True / page_match: True / keyword_recall: 0.778 / 소요시간: 69.6초

> 무선랜 구축운영 가이드라인에서 설명하는 802.11ac와 802.11ax 무선랜 표준의 주파수 대역 및 속도 특성은 다음과 같습니다.
>
> * **802.11ac (WiFi5):** 5GHz 대역에서, 6Gbps 속도 목표(일명 '기가급 와이파이')
> * **802.11ax (Wifi6):** 2.4GHz/5GHz대역에서, 대규모 혼잡환경에서도 높은 수율 달성
>
> 또한, **802.11ax를 160MHz 대역폭으로 사용할 때 제공 가능한 최대 속도는 9.6Gbps**입니다.

---

## 한눈에 보기 — 답변 정오 요약

| 문항 | test1.catalog | test1.no_catalog | test2.catalog | test2.no_catalog |
|---|---|---|---|---|
| final_001 (Cat.5 Data Performance) | 100Mbps ❌ | 100Mbps ❌ (Cat.5e도 오독) | 10Mbps ✅ | 10Mbps ✅ |
| final_002 (고객센터 번호) | 1899-0970 ❌ | 1899-0970 ❌ | 1899-0979 ✅ | 1899-0979 ✅ |
| final_002 (메일 발송 시간) | 확인 불가 ❌ | 약 1분 ❌ | 최대 5분 ✅ | 최대 5분 ✅ |
| final_003 (802.11ax 대역) | 2.4/5GHz ✅ | 1GHz~6GHz ❌ | 2.4/5GHz ✅ | 2.4/5GHz ✅ |

정답: `(정답)`, ✅ 부합 / ❌ 불일치. 상세 원인 분석은 [`FINAL_REPORT.md §2.3, §8.1`](FINAL_REPORT.md)에 정리되어 있다.
