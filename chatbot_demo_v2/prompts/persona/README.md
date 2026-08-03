# 독립 답변 페르소나 확장

이 폴더의 파일은 기존 `composer_faq.md`, `composer_rag.md`와 분리된 선택적 확장입니다.

- `persona.md`: 상담자 역할과 말투
- `response_policy.md`: 질문 유형별 답변 구조·길이·표현 정책
- `response_examples.md`: few-shot 답변 형식 예시

기본값은 활성화입니다. 비활성화하려면 `chatbot_demo_v2/.env`에 다음을 설정합니다.

```env
PERSONA_PROMPTS_ENABLED=false
```

파일을 하나 이상 삭제해도 남아 있는 파일만 적용되며, 전부 삭제하면 기존 composer 프롬프트만 사용합니다.
프롬프트는 `PromptLoader`의 핫리로드 대상이므로 서버 재시작 없이 다음 호출부터 수정 내용이 반영됩니다.
