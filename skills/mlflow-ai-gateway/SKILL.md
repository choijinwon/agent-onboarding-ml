---
name: mlflow-ai-gateway
description: MLflow AI Gateway 개념을 참고해 내부 LLM 라우팅, 비용, 접근 제어, fallback 설정을 점검한다.
---

# MLflow AI Gateway

## When To Use

- 여러 LLM provider 또는 내부 모델을 하나의 OpenAI 호환 endpoint로 쓰고 싶을 때
- Qwen, GPT 계열, 내부 모델 라우팅과 fallback을 관리해야 할 때
- 비용, rate limit, credential 노출을 통제해야 할 때

## Checklist

- base URL, model catalog, route name, fallback 순서를 확인한다.
- API key가 코드나 로그에 노출되지 않는지 확인한다.
- rate limit, timeout, retry, streaming 옵션을 정리한다.
- 폐쇄망에서는 외부 provider 호출을 차단하고 내부 endpoint만 허용한다.

## Output

- gateway readiness 요약
- model routing 표
- 보안/비용 위험
- 재검증 명령
