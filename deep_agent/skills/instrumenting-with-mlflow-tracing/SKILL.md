---
name: instrumenting-with-mlflow-tracing
description: Python/TypeScript GenAI 앱에 MLflow Tracing을 추가하기 위한 점검과 수정안을 만든다.
---

# Instrumenting With MLflow Tracing

## When To Use

- OpenAI, LangChain, LangGraph, LiteLLM, Anthropic, custom agent 코드에 tracing을 붙일 때
- LLM 호출, tool call, retriever, chain 실행 흐름을 MLflow trace로 남기고 싶을 때
- 폐쇄망 POC에서 tracing 적용 전 변경 범위를 검토해야 할 때

## Workflow

- 프레임워크와 LLM 호출 지점을 찾는다.
- MLflow tracking URI와 experiment 설정 위치를 확인한다.
- 가능한 경우 autolog 또는 trace decorator 방식의 최소 수정안을 제안한다.
- tracing 적용 후 생성될 span, inputs, outputs, metadata 범위를 설명한다.
- 수정 전 dry-run preview를 보여주고 승인 후에만 적용한다.

## Safety

- prompt, token, API key, 개인정보가 trace에 저장될 수 있는지 확인한다.
- 민감정보 마스킹 설정이 없으면 먼저 보완 항목으로 표시한다.
- 운영 코드 변경 전 로컬 샘플 요청으로 검증한다.
