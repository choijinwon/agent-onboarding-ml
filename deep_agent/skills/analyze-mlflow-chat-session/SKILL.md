---
name: analyze-mlflow-chat-session
description: MLflow에 남은 multi-turn chat session을 재구성하고 대화 실패 지점을 찾는다.
---

# Analyze MLflow Chat Session

## When To Use

- 여러 턴의 agent/chat 실행 중 어느 시점에서 품질이 깨졌는지 확인할 때
- session ID, user ID, trace 묶음 기반으로 대화를 분석할 때
- 응답 누락, tool loop, context 손실을 진단할 때

## Workflow

- session 단위로 message, trace, assessment를 시간순으로 정렬한다.
- 실패 전후의 사용자 요청, tool call, model output 차이를 비교한다.
- context overflow, 잘못된 routing, 누락된 memory, tool error 여부를 확인한다.
- 다시 실행할 최소 테스트 시나리오를 제안한다.

## Output

- 대화 흐름 요약
- 실패 턴
- 원인 후보
- 수정 및 재검증 계획
