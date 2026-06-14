---
name: retrieving-mlflow-traces
description: MLflow trace를 상태, 시간, session, user, metadata 기준으로 검색하는 방법을 안내한다.
---

# Retrieving MLflow Traces

## When To Use

- 실패 trace를 찾아야 할 때
- 특정 session/user/time range의 실행 기록을 모아야 할 때
- batch 검증이나 운영 장애 분석용 trace 목록이 필요할 때

## Query Checklist

- tracking URI와 experiment를 먼저 확인한다.
- status, request time, session ID, run ID, user metadata를 기준으로 좁힌다.
- 폐쇄망에서는 외부 서비스 호출 없이 로컬/내부 MLflow endpoint만 사용한다.
- 결과는 trace ID, status, latency, 주요 error message 중심으로 요약한다.

## Output

- 검색 조건
- 후보 trace 목록
- 다음 분석에 넘길 trace ID
- 재현 명령 또는 후속 skill 추천
