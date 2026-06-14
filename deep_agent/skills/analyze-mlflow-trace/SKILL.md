---
name: analyze-mlflow-trace
description: MLflow trace/span 정보를 바탕으로 실패 원인과 수정 후보를 분석한다.
---

# Analyze MLflow Trace

## When To Use

- trace ID, span ID, 실패한 LLM 실행 로그가 있을 때
- tool call 실패, latency 증가, 빈 응답, 잘못된 답변 원인을 찾을 때
- 오류 로그 기반 재수정이 필요할 때

## Workflow

- trace 상태, span tree, error span, latency가 큰 span을 우선 확인한다.
- 입력, 출력, metadata, assessment가 있으면 실패 지점을 요약한다.
- 코드 위치와 requirements, 환경변수를 연결해 원인 후보를 만든다.
- 수정안은 evidence, impact, dry-run preview 순서로 제시한다.

## Output

- 실패 지점 요약
- 원인 후보
- 관련 파일 또는 설정
- 재현/재검증 명령
