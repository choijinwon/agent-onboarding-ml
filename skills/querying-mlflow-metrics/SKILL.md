---
name: querying-mlflow-metrics
description: MLflow metrics를 조회해 token usage, latency, error rate, 품질 추세를 분석한다.
---

# Querying MLflow Metrics

## When To Use

- 최근 실행의 latency, token, cost, error trend를 보고 싶을 때
- 모델/agent 수정 전후 지표를 비교할 때
- 등록 후보 모델의 운영 안정성을 요약해야 할 때

## Workflow

- experiment/run 범위와 시간 구간을 정한다.
- metric key, tag, model name, dataset 기준으로 집계한다.
- 이상치와 regression을 찾는다.
- 표 또는 JSON으로 결과를 요약한다.

## Output

- 조회 범위
- 핵심 지표 요약
- 이상 징후
- 다음 조치
