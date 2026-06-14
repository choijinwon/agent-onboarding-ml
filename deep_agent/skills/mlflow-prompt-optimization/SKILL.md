---
name: mlflow-prompt-optimization
description: MLflow Prompt Optimization 흐름을 참고해 프롬프트 개선 실험과 평가 루프를 설계한다.
---

# MLflow Prompt Optimization

## When To Use

- 프롬프트를 수정했을 때 실제 품질이 좋아졌는지 검증해야 할 때
- 여러 prompt 후보를 평가하고 최고안을 고르고 싶을 때
- agent 응답 품질 회귀를 방지하고 싶을 때

## Workflow

- 목표 metric과 평가 데이터셋을 먼저 정한다.
- baseline prompt와 후보 prompt를 분리한다.
- judge model, rule-based scorer, human review 중 폐쇄망에서 가능한 방식을 선택한다.
- 결과는 prompt version, score, 실패 케이스, 다음 실험으로 정리한다.

## Safety

- 민감정보가 포함된 prompt 또는 평가 데이터는 마스킹한다.
- 자동 개선안은 운영 반영 전 반드시 dry-run과 평가 리포트를 거친다.
