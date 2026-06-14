---
name: agent-evaluation
description: MLflow Evaluation으로 agent 품질 평가 데이터셋, scorer, 실행, 결과 분석 흐름을 만든다.
---

# Agent Evaluation

## When To Use

- agent 또는 모델 등록 전 품질 기준을 만들 때
- accuracy, relevance, groundedness, latency 같은 평가 지표가 필요할 때
- 수정 전후 품질이 좋아졌는지 비교해야 할 때

## Workflow

- 평가 목적과 입력/정답/참조 데이터 형태를 정한다.
- dataset 후보를 만들거나 기존 evaluation set을 찾는다.
- scorer와 threshold를 선택한다.
- evaluation 실행 계획과 결과 리포트 위치를 정한다.
- 실패 케이스를 수정 후보로 연결한다.

## Safety

- 평가 데이터에 민감정보가 있으면 마스킹하거나 샘플링한다.
- 폐쇄망에서 사용 가능한 judge model 또는 rule-based scorer를 우선 사용한다.
