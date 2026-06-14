---
name: mlflow-experiment-tracking
description: MLflow Experiment Tracking으로 params, metrics, artifacts, datasets 기록 상태를 점검한다.
---

# MLflow Experiment Tracking

## When To Use

- 학습 코드가 실험 재현에 필요한 정보를 충분히 남기는지 확인할 때
- params, metrics, artifacts, datasets, tags 누락을 점검할 때
- 등록 전 학습 실행 이력을 리포트로 정리해야 할 때

## Checklist

- tracking URI와 experiment name 설정
- run lifecycle 시작/종료 위치
- hyperparameter와 runtime argument logging
- metric key와 step 기록
- model artifact, dataset, source version 기록

## Output

- tracking readiness
- 누락 logging 항목
- 코드 수정 preview
- validate/report 명령
