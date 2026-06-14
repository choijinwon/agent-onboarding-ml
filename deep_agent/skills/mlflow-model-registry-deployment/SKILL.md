---
name: mlflow-model-registry-deployment
description: MLflow Model Registry와 Deployment 흐름을 기준으로 모델 등록, 버전, 배포 준비 상태를 점검한다.
---

# MLflow Model Registry And Deployment

## When To Use

- 모델 파일을 registry에 등록하거나 버전 관리해야 할 때
- batch 또는 real-time serving 배포 준비 상태를 확인할 때
- Docker, Kubernetes, SageMaker, Azure ML 같은 배포 목표를 정리해야 할 때

## Checklist

- model name, version, alias 또는 stage 정책
- signature, input example, dependency, Python version
- artifact URI와 registry URI
- local serving health/predict 테스트
- rollback과 promotion 기준

## Output

- registry readiness
- deployment target 후보
- serving 검증 결과
- 등록 패키지에 포함할 파일 목록
