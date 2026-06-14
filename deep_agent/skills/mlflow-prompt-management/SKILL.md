---
name: mlflow-prompt-management
description: MLflow Prompt Management로 프롬프트 버전, lineage, 평가 연결 흐름을 점검한다.
---

# MLflow Prompt Management

## When To Use

- 프롬프트를 코드에 직접 박아두지 않고 버전 관리하고 싶을 때
- agent 또는 LLM 앱의 prompt 변경 이력을 추적해야 할 때
- 평가 결과와 prompt 버전을 연결해 회귀를 확인해야 할 때

## Workflow

- 프롬프트가 저장된 파일, 템플릿, 환경변수, 코드 위치를 찾는다.
- prompt name, version, tag, owner, 사용 모델을 정리한다.
- 평가 데이터셋과 prompt 버전 연결 기준을 제안한다.
- 폐쇄망에서는 내부 prompt registry 또는 파일 기반 prompt store를 우선 사용한다.

## Output

- prompt inventory
- 버전 관리 후보
- 평가 연결 계획
- dry-run 수정안
