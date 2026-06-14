---
name: mlflow-registration-check
description: MLflow 등록 준비 상태를 점검하고 누락된 logging, artifact, registry 조건을 설명한다.
---

# MLflow Registration Check

## When To Use

- 사용자가 MLflow 설정 검증을 요청할 때
- 프로젝트가 모델 등록 대상인지 판단해야 할 때
- `mlflow`, `tracking_uri`, `experiment`, `artifact`, `registry` 관련 오류가 있을 때

## Checklist

- MLflow tracking URI 설정 확인
- experiment/run 생성 흐름 확인
- params, metrics, artifacts logging 확인
- model artifact 저장 경로 확인
- requirements와 Python version 확인
- model registry 등록에 필요한 이름, stage, signature 후보 확인

## Output

- 등록 가능 여부 요약
- 누락 항목 목록
- 영향도
- dry-run 수정 제안
- 재검증 명령

## Safety

- 기본은 read-only scan이다.
- 파일 수정 전에는 반드시 preview를 먼저 제공한다.
- API key, token, password는 출력하지 않는다.
