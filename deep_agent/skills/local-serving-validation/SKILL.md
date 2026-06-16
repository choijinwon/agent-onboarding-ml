---
name: local-serving-validation
description: 등록 전 로컬 FastAPI 서빙, health/predict 호출, input_example 추론 테스트를 검증한다.
---

# Local Serving Validation

## When To Use

- 사용자가 로컬 모델 서빙까지 확인하고 싶을 때
- `serving_app.py`, `/health`, `/predict`, input example 테스트가 필요할 때
- MLflow에서 다운로드한 모델을 로드해 추론까지 확인해야 할 때

## Workflow

- `serving_app.py` 존재 여부와 FastAPI/uvicorn 의존성을 확인한다.
- `input_example.json`을 요청 payload로 사용한다.
- `/health`와 `/predict` 응답 코드와 body를 기록한다.
- 성공/실패 결과를 Step 리포트와 Job Template YAML에 반영한다.

## Output

- health endpoint
- predict endpoint
- local serve command
- smoke test status
- 실패 원인과 재수정 후보

## Safety

- 실제 서버 상시 실행은 사용자가 명시적으로 요청한 경우에만 안내한다.
- 외부 네트워크 호출 대신 `127.0.0.1` 기본값을 사용한다.
