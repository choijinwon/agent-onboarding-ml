---
name: ai-studio-runtime-template
description: AI Studio 환경변수, run_model.py, ModelWrapper, MLflow logging 코드를 표준 템플릿으로 구성한다.
---

# AI Studio Runtime Template

## When To Use

- AI Studio 환경 설정값을 파일이나 화면으로 입력받아 MLflow 등록 코드에 연결해야 할 때
- `MLFLOW_TRACKING_URL`, username, password, experiment, registered model name 매핑이 필요할 때
- `run_model.py`와 `ModelWrapper`가 없거나 표준 파라미터가 빠졌을 때

## Required Settings

- `MLFLOW_TRACKING_URL`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_EXPERIMENT_NAME`
- `MLFLOW_REGISTER_MODEL_NAME`

## Workflow

- 빈 설정은 placeholder로 남기고 credential은 코드에 직접 쓰지 않는다.
- `ai_studio.env`를 우선 사용하고, 비어 있으면 로컬 `file:./mlruns`로 dry-run 가능하게 한다.
- `mlflow.start_run`, params, metrics, artifacts, input_example, pyfunc.log_model 연결을 점검한다.
- `python_model`, `artifact_path`, `code_paths`, `artifacts`, `pip_requirements`, `registered_model_name`을 확인한다.

## Safety

- password/API key는 로그와 리포트에 원문 출력하지 않는다.
- 기존 `run_model.py` 덮어쓰기는 review_required로 분류한다.
