---
name: analysis-reporting
description: 프로젝트 분석, 수정, 재검증, MLflow 등록, 로컬 서빙 결과를 사용자용 리포트로 정리한다.
---

# Analysis Reporting

## When To Use

- Step 10 분석 리포트를 콘솔이나 파일로 보여줘야 할 때
- `ml-agent-report.json`, `mlflow-run-verification.json`, `job_template.yml` 결과를 한 화면에 요약해야 할 때
- 초급자에게 남은 문제와 다음 조치를 표 또는 짧은 목록으로 안내해야 할 때

## Report Contents

- project path
- registration status
- MLflow dependency and logging status
- Job Template readiness
- local serving status, health endpoint, predict endpoint
- run_model execution result
- MLflow run id, params, metrics, artifacts
- applied changes and remaining issues
- next actions

## Workflow

- 분석 결과와 검증 결과를 먼저 수집한다.
- 초급자 화면에는 핵심 요약과 다음 선택지만 짧게 보여준다.
- 파일 리포트에는 JSON 형태로 상세 정보를 저장한다.
- 실패한 항목은 error-log-repair skill로 넘길 수 있게 원인과 재검증 명령을 남긴다.

## Safety

- API key, password, token은 리포트에 원문으로 남기지 않는다.
- 사용자가 승인하지 않은 파일 변경을 적용 완료로 표시하지 않는다.
