---
name: job-template-draft
description: 학습 entrypoint, MLflow 등록 명령, 로컬 서빙 호출 결과를 기반으로 Job Template YAML 초안을 만든다.
---

# Job Template Draft

## When To Use

- 사용자가 Job Template을 모르지만 학습/등록/서빙 값을 YAML로 남겨야 할 때
- `run_model.py`, resource, MLflow 설정, serving endpoint를 한 파일로 정리해야 할 때
- 로컬 서빙 테스트 후 호출 값을 ML Platform 실행 설정으로 연결해야 할 때

## YAML Contents

- project metadata
- entrypoint and command
- resource defaults: queue, cpu, gpu, memory, python_version
- MLflow tracking mode and registry model name placeholder
- artifact paths
- local serving app, health endpoint, predict endpoint
- smoke test result

## Workflow

- 분석 결과에서 entrypoint, requirements, model artifact를 가져온다.
- 로컬 서빙 plan 또는 smoke test 결과를 YAML에 반영한다.
- 값이 없으면 안전한 placeholder를 사용하고 사용자 입력이 필요한 항목을 표시한다.
- YAML 생성 후 재검증 명령을 안내한다.

## Safety

- credential 값을 YAML에 직접 쓰지 않는다.
- 원격 업로드나 배치 실행은 YAML 생성과 분리한다.
