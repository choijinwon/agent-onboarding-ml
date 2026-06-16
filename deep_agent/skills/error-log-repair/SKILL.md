---
name: error-log-repair
description: 저장된 에러 로그와 실패한 실행 결과를 기반으로 원인 후보를 찾고 승인 기반 재수정안을 만든다.
---

# Error Log Repair

## When To Use

- `run_model.py`, MLflow 등록, local serving, TUI chatbot 실행이 실패했을 때
- 사용자가 에러 로그를 붙여넣거나 `chat_errors`, `dev_runs`, `sessions` 로그를 분석해달라고 할 때
- 같은 오류를 바탕으로 다시 자동 수정해야 할 때

## Workflow

- traceback, command, exit code, stderr, 최근 변경 파일을 분리한다.
- dependency, path, encoding, MLflow config, model artifact, serving request 문제로 분류한다.
- safe 항목은 승인 후 자동 보완하고 review_required 항목은 미리보기를 만든다.
- 수정 후 같은 검증 명령을 다시 안내한다.

## Output

- 에러 요약
- 원인 후보
- 수정 정책 분류
- 적용 가능한 수정안
- 재검증 명령

## Safety

- 로그 안 credential은 마스킹한다.
- 삭제, 프로젝트 외부 수정, 모델 artifact 교체는 차단한다.
