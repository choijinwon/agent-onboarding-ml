---
name: error-log-repair
description: 저장된 에러 로그를 기반으로 원인 후보를 찾고 dry-run 수정안을 다시 생성한다.
---

# Error Log Repair

## When To Use

- 사용자가 에러 로그 기반 재수정을 요청할 때
- `chat_errors/`에 저장된 로그를 다시 분석해야 할 때
- validate, apply, report 이후 실패 로그가 남았을 때

## Inputs

- error log id 또는 json 파일 경로
- project path
- previous fix report
- command output
- stack trace

## Workflow

1. 에러 로그를 읽는다.
2. 민감정보가 마스킹되어 있는지 확인한다.
3. MLflow, requirements, arguments, Job Template, Qwen 연결 문제로 분류한다.
4. 원인 후보와 증거를 요약한다.
5. `ml-agent fix <project> --dry-run` 형태의 재수정 명령을 제안한다.
6. apply 전에는 반드시 preview와 사용자 승인을 요구한다.

## Output

- 에러 요약
- 원인 후보
- 재수정 액션
- 추천 명령
- 재검증 명령

## Safety

- API key, token, password, secret은 출력하지 않는다.
- 로그 원문 전체를 불필요하게 반복하지 않는다.
- 삭제 작업은 제안하지 않는다.
