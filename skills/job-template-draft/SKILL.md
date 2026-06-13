---
name: job-template-draft
description: 학습 entrypoint와 실행 인자를 기반으로 AI Studio 또는 ML Platform Job Template 초안을 만든다.
---

# Job Template Draft

## When To Use

- 사용자가 Job Template 초안을 요청할 때
- 학습 코드의 entrypoint, arguments, resource 요구사항을 정리해야 할 때
- AI Studio 또는 ML Platform 제출 양식을 만들 때

## Inputs

- entrypoint script
- command arguments
- requirements file
- Python version
- queue
- CPU, GPU, memory
- environment variables

## Defaults

환경 변수에서 다음 값을 우선 사용한다.

- `AI_STUDIO_DEFAULT_QUEUE`
- `AI_STUDIO_DEFAULT_GPU`
- `AI_STUDIO_DEFAULT_CPU`
- `AI_STUDIO_DEFAULT_MEMORY`
- `AI_STUDIO_PYTHON_VERSION`
- `ML_PLATFORM_DEFAULT_QUEUE`
- `ML_PLATFORM_DEFAULT_GPU`
- `ML_PLATFORM_DEFAULT_CPU`
- `ML_PLATFORM_DEFAULT_MEMORY`
- `ML_PLATFORM_PYTHON_VERSION`

## Output

- Job Template 초안
- 필요한 파일 목록
- dry-run 변경안
- validate 명령

## Safety

- `fix`는 기본 dry-run이다.
- `apply`가 명시적으로 실행되기 전에는 파일을 쓰지 않는다.
- 삭제 작업은 하지 않는다.
