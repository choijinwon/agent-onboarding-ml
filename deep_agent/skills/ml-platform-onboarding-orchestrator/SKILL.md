---
name: ml-platform-onboarding-orchestrator
description: 모델 프로젝트를 ML Platform 표준 등록 흐름으로 끝까지 안내하는 최상위 오케스트레이션 skill이다.
---

# ML Platform Onboarding Orchestrator

## When To Use

- 사용자가 프로젝트를 등록 가능하게 만들거나 전체 자동화를 요청할 때
- 초급자/챗봇 모드에서 사용자가 MLflow, Job Template, Serving 절차를 몰라도 진행해야 할 때
- 여러 skill 중 어떤 것을 먼저 적용할지 결정해야 할 때

## Workflow

1. 프로젝트 경로와 사용자 모드를 확인한다.
2. read-only scan으로 framework, entrypoint, artifact, requirements, config, input example을 확인한다.
3. 필요한 하위 skill을 순서대로 적용한다: standardization, experiment tracking, registry, serving, job template, error repair.
4. 수정은 정책에 따라 safe, review_required, blocked로 분류한다.
5. 승인된 항목만 적용하고 재검증 결과와 다음 조치를 리포트한다.

## Output

- 현재 단계
- 사용한 skill 이름
- 자동 적용 가능 항목
- 검토 필요 항목
- 차단 항목
- 다음 사용자 선택지

## Safety

- 초급자와 챗봇 모드에서는 파일 수정 전 승인 또는 정책 확인을 요구한다.
- 파일 삭제, 프로젝트 외부 수정, credential 직접 삽입은 차단한다.
- 원격 MLflow/Bucket 업로드는 사용자가 설정값을 제공하고 실행을 승인한 경우에만 안내한다.
