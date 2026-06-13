# ML Platform Console Assistant POC

폐쇄망 ML Platform 등록 지원을 위한 Launch Mode POC입니다.

처음 실행하면 사용자의 숙련도에 따라 세 가지 모드 중 하나를 선택합니다.

1. 초급자 모드: 단계별 Wizard 방식
2. 중급자 모드: Chat + Wizard 혼합
3. 고급자 모드: CLI Command 중심

## 실행

```bash
python3 ml_agent.py
```

또는 저장소 루트에서 실행 스크립트를 사용할 수 있습니다.

```bash
./ml-agent
```

## 고급자 명령

```bash
./ml-agent analyze ./project
./ml-agent validate ./project
./ml-agent fix ./project --dry-run
./ml-agent apply ./project
./ml-agent report ./project
./ml-agent chat
./ml-agent profile
```

JSON 출력이 필요한 경우 `--json` 옵션을 사용할 수 있습니다.

```bash
./ml-agent validate ./project --json
./ml-agent profile --json
```

## Deep Agents 참고 구조

이 POC는 [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)의 agent harness 개념을 참고했습니다.

- sub-agents: `project-scanner`, `mlflow-validator`, `job-template-planner`, `log-analyzer`
- filesystem permissions: 읽기는 허용, 쓰기는 human-in-the-loop 승인, `.git`과 secret 경로 쓰기는 차단
- skills: MLflow 등록 점검, Job Template 초안, 폐쇄망 검증 절차
- memory: 등록 규칙과 팀 Job Template 컨벤션을 별도 메모리 경로로 선언
- context policy: 긴 분석 결과는 요약하고 상세 증거는 리포트 산출물로 남김

현재 구현은 폐쇄망 POC를 위해 외부 런타임 의존성을 강제하지 않는 독립 프로파일입니다.
나중에 실제 LLM 런타임을 연결할 때 `deepagents.create_deep_agent`의 `tools`, `subagents`, `skills`, `permissions`, `memory` 설정으로 옮길 수 있습니다.

## 모드 전환

실행 중 다음 명령으로 모드를 바꿀 수 있습니다.

```text
/mode beginner
/mode intermediate
/mode advanced
/모드 초급자
/모드 중급자
/모드 고급자
```

## 안전 규칙

- 기본 동작은 read-only scan입니다.
- 파일 수정 전에는 dry-run 또는 수정안 미리보기를 보여줍니다.
- 사용자 승인 없이 파일을 수정하지 않습니다.
- 삭제 작업은 하지 않습니다.
- 적용 후에는 재검증을 수행합니다.
