# AI ML 온보딩 POC

폐쇄망 AI/ML 프로젝트 온보딩과 모델 등록 지원을 위한 Launch Mode POC입니다.

빠른 셋팅은 [QUICKSTART.md](QUICKSTART.md)를 먼저 확인하세요.

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

Windows 10/11에서는 PowerShell 또는 CMD에서 다음처럼 실행합니다.

```powershell
py -3 ml_agent.py
.\ml-agent.cmd
```

필수 조건:

- Windows 10 또는 Windows 11
- Python 3.10 이상
- 폐쇄망 환경에서는 Python 설치 파일과 이 저장소를 사전에 반입

처음 셋팅할 때는 샘플 환경 파일을 복사한 뒤 내부 Qwen endpoint 값을 수정합니다.

Linux/macOS:

```bash
cp .env.example .env
python3 ml_agent.py init
```

Windows 10/11:

```powershell
copy .env.example .env
.\ml-agent.cmd init
```

Python 확인:

```powershell
py -3 --version
python --version
```

PowerShell에서 한글이 깨지면 다음을 먼저 실행합니다.

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
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
./ml-agent config
./ml-agent init
./ml-agent prompts
./ml-agent errors list
./ml-agent errors analyze error-YYYYMMDDTHHMMSSZ
```

Windows 10/11:

```powershell
.\ml-agent.cmd analyze .\project
.\ml-agent.cmd validate .\project
.\ml-agent.cmd fix .\project --dry-run
.\ml-agent.cmd apply .\project
.\ml-agent.cmd report .\project
.\ml-agent.cmd chat
.\ml-agent.cmd profile
.\ml-agent.cmd config
.\ml-agent.cmd init
.\ml-agent.cmd prompts
.\ml-agent.cmd errors list
.\ml-agent.cmd errors analyze error-YYYYMMDDTHHMMSSZ
```

JSON 출력이 필요한 경우 `--json` 옵션을 사용할 수 있습니다.

```bash
./ml-agent validate ./project --json
./ml-agent profile --json
```

Windows 10/11:

```powershell
.\ml-agent.cmd validate .\project --json
.\ml-agent.cmd profile --json
.\ml-agent.cmd prompts --json
```

## Deep Agents 참고 구조

이 POC는 [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)의 agent harness 개념을 참고했습니다.

- sub-agents: `project-scanner`, `mlflow-validator`, `job-template-planner`, `log-analyzer`
- filesystem permissions: 읽기는 허용, 쓰기는 human-in-the-loop 승인, `.git`과 secret 경로 쓰기는 차단
- skills: MLflow 등록 점검, Job Template 초안, 폐쇄망 검증 절차
- memory: 등록 규칙과 팀 Job Template 컨벤션을 별도 메모리 경로로 선언
- context policy: 긴 분석 결과는 요약하고 상세 증거는 리포트 산출물로 남김
- skills 저장소: `SKILL_STORE_DIR` 값으로 지정하며 기본값은 `skills`

현재 구현은 폐쇄망 POC를 위해 외부 런타임 의존성을 강제하지 않는 독립 프로파일입니다.
나중에 실제 LLM 런타임을 연결할 때 `deepagents.create_deep_agent`의 `tools`, `subagents`, `skills`, `permissions`, `memory` 설정으로 옮길 수 있습니다.

## 환경 변수

`.env.example`에는 폐쇄망 Qwen endpoint, 모델 목록, Deep Agent 옵션, 작업 디렉터리, 등록 패키지, 리포트 경로 샘플이 포함되어 있습니다.

주요 값:

- `QWEN_API_KEY`: 내부 Qwen API 키
- `QWEN_BASE_URL`: 내부 OpenAI-compatible Qwen endpoint
- `QWEN_MODEL`: 기본 모델
- `QWEN_MODELS`: 선택 가능한 모델 목록
- `ENABLE_MULTI_AGENT`: sub-agent 분담 사용 여부
- `ENABLE_HARNESS_SKILLS`: Deep Agent skill 저장/로드 사용 여부
- `SKILL_STORE_DIR`: skill 저장 경로, 기본값 `skills`
- `CHAT_WORKSPACE_DIR`: agent 작업 공간
- `REGISTRATION_PACKAGE_DIR`: 등록 패키지 산출물 경로
- `FIX_REPORT_DIR`: 수정 리포트 경로

설정 요약 확인:

```bash
./ml-agent config
```

Windows 10/11:

```powershell
.\ml-agent.cmd config
```

## 프롬프트와 스킬

기본 프롬프트는 `PROMPT_STORE_PATH` 값이 가리키는 `prompt_templates.json`에 저장됩니다.

```bash
./ml-agent prompts
./ml-agent prompts --json
```

Windows 10/11:

```powershell
.\ml-agent.cmd prompts
.\ml-agent.cmd prompts --json
```

기본 스킬은 `skills/` 아래에 저장됩니다.

```text
skills/
├── closed-network-validation/
├── error-log-repair/
├── job-template-draft/
└── mlflow-registration-check/
```

## 에러 로그 관리

에러 로그는 `CHAT_ERROR_DIR` 값이 가리키는 `chat_errors/`에 저장됩니다.
저장된 에러 로그는 이후 재분석해서 dry-run 수정안을 다시 만드는 기준으로 사용합니다.

에러 로그 저장:

```bash
./ml-agent errors record "ModuleNotFoundError: No module named mlflow"
```

에러 로그 목록:

```bash
./ml-agent errors list
```

에러 로그 분석:

```bash
./ml-agent errors analyze error-YYYYMMDDTHHMMSSZ
```

Windows 10/11:

```powershell
.\ml-agent.cmd errors record "ModuleNotFoundError: No module named mlflow"
.\ml-agent.cmd errors list
.\ml-agent.cmd errors analyze error-YYYYMMDDTHHMMSSZ
```

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
