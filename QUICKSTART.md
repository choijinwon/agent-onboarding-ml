# AI ML 온보딩 퀵 가이드

Windows 10/11 폐쇄망 환경에서 AI ML 온보딩 POC를 빠르게 실행하는 절차입니다.
기본 실행은 `ml-agent.cmd`를 사용합니다.

## 1. 준비물

- Windows 10 또는 Windows 11
- Python 3.10 이상
- 내부 Qwen OpenAI-compatible endpoint
- 이 저장소 파일
- 권장 터미널: Windows Terminal, WezTerm, Alacritty

Python 확인:

```powershell
py -3 --version
python --version
```

한글 출력이 깨지면 PowerShell에서 먼저 실행합니다.

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## 2. 저장소 받기

```powershell
git clone https://github.com/choijinwon/agent-onboarding-ml.git
cd agent-onboarding-ml
```

폐쇄망이면 인터넷 가능한 PC에서 저장소를 압축해 반입해도 됩니다.

## 3. 환경 파일 만들기

```powershell
copy .env.example .env
```

`.env`에서 내부 Qwen 값만 먼저 수정합니다.

```env
QWEN_API_KEY=your-internal-qwen-key
QWEN_BASE_URL=http://xxx.xxx.xxx.xxx:포트/v1
QWEN_MODEL=qwen3.6
QWEN_MODELS=qwen3.6,qwen3.5,gpt20,gamma
ENABLE_RICH_CONSOLE=true
```

Qwen은 Agent가 분석과 안내에 사용하는 LLM입니다.
등록 대상 ML 모델은 실행할 때 프로젝트 경로로 넘깁니다.

## 4. 초기화

```powershell
.\ml-agent.cmd init
```

생성 또는 확인되는 주요 경로:

```text
skills/
chat_errors/
fix_reports/
registration_packages/
```

## 5. 설정 확인

```powershell
.\ml-agent.cmd config
```

주요 확인 값:

- `qwen_base_url`
- `qwen_model`
- `qwen_models`
- `multi_agent`
- `harness_skills`
- `skill_store_dir`

## 6. 초급자 Wizard 실행

```powershell
.\ml-agent.cmd
```

첫 화면에서 모드를 선택합니다.

```text
1. 초급자 모드
2. 중급자 모드
3. 고급자 모드
```

처음 사용하는 경우에는 `1. 초급자 모드`를 권장합니다.
초급자 모드는 `Tab 1/10`부터 `Tab 10/10`까지 한 단계씩 보여줍니다.
파일 수정은 미리보기와 사용자 승인 후에만 진행됩니다.

OpenCode처럼 하단 입력 박스 안에서 바로 타이핑되는 TUI를 쓰려면 Textual 옵션을 설치한 뒤 실행합니다.

```powershell
py -3 -m pip install ".[tui]"
.\ml-agent.cmd tui
```

Textual 미설치 상태에서는 설치 안내가 출력되고 기존 콘솔 Wizard는 계속 사용할 수 있습니다.

## 7. 자주 쓰는 명령

```powershell
.\ml-agent.cmd analyze .\project
.\ml-agent.cmd validate .\project
.\ml-agent.cmd fix .\project --dry-run
.\ml-agent.cmd apply .\project
.\ml-agent.cmd serve .\project --dry-run
.\ml-agent.cmd report .\project
.\ml-agent.cmd profile
.\ml-agent.cmd tui
.\ml-agent.cmd deepagents
.\ml-agent.cmd deepagents --source "$env:USERPROFILE\Downloads\deepagents-main.zip"
.\ml-agent.cmd config
.\ml-agent.cmd prompts
.\ml-agent.cmd errors list
```

여기서 `.\project`는 등록하려는 ML 모델 프로젝트 경로입니다.
예를 들어 학습 코드, `requirements.txt`, MLflow artifact, 모델 파일이 들어 있는 폴더를 넘깁니다.

JSON 출력:

```powershell
.\ml-agent.cmd validate .\project --json
.\ml-agent.cmd profile --json
.\ml-agent.cmd deepagents --json
.\ml-agent.cmd deepagents --source "$env:USERPROFILE\Downloads\deepagents-main.zip" --json
.\ml-agent.cmd prompts --json
```

`deepagents`는 repo 내부 `deepagents_source`를 확인합니다.
`--source`는 다운로드한 DeepAgents zip으로 비교 검증할 때 사용합니다.

## 8. DeepAgents 소스 확인

DeepAgents 참고 소스는 repo 안에 포함되어 있습니다.
OpenCode 설정은 `.opencode/`에 있으며, Deep Agent만 적용되어 있습니다.

```text
deepagents_source/
└── deepagents-main/
    ├── LICENSE
    ├── README.md
    └── libs/
```

확인 명령:

```powershell
.\ml-agent.cmd deepagents
```

다운로드한 zip과 비교할 때:

```powershell
.\ml-agent.cmd deepagents --source "$env:USERPROFILE\Downloads\deepagents-main.zip"
```

OpenCode Deep Agent 설정 확인:

```powershell
type .opencode\agent\deep-agent.md
```

## 9. 이미지형 TUI 화면

OpenCode처럼 어두운 배경과 하단 `Plan / Build` 상태바가 있는 TUI를 기본으로 사용합니다.
최신 터미널에서 가장 잘 보입니다.

![AI ML 온보딩 TUI 미리보기](docs/tui-preview.svg)

```env
ENABLE_RICH_CONSOLE=true
ENABLE_TUI_BACKGROUND=false
ENABLE_TUI_INPUT_PANEL=true
```

위 이미지처럼 보이지 않으면 현재 세션에서 컬러 출력을 강제로 켭니다.

```powershell
$env:FORCE_COLOR=1
.\ml-agent.cmd
```

화면에 흰 박스가 생기면 배경색 렌더링을 끄세요.

```powershell
$env:DISABLE_TUI_BACKGROUND=1
.\ml-agent.cmd
```

입력 박스까지 깨지는 터미널에서는 입력 패널 배경만 끕니다.

```powershell
$env:DISABLE_TUI_INPUT_PANEL=1
.\ml-agent.cmd
```

색상이 깨지거나 로그 파일로 저장할 때는 끌 수 있습니다.

```powershell
$env:NO_COLOR=1
.\ml-agent.cmd
```

실제 editable input box가 필요한 경우에는 이미지형 콘솔 대신 Textual TUI를 실행합니다.

```powershell
.\ml-agent.cmd tui
```

키 조작:

- `Tab`: Plan / Build 전환
- `Enter`: 하단 입력 박스 내용 제출
- `Esc` 또는 `/exit`: 종료

## 10. 프롬프트 확인

프롬프트는 기본적으로 `prompt_templates.json`에 저장됩니다.

```powershell
.\ml-agent.cmd prompts
```

주요 프롬프트:

- `launch_mode_router`
- `beginner_wizard`
- `intermediate_analysis`
- `advanced_cli`
- `mlflow_registration_check`
- `job_template_draft`
- `closed_network_validation`
- `error_log_analysis`
- `retry_fix_from_error`

## 11. 에러 로그 기반 재수정

에러 로그는 기본적으로 `chat_errors/`에 저장됩니다.

수동 저장:

```powershell
.\ml-agent.cmd errors record "ModuleNotFoundError: No module named mlflow"
```

목록 확인:

```powershell
.\ml-agent.cmd errors list
```

분석:

```powershell
.\ml-agent.cmd errors analyze error-YYYYMMDDTHHMMSSZ
```

분석 결과에서 추천되는 명령은 보통 다음 흐름입니다.

```powershell
.\ml-agent.cmd analyze .\project
.\ml-agent.cmd validate .\project
.\ml-agent.cmd fix .\project --dry-run
```

## 12. Windows 10/11 실행 환경

이 POC의 기본 대상은 Windows 10/11 폐쇄망 환경입니다.
PowerShell 또는 CMD에서 `ml-agent.cmd`를 사용합니다.

```powershell
copy .env.example .env
.\ml-agent.cmd init
.\ml-agent.cmd config
.\ml-agent.cmd prompts
.\ml-agent.cmd
```

Linux/macOS에서 확인할 때만 아래 명령을 사용합니다.

```bash
cp .env.example .env
./ml-agent init
./ml-agent config
./ml-agent prompts
./ml-agent
```

## 13. 스킬 저장

스킬 저장 위치는 `.env`의 `SKILL_STORE_DIR` 값으로 정합니다.

```env
ENABLE_HARNESS_SKILLS=true
SKILL_STORE_DIR=skills
```

스킬은 다음 구조로 저장합니다.

```text
skills/
└── mlflow-registration-check/
    └── SKILL.md
```

기본 제공 스킬:

```text
skills/
├── agent-evaluation/
├── analyze-mlflow-chat-session/
├── analyze-mlflow-trace/
├── closed-network-validation/
├── error-log-repair/
├── instrumenting-with-mlflow-tracing/
├── job-template-draft/
├── mlflow-ai-gateway/
├── mlflow-experiment-tracking/
├── mlflow-model-registry-deployment/
├── mlflow-onboarding/
├── mlflow-prompt-management/
├── mlflow-prompt-optimization/
├── mlflow-registration-check/
├── querying-mlflow-metrics/
├── retrieving-mlflow-traces/
└── searching-mlflow-docs/
```

## 14. 문제 해결

Python을 찾지 못하는 경우:

```powershell
py -3 --version
python --version
```

한글이 깨지는 경우:

```powershell
chcp 65001
```

설정값이 반영되지 않는 경우:

```powershell
.\ml-agent.cmd config
```

스킬 폴더가 없는 경우:

```powershell
.\ml-agent.cmd init
```
