# AI ML 온보딩 퀵 가이드

Windows 10/11 폐쇄망 환경에서 AI ML 온보딩 POC를 빠르게 셋팅하는 절차입니다.

## 1. 준비물

- Windows 10 또는 Windows 11
- Python 3.10 이상
- 내부 Qwen OpenAI-compatible endpoint
- 이 저장소 파일

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
Qwen은 Agent가 분석과 안내에 사용하는 LLM이고, 등록 대상 ML 모델은 나중에 프로젝트 경로로 넘깁니다.

```env
QWEN_API_KEY=your-internal-qwen-key
QWEN_BASE_URL=http://xxx.xxx.xxx.xxx:포트/v1
QWEN_MODEL=qwen3.5
QWEN_MODELS=qwen3.5,gpt20,gamma
```

## 4. 초기 디렉터리 생성

```powershell
.\ml-agent.cmd init
```

생성되는 주요 경로:

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

확인할 값:

- `qwen_base_url`
- `qwen_model`
- `qwen_models`
- `multi_agent`
- `harness_skills`
- `skill_store_dir`

## 6. 실행

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
초급자 모드는 `Tab 1/10`부터 `Tab 10/10`까지 한 단계씩 보여주며, 좌측 단계 목록과 우측 현재 패널을 함께 표시합니다.
초급자 모드에서는 `apply` 명령어를 직접 입력하지 않고, 화면의 `적용하기 / 다시 보기 / 취소하기` 선택지로 진행합니다.

## 7. 자주 쓰는 명령

```powershell
.\ml-agent.cmd analyze .\project
.\ml-agent.cmd validate .\project
.\ml-agent.cmd fix .\project --dry-run
.\ml-agent.cmd apply .\project
.\ml-agent.cmd serve .\project --dry-run
.\ml-agent.cmd report .\project
.\ml-agent.cmd profile
.\ml-agent.cmd config
.\ml-agent.cmd prompts
.\ml-agent.cmd errors list
```

여기서 `.\project`는 등록하려는 ML 모델 프로젝트 경로입니다.
예를 들어 학습 코드, requirements, MLflow artifact, 모델 파일이 들어 있는 폴더를 넘깁니다.

JSON 출력:

```powershell
.\ml-agent.cmd validate .\project --json
.\ml-agent.cmd profile --json
.\ml-agent.cmd prompts --json
```

## 8. 프롬프트 확인

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

## 9. 에러 로그 기반 재수정

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

## 10. 스킬 저장

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

`SKILL.md` 예시:

```markdown
---
name: mlflow-registration-check
description: MLflow 등록 준비 상태를 점검한다
---

# MLflow Registration Check

- tracking URI 확인
- experiment/run logging 확인
- artifact 저장 경로 확인
- model registry 등록 조건 확인
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

## 11. Linux/macOS 실행

```bash
cp .env.example .env
./ml-agent init
./ml-agent config
./ml-agent prompts
./ml-agent
```

## 12. 문제 해결

Python을 찾지 못하는 경우:

```powershell
py -3 --version
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
