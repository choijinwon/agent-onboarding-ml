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
agent_workspace/
agent_workspace/registered/
skills/
plans/
goals/
sessions/
wiki_logs/
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

## 7. 자주 쓰는 명령

```powershell
.\ml-agent.cmd analyze .\project
.\ml-agent.cmd validate .\project
.\ml-agent.cmd fix .\project --dry-run
.\ml-agent.cmd apply .\project
.\ml-agent.cmd report .\project
.\ml-agent.cmd profile
.\ml-agent.cmd config
```

JSON 출력:

```powershell
.\ml-agent.cmd validate .\project --json
.\ml-agent.cmd profile --json
```

## 8. 스킬 저장

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

## 9. Linux/macOS 실행

```bash
cp .env.example .env
./ml-agent init
./ml-agent config
./ml-agent
```

## 10. 문제 해결

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
