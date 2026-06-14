"""Environment configuration and runtime directory setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ENV = {
    "QWEN_API_KEY": "your-internal-qwen-key",
    "QWEN_BASE_URL": "http://xxx.xxx.xxx.xxx:port/v1",
    "QWEN_MODEL": "qwen3.5",
    "QWEN_MODELS": "qwen3.5,gpt20,gamma",
    "ENABLE_MULTI_AGENT": "true",
    "ENABLE_HARNESS_SKILLS": "true",
    "ENABLE_RICH_CONSOLE": "true",
    "ENABLE_TUI_BACKGROUND": "false",
    "ENABLE_TUI_INPUT_PANEL": "true",
    "PROMPT_STORE_PATH": "prompt_templates.json",
    "CHAT_ERROR_DIR": "chat_errors",
    "MASK_SENSITIVE_LOGS": "true",
    "REGISTRATION_PACKAGE_DIR": "registration_packages",
    "FIX_REPORT_DIR": "fix_reports",
    "SKILL_STORE_DIR": "skills",
}

DIRECTORY_KEYS = (
    "CHAT_ERROR_DIR",
    "REGISTRATION_PACKAGE_DIR",
    "FIX_REPORT_DIR",
    "SKILL_STORE_DIR",
)


DEFAULT_SKILLS = {
    "mlflow-registration-check": """---
name: mlflow-registration-check
description: MLflow 등록 준비 상태를 점검하고 누락된 logging, artifact, registry 조건을 설명한다.
---

# MLflow Registration Check

## When To Use

- 사용자가 MLflow 설정 검증을 요청할 때
- 프로젝트가 모델 등록 대상인지 판단해야 할 때
- `mlflow`, `tracking_uri`, `experiment`, `artifact`, `registry` 관련 오류가 있을 때

## Checklist

- MLflow tracking URI 설정 확인
- experiment/run 생성 흐름 확인
- params, metrics, artifacts logging 확인
- model artifact 저장 경로 확인
- requirements와 Python version 확인
- model registry 등록에 필요한 이름, stage, signature 후보 확인

## Output

- 등록 가능 여부 요약
- 누락 항목 목록
- 영향도
- dry-run 수정 제안
- 재검증 명령

## Safety

- 기본은 read-only scan이다.
- 파일 수정 전에는 반드시 preview를 먼저 제공한다.
- API key, token, password는 출력하지 않는다.
""",
    "instrumenting-with-mlflow-tracing": """---
name: instrumenting-with-mlflow-tracing
description: Python/TypeScript GenAI 앱에 MLflow Tracing을 추가하기 위한 점검과 수정안을 만든다.
---

# Instrumenting With MLflow Tracing

## When To Use

- OpenAI, LangChain, LangGraph, LiteLLM, Anthropic, custom agent 코드에 tracing을 붙일 때
- LLM 호출, tool call, retriever, chain 실행 흐름을 MLflow trace로 남기고 싶을 때
- 폐쇄망 POC에서 tracing 적용 전 변경 범위를 검토해야 할 때

## Workflow

- 프레임워크와 LLM 호출 지점을 찾는다.
- MLflow tracking URI와 experiment 설정 위치를 확인한다.
- 가능한 경우 autolog 또는 trace decorator 방식의 최소 수정안을 제안한다.
- tracing 적용 후 생성될 span, inputs, outputs, metadata 범위를 설명한다.
- 수정 전 dry-run preview를 보여주고 승인 후에만 적용한다.

## Safety

- prompt, token, API key, 개인정보가 trace에 저장될 수 있는지 확인한다.
- 민감정보 마스킹 설정이 없으면 먼저 보완 항목으로 표시한다.
- 운영 코드 변경 전 로컬 샘플 요청으로 검증한다.
""",
    "analyze-mlflow-trace": """---
name: analyze-mlflow-trace
description: MLflow trace/span 정보를 바탕으로 실패 원인과 수정 후보를 분석한다.
---

# Analyze MLflow Trace

## When To Use

- trace ID, span ID, 실패한 LLM 실행 로그가 있을 때
- tool call 실패, latency 증가, 빈 응답, 잘못된 답변 원인을 찾을 때
- 오류 로그 기반 재수정이 필요할 때

## Workflow

- trace 상태, span tree, error span, latency가 큰 span을 우선 확인한다.
- 입력, 출력, metadata, assessment가 있으면 실패 지점을 요약한다.
- 코드 위치와 requirements, 환경변수를 연결해 원인 후보를 만든다.
- 수정안은 evidence, impact, dry-run preview 순서로 제시한다.

## Output

- 실패 지점 요약
- 원인 후보
- 관련 파일 또는 설정
- 재현/재검증 명령
""",
    "analyze-mlflow-chat-session": """---
name: analyze-mlflow-chat-session
description: MLflow에 남은 multi-turn chat session을 재구성하고 대화 실패 지점을 찾는다.
---

# Analyze MLflow Chat Session

## When To Use

- 여러 턴의 agent/chat 실행 중 어느 시점에서 품질이 깨졌는지 확인할 때
- session ID, user ID, trace 묶음 기반으로 대화를 분석할 때
- 응답 누락, tool loop, context 손실을 진단할 때

## Workflow

- session 단위로 message, trace, assessment를 시간순으로 정렬한다.
- 실패 전후의 사용자 요청, tool call, model output 차이를 비교한다.
- context overflow, 잘못된 routing, 누락된 memory, tool error 여부를 확인한다.
- 다시 실행할 최소 테스트 시나리오를 제안한다.

## Output

- 대화 흐름 요약
- 실패 턴
- 원인 후보
- 수정 및 재검증 계획
""",
    "retrieving-mlflow-traces": """---
name: retrieving-mlflow-traces
description: MLflow trace를 상태, 시간, session, user, metadata 기준으로 검색하는 방법을 안내한다.
---

# Retrieving MLflow Traces

## When To Use

- 실패 trace를 찾아야 할 때
- 특정 session/user/time range의 실행 기록을 모아야 할 때
- batch 검증이나 운영 장애 분석용 trace 목록이 필요할 때

## Query Checklist

- tracking URI와 experiment를 먼저 확인한다.
- status, request time, session ID, run ID, user metadata를 기준으로 좁힌다.
- 폐쇄망에서는 외부 서비스 호출 없이 로컬/내부 MLflow endpoint만 사용한다.
- 결과는 trace ID, status, latency, 주요 error message 중심으로 요약한다.

## Output

- 검색 조건
- 후보 trace 목록
- 다음 분석에 넘길 trace ID
- 재현 명령 또는 후속 skill 추천
""",
    "agent-evaluation": """---
name: agent-evaluation
description: MLflow Evaluation으로 agent 품질 평가 데이터셋, scorer, 실행, 결과 분석 흐름을 만든다.
---

# Agent Evaluation

## When To Use

- agent 또는 모델 등록 전 품질 기준을 만들 때
- accuracy, relevance, groundedness, latency 같은 평가 지표가 필요할 때
- 수정 전후 품질이 좋아졌는지 비교해야 할 때

## Workflow

- 평가 목적과 입력/정답/참조 데이터 형태를 정한다.
- dataset 후보를 만들거나 기존 evaluation set을 찾는다.
- scorer와 threshold를 선택한다.
- evaluation 실행 계획과 결과 리포트 위치를 정한다.
- 실패 케이스를 수정 후보로 연결한다.

## Safety

- 평가 데이터에 민감정보가 있으면 마스킹하거나 샘플링한다.
- 폐쇄망에서 사용 가능한 judge model 또는 rule-based scorer를 우선 사용한다.
""",
    "querying-mlflow-metrics": """---
name: querying-mlflow-metrics
description: MLflow metrics를 조회해 token usage, latency, error rate, 품질 추세를 분석한다.
---

# Querying MLflow Metrics

## When To Use

- 최근 실행의 latency, token, cost, error trend를 보고 싶을 때
- 모델/agent 수정 전후 지표를 비교할 때
- 등록 후보 모델의 운영 안정성을 요약해야 할 때

## Workflow

- experiment/run 범위와 시간 구간을 정한다.
- metric key, tag, model name, dataset 기준으로 집계한다.
- 이상치와 regression을 찾는다.
- 표 또는 JSON으로 결과를 요약한다.

## Output

- 조회 범위
- 핵심 지표 요약
- 이상 징후
- 다음 조치
""",
    "mlflow-onboarding": """---
name: mlflow-onboarding
description: 신규 사용자가 MLflow Tracking, Evaluation, Registry, Tracing 중 필요한 경로를 선택하도록 안내한다.
---

# MLflow Onboarding

## When To Use

- 사용자가 MLflow에 익숙하지 않을 때
- 전통 ML 모델 등록과 GenAI/agent observability 중 무엇부터 해야 할지 모를 때
- 초급자 모드에서 어려운 용어를 쉽게 풀어야 할 때

## Workflow

- 사용 목적을 먼저 분류한다: 모델 등록, 실험 추적, agent tracing, 평가, 운영 분석.
- 필요한 최소 파일과 설정을 안내한다.
- read-only scan 후 누락 항목을 쉬운 말로 설명한다.
- 수정은 preview와 승인 이후에만 진행한다.

## Output

- 사용자 상황 요약
- 추천 시작 경로
- 다음 단계 선택지
- 필요한 파일/설정 목록
""",
    "searching-mlflow-docs": """---
name: searching-mlflow-docs
description: MLflow 공식 문서 또는 반입된 문서 인덱스를 기준으로 필요한 내용을 찾는 절차를 안내한다.
---

# Searching MLflow Docs

## When To Use

- MLflow API, tracing, evaluation, registry 사용법을 확인해야 할 때
- 폐쇄망에 반입된 문서나 llms.txt 인덱스를 기준으로 답해야 할 때
- 버전별 동작 차이를 확인해야 할 때

## Workflow

- 현재 MLflow 버전과 사용 기능을 먼저 확인한다.
- 공식 문서 또는 내부 반입 문서의 경로를 우선 사용한다.
- 찾은 내용은 짧게 요약하고 적용 파일/명령으로 연결한다.
- 문서 근거가 없으면 추정이라고 표시한다.

## Output

- 검색 키워드
- 문서 위치
- 요약
- 적용 예시
""",
    "mlflow-prompt-management": """---
name: mlflow-prompt-management
description: MLflow Prompt Management로 프롬프트 버전, lineage, 평가 연결 흐름을 점검한다.
---

# MLflow Prompt Management

## When To Use

- 프롬프트를 코드에 직접 박아두지 않고 버전 관리하고 싶을 때
- agent 또는 LLM 앱의 prompt 변경 이력을 추적해야 할 때
- 평가 결과와 prompt 버전을 연결해 회귀를 확인해야 할 때

## Workflow

- 프롬프트가 저장된 파일, 템플릿, 환경변수, 코드 위치를 찾는다.
- prompt name, version, tag, owner, 사용 모델을 정리한다.
- 평가 데이터셋과 prompt 버전 연결 기준을 제안한다.
- 폐쇄망에서는 내부 prompt registry 또는 파일 기반 prompt store를 우선 사용한다.

## Output

- prompt inventory
- 버전 관리 후보
- 평가 연결 계획
- dry-run 수정안
""",
    "mlflow-prompt-optimization": """---
name: mlflow-prompt-optimization
description: MLflow Prompt Optimization 흐름을 참고해 프롬프트 개선 실험과 평가 루프를 설계한다.
---

# MLflow Prompt Optimization

## When To Use

- 프롬프트를 수정했을 때 실제 품질이 좋아졌는지 검증해야 할 때
- 여러 prompt 후보를 평가하고 최고안을 고르고 싶을 때
- agent 응답 품질 회귀를 방지하고 싶을 때

## Workflow

- 목표 metric과 평가 데이터셋을 먼저 정한다.
- baseline prompt와 후보 prompt를 분리한다.
- judge model, rule-based scorer, human review 중 폐쇄망에서 가능한 방식을 선택한다.
- 결과는 prompt version, score, 실패 케이스, 다음 실험으로 정리한다.

## Safety

- 민감정보가 포함된 prompt 또는 평가 데이터는 마스킹한다.
- 자동 개선안은 운영 반영 전 반드시 dry-run과 평가 리포트를 거친다.
""",
    "mlflow-ai-gateway": """---
name: mlflow-ai-gateway
description: MLflow AI Gateway 개념을 참고해 내부 LLM 라우팅, 비용, 접근 제어, fallback 설정을 점검한다.
---

# MLflow AI Gateway

## When To Use

- 여러 LLM provider 또는 내부 모델을 하나의 OpenAI 호환 endpoint로 쓰고 싶을 때
- Qwen, GPT 계열, 내부 모델 라우팅과 fallback을 관리해야 할 때
- 비용, rate limit, credential 노출을 통제해야 할 때

## Checklist

- base URL, model catalog, route name, fallback 순서를 확인한다.
- API key가 코드나 로그에 노출되지 않는지 확인한다.
- rate limit, timeout, retry, streaming 옵션을 정리한다.
- 폐쇄망에서는 외부 provider 호출을 차단하고 내부 endpoint만 허용한다.

## Output

- gateway readiness 요약
- model routing 표
- 보안/비용 위험
- 재검증 명령
""",
    "mlflow-experiment-tracking": """---
name: mlflow-experiment-tracking
description: MLflow Experiment Tracking으로 params, metrics, artifacts, datasets 기록 상태를 점검한다.
---

# MLflow Experiment Tracking

## When To Use

- 학습 코드가 실험 재현에 필요한 정보를 충분히 남기는지 확인할 때
- params, metrics, artifacts, datasets, tags 누락을 점검할 때
- 등록 전 학습 실행 이력을 리포트로 정리해야 할 때

## Checklist

- tracking URI와 experiment name 설정
- run lifecycle 시작/종료 위치
- hyperparameter와 runtime argument logging
- metric key와 step 기록
- model artifact, dataset, source version 기록

## Output

- tracking readiness
- 누락 logging 항목
- 코드 수정 preview
- validate/report 명령
""",
    "mlflow-model-registry-deployment": """---
name: mlflow-model-registry-deployment
description: MLflow Model Registry와 Deployment 흐름을 기준으로 모델 등록, 버전, 배포 준비 상태를 점검한다.
---

# MLflow Model Registry And Deployment

## When To Use

- 모델 파일을 registry에 등록하거나 버전 관리해야 할 때
- batch 또는 real-time serving 배포 준비 상태를 확인할 때
- Docker, Kubernetes, SageMaker, Azure ML 같은 배포 목표를 정리해야 할 때

## Checklist

- model name, version, alias 또는 stage 정책
- signature, input example, dependency, Python version
- artifact URI와 registry URI
- local serving health/predict 테스트
- rollback과 promotion 기준

## Output

- registry readiness
- deployment target 후보
- serving 검증 결과
- 등록 패키지에 포함할 파일 목록
""",
}


@dataclass(frozen=True)
class AppConfig:
    values: dict[str, str]
    root_dir: Path

    @classmethod
    def load(cls, env_file: str = ".env", root_dir: Path | None = None) -> "AppConfig":
        root = Path.cwd() if root_dir is None else root_dir
        values = dict(DEFAULT_ENV)
        values.update(_read_env_file(root / env_file))
        for key in DEFAULT_ENV:
            if key in os.environ:
                values[key] = os.environ[key]
        return cls(values=values, root_dir=root)

    def get(self, key: str) -> str:
        return self.values.get(key, "")

    def get_bool(self, key: str) -> bool:
        return self.get(key).strip().lower() in {"1", "true", "yes", "on"}

    def get_int(self, key: str) -> int:
        return int(self.get(key))

    def runtime_directories(self) -> list[Path]:
        dirs = [self.root_dir / self.get(key) for key in DIRECTORY_KEYS if self.get(key)]
        if self.get_bool("ENABLE_HARNESS_SKILLS"):
            dirs.append(self.skill_store_dir())
        return _dedupe_paths(dirs)

    def skill_store_dir(self) -> Path:
        return self.root_dir / self.get("SKILL_STORE_DIR")


def ensure_runtime_layout(config: AppConfig) -> list[Path]:
    created_or_existing = []
    for directory in config.runtime_directories():
        directory.mkdir(parents=True, exist_ok=True)
        created_or_existing.append(directory)
    _ensure_skill_readme(config.skill_store_dir())
    _ensure_default_skills(config.skill_store_dir())
    return created_or_existing


def format_config_summary(config: AppConfig) -> str:
    directories = "\n".join(f"- {path}" for path in config.runtime_directories())
    models = ", ".join(model.strip() for model in config.get("QWEN_MODELS").split(","))
    return (
        "Environment Config\n"
        f"- qwen_base_url={config.get('QWEN_BASE_URL')}\n"
        f"- qwen_model={config.get('QWEN_MODEL')}\n"
        f"- qwen_models={models}\n"
        f"- multi_agent={config.get_bool('ENABLE_MULTI_AGENT')}\n"
        f"- harness_skills={config.get_bool('ENABLE_HARNESS_SKILLS')}\n"
        f"- rich_console={config.get_bool('ENABLE_RICH_CONSOLE')}\n"
        f"- tui_background={config.get_bool('ENABLE_TUI_BACKGROUND')}\n"
        f"- tui_input_panel={config.get_bool('ENABLE_TUI_INPUT_PANEL')}\n"
        f"- skill_store_dir={config.skill_store_dir()}\n\n"
        "runtime_directories:\n"
        f"{directories}"
    )


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _ensure_skill_readme(skill_dir: Path) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    readme = skill_dir / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "# Agent Skills\n\n"
        "Deep Agent harness skills are stored here.\n\n"
        "Each skill should live in its own directory with a `SKILL.md` file.\n",
        encoding="utf-8",
    )


def _ensure_default_skills(skill_dir: Path) -> None:
    for name, content in DEFAULT_SKILLS.items():
        skill_file = skill_dir / name / "SKILL.md"
        if skill_file.exists():
            continue
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(content.strip() + "\n", encoding="utf-8")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    deduped = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
