# Agent Skills

Deep Agent harness skills are stored here.

Each skill should live in its own directory with a `SKILL.md` file.

Example:

```text
skills/
└── mlflow-registration-check/
    └── SKILL.md
```

Default skills:

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

The MLflow-specific skills are adapted for this closed-network POC from `mlflow/skills` and `mlflow/mlflow`: tracing, trace analysis, chat session analysis, trace retrieval, agent evaluation, metrics querying, onboarding, documentation search, prompt management, prompt optimization, AI Gateway, experiment tracking, model registry, and deployment.
