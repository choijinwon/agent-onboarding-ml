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
├── mlflow-onboarding/
├── mlflow-registration-check/
├── querying-mlflow-metrics/
├── retrieving-mlflow-traces/
└── searching-mlflow-docs/
```

The MLflow-specific skills are adapted for this closed-network POC from the workflow categories in `mlflow/skills`: tracing, trace analysis, chat session analysis, trace retrieval, agent evaluation, metrics querying, onboarding, and documentation search.
