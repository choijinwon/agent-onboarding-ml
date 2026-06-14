# DeepAgents Source

This directory contains the DeepAgents source files included with this POC.

- Source root: `deep_agent/vendor/deepagents/deepagents-main`
- Libs root: `deep_agent/vendor/deepagents/deepagents-main/libs`
- Upstream reference: https://github.com/langchain-ai/deepagents

The AI ML onboarding POC reads `libs/**/pyproject.toml` from this directory first when running:

```bash
aiu deepagents
```

If a different archive must be compared, pass it explicitly:

```bash
aiu deepagents --source ~/Downloads/deepagents-main.zip
```
