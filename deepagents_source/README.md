# DeepAgents Source

This directory contains the DeepAgents source files imported from `deepagents-main.zip`.

- Source root: `deepagents_source/deepagents-main`
- Libs root: `deepagents_source/deepagents-main/libs`
- Upstream reference: https://github.com/langchain-ai/deepagents

The AI ML onboarding POC reads `libs/**/pyproject.toml` from this directory first when running:

```bash
./ml-agent deepagents
```

If a different archive is needed, pass it explicitly:

```bash
./ml-agent deepagents --source ~/Downloads/deepagents-main.zip
```
