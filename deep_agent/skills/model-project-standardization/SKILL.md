---
name: model-project-standardization
description: 사용자가 가져온 TensorFlow, PyTorch, sklearn, ONNX, HuggingFace 스타일 모델을 AI Studio 표준 폴더 구조로 정리한다.
---

# Model Project Standardization

## When To Use

- 사용자가 기존 모델 파일이나 학습 코드를 가져와 테스트할 때
- `run_model.py`, `aiu_custom/predict.py`, `config.json`, `input_example.json`, `requirements.txt`가 누락됐을 때
- pre-trained artifact 등록과 직접 학습 코드 생성 중 흐름을 선택해야 할 때

## Target Structure

- `aiu_custom/`
- `config/`
- `saved_model/`
- `run_model.py`
- `predict.py` 또는 `aiu_custom/predict.py`
- `requirements.txt`
- `input_example.json`
- `ai_studio.env`

## Workflow

- framework와 artifact 확장자를 먼저 식별한다.
- 기존 학습 파일은 바로 덮어쓰지 않고 review_required로 분류한다.
- 누락된 보완 파일은 표준 템플릿으로 생성 가능한 safe 항목으로 분류한다.
- 모델 artifact 자체는 삭제하거나 교체하지 않는다.

## Output

- 감지된 framework
- 현재 구조와 표준 구조 차이
- 생성/보완할 파일 목록
- 등록 전 검증 명령
