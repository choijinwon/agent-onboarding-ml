---
name: closed-network-validation
description: 폐쇄망 AI ML 온보딩 환경에서 내부 Qwen, 의존성, 파일 권한, 로그 마스킹, 산출물 경로를 검증한다.
---

# Closed Network Validation

## When To Use

- Windows 10/11 또는 폐쇄망 Mac/Linux 환경에서 설치와 실행을 점검할 때
- 외부 다운로드 없이 로컬 샘플, 내부 Qwen, 로컬 MLflow file store로 검증해야 할 때
- 폴더 권한, 한글 인코딩, 절대 경로, 드래그앤드랍 경로 문제가 있을 때

## Checklist

- Qwen base URL과 model catalog 설정
- `deep_agent/skills`, `deep_agent/wiki`, `.aiu/sessions` 생성 여부
- Windows 절대 경로, UNC 경로, 공백 포함 경로 처리
- `MASK_SENSITIVE_LOGS` 활성화
- `file:./mlruns` 로컬 fallback
- FastAPI/uvicorn 미설치 시 친절한 안내

## Output

- 환경 점검 결과
- 차단/보완 항목
- 폐쇄망 설치 명령
- 재검증 명령

## Safety

- 외부 네트워크 호출을 가정하지 않는다.
- API key와 password는 출력하지 않는다.
