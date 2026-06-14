---
name: closed-network-validation
description: 폐쇄망 AI ML 온보딩 환경에서 내부 Qwen, 의존성, 로그 마스킹, 산출물 경로를 검증한다.
---

# Closed Network Validation

## When To Use

- Windows 10/11 폐쇄망 환경 셋팅을 검증할 때
- 내부 Qwen endpoint 연결 정보를 점검할 때
- 외부 네트워크 의존성을 줄여야 할 때
- 로그와 리포트에 민감정보가 남지 않게 해야 할 때

## Checklist

- `QWEN_BASE_URL`이 내부 `/v1` endpoint인지 확인
- `QWEN_MODEL`과 `QWEN_MODELS`가 실제 사용 가능한 모델인지 확인
- `MASK_SENSITIVE_LOGS=true` 권장
- `REGISTRATION_PACKAGE_DIR`, `CHAT_ERROR_DIR`, `FIX_REPORT_DIR` 생성 여부 확인
- requirements와 Python 설치 파일의 폐쇄망 반입 가능 여부 확인
- MLflow tracking endpoint가 내부망에서 접근 가능한지 확인

## Output

- 폐쇄망 실행 가능 여부
- 누락된 환경 변수
- 외부 의존성 위험
- 로그 마스킹 상태
- 다음 점검 명령

## Safety

- API key는 출력하지 않는다.
- 설정 파일을 수정하기 전 preview를 제공한다.
- 검증 결과는 요약하고 상세 로그는 지정된 로그 디렉터리에 남긴다.
