# ML Platform Console Assistant POC

폐쇄망 ML Platform 등록 지원을 위한 Launch Mode POC입니다.

처음 실행하면 사용자의 숙련도에 따라 세 가지 모드 중 하나를 선택합니다.

1. 초급자 모드: 단계별 Wizard 방식
2. 중급자 모드: Chat + Wizard 혼합
3. 고급자 모드: CLI Command 중심

## 실행

```bash
python3 ml_agent.py
```

또는 저장소 루트에서 실행 스크립트를 사용할 수 있습니다.

```bash
./ml-agent
```

## 고급자 명령

```bash
./ml-agent analyze ./project
./ml-agent validate ./project
./ml-agent fix ./project --dry-run
./ml-agent apply ./project
./ml-agent report ./project
./ml-agent chat
```

JSON 출력이 필요한 경우 `--json` 옵션을 사용할 수 있습니다.

```bash
./ml-agent validate ./project --json
```

## 모드 전환

실행 중 다음 명령으로 모드를 바꿀 수 있습니다.

```text
/mode beginner
/mode intermediate
/mode advanced
/모드 초급자
/모드 중급자
/모드 고급자
```

## 안전 규칙

- 기본 동작은 read-only scan입니다.
- 파일 수정 전에는 dry-run 또는 수정안 미리보기를 보여줍니다.
- 사용자 승인 없이 파일을 수정하지 않습니다.
- 삭제 작업은 하지 않습니다.
- 적용 후에는 재검증을 수행합니다.
