import json
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from deep_agent import cli as ml_agent
from deep_agent.app_config import AppConfig, DEFAULT_SKILLS, ensure_runtime_layout
from deep_agent.profile import build_ml_platform_profile, format_profile
from deep_agent.libs import deepagents_libs_as_dict
from deep_agent.runtime import DeepAgentsRunResult, DeepAgentsRuntime, build_deepagents_system_prompt, extract_deepagents_content
from deep_agent.stores.chat_session_store import append_chat_session_event, mask_sensitive_text
from deep_agent.stores.error_log_store import analyze_error_log, list_error_logs, save_error_log
from deep_agent.stores.prompt_store import export_prompt_templates_to_wiki, load_prompt_templates
from deep_agent.qwen_chat import QwenChatConfig, chat_with_qwen
from deep_agent.cli import (
    ConsoleAssistant,
    LAUNCH_SCREEN,
    MODE_ADVANCED,
    MODE_BEGINNER,
    MODE_INTERMEDIATE,
    analyze_project,
    build_beginner_intro,
    build_beginner_step_tabs,
    build_beginner_wizard,
    build_parser,
    create_heavy_model_sample,
    create_large_model_samples,
    format_beginner_tab,
    handle_advanced_input,
    handle_intermediate_request,
    list_existing_sample_projects,
    parse_mode,
    parse_mode_command,
    resolve_existing_sample_project,
    resolve_beginner_project_input,
    sample_projects_root,
)
from deep_agent.tui import (
    BeginnerTuiController,
    command_placeholder_for_mode,
    format_agent_mode_selector,
    format_model_choices,
    is_fix_request,
    missing_textual_message,
    model_selection_placeholder,
    normalize_input_path,
    parse_agent_mode_command,
    parse_model_command,
    path_candidates_from_input,
    strip_path_command,
)


class QwenChatTest(unittest.TestCase):
    def test_qwen_config_treats_sample_values_as_unconfigured(self):
        config = QwenChatConfig(
            api_key="your-internal-qwen-key",
            base_url="http://xxx.xxx.xxx.xxx:port/v1",
            model="qwen3.6",
        )

        self.assertFalse(config.is_configured())
        self.assertIn("Qwen 3.6 연결 설정", chat_with_qwen("안녕", config=config))

    def test_qwen_endpoint_uses_openai_compatible_chat_completions(self):
        config = QwenChatConfig(api_key="secret", base_url="http://qwen.local/v1", model="qwen3.6")

        self.assertTrue(config.is_configured())
        self.assertEqual(config.endpoint(), "http://qwen.local/v1/chat/completions")


class DeepAgentsRuntimeTest(unittest.TestCase):
    def test_deepagents_runtime_reports_missing_qwen_config(self):
        config = AppConfig(values={
            "QWEN_API_KEY": "your-internal-qwen-key",
            "QWEN_BASE_URL": "http://xxx.xxx.xxx.xxx:port/v1",
            "QWEN_MODEL": "qwen3.6",
            "QWEN_MODELS": "qwen3.6,qwen3.5,gpt20,gamma",
        }, root_dir=Path.cwd())

        result = DeepAgentsRuntime(config).invoke("분석해줘")

        self.assertFalse(result.used_deepagents)
        self.assertEqual(result.error, "qwen_not_configured")
        self.assertIn("QWEN_API_KEY", result.content)

    def test_deepagents_prompt_enforces_plan_and_build_policies(self):
        plan_prompt = build_deepagents_system_prompt("/tmp/model", "Plan")
        build_prompt = build_deepagents_system_prompt("/tmp/model", "Build")
        autofix_prompt = build_deepagents_system_prompt("/tmp/model", "AutoFix")

        self.assertIn("Plan mode: do not modify files", plan_prompt)
        self.assertIn("preview_ml_fixes", plan_prompt)
        self.assertIn("Build mode", build_prompt)
        self.assertIn("apply_ml_fixes", build_prompt)
        self.assertIn("AutoFix mode", autofix_prompt)
        self.assertIn("apply_ml_fixes automatically", autofix_prompt)

    def test_extract_deepagents_content_reads_last_message(self):
        class Message:
            content = "마지막 응답"

        self.assertEqual(extract_deepagents_content({"messages": [Message()]}), "마지막 응답")
        self.assertEqual(extract_deepagents_content({"output": "출력"}), "출력")


class ModeParsingTest(unittest.TestCase):
    def test_parse_numeric_modes(self):
        self.assertEqual(parse_mode("1"), MODE_BEGINNER)
        self.assertEqual(parse_mode("2"), MODE_INTERMEDIATE)
        self.assertEqual(parse_mode("3"), MODE_ADVANCED)

    def test_parse_korean_mode_commands(self):
        self.assertEqual(parse_mode_command("/모드 초급자"), MODE_BEGINNER)
        self.assertEqual(parse_mode_command("/모드 중급자"), MODE_INTERMEDIATE)
        self.assertEqual(parse_mode_command("/모드 고급자"), MODE_ADVANCED)

    def test_parse_english_mode_commands(self):
        self.assertEqual(parse_mode_command("/mode beginner"), MODE_BEGINNER)
        self.assertEqual(parse_mode_command("/mode intermediate"), MODE_INTERMEDIATE)
        self.assertEqual(parse_mode_command("/mode advanced"), MODE_ADVANCED)


class BeginnerWizardTest(unittest.TestCase):
    def test_beginner_wizard_builds_separate_step_tabs(self):
        steps = build_beginner_step_tabs("/workspace/my-model")
        first_tab = format_beginner_tab(0, len(steps), steps[0])

        self.assertEqual(len(steps), 10)
        self.assertIn("Step 1. 프로젝트 선택", steps[0])
        self.assertIn("Step 2. 프로젝트 자동 스캔", steps[1])
        self.assertNotIn("Step 2. 프로젝트 자동 스캔", steps[0])
        self.assertIn("[Tab 1]", first_tab)
        self.assertIn("Current: Tab 1/10", first_tab)
        self.assertIn("STEPS", first_tab)
        self.assertIn("CURRENT PANEL", first_tab)
        self.assertIn("AI ML Onboarding Console", first_tab)
        self.assertIn("User request", first_tab)
        self.assertIn("Plan*  Build", first_tab)
        self.assertIn("Active agent: Plan read-only", first_tab)
        self.assertIn("esc interrupt", first_tab)
        self.assertIn("> 01 프로젝트 선택", first_tab)
        self.assertIn("Enter=다음", first_tab)

        apply_tab = format_beginner_tab(6, len(steps), steps[6])
        self.assertIn("Plan  Build*", apply_tab)
        self.assertIn("Active agent: Build approval", apply_tab)

    def test_launch_screen_uses_terminal_console_layout(self):
        self.assertIn("AI ML Onboarding Console", LAUNCH_SCREEN)
        self.assertIn("# Launch workflow", LAUNCH_SCREEN)
        self.assertIn("Plan(read-only)", LAUNCH_SCREEN)
        self.assertIn("esc interrupt", LAUNCH_SCREEN)

    def test_forced_rich_tui_uses_deepagents_layout(self):
        previous_force = os.environ.get("FORCE_COLOR")
        previous_no_color = os.environ.get("NO_COLOR")
        try:
            os.environ["FORCE_COLOR"] = "1"
            os.environ.pop("NO_COLOR", None)
            ml_agent._RICH_CONSOLE_ENABLED = None
            steps = build_beginner_step_tabs("/workspace/my-model")
            output = format_beginner_tab(0, len(steps), steps[0])

            self.assertIn("\033[", output)
            self.assertIn("DeepAgents", output)
            self.assertIn("ctrl+p commands", output)
            self.assertNotIn("+====", output)
            self.assertNotIn("\033[48;2;", output)
            self.assertIn("\033[48;5;235m", output)
        finally:
            if previous_force is None:
                os.environ.pop("FORCE_COLOR", None)
            else:
                os.environ["FORCE_COLOR"] = previous_force
            if previous_no_color is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = previous_no_color
            ml_agent._RICH_CONSOLE_ENABLED = None

    def test_beginner_console_advances_one_tab_at_a_time(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow==2.17.0\n")
            (root / "train.py").write_text("import mlflow\n")
            (root / "model.onnx").write_text("sample")
            inputs = iter([str(root), "", "종료"])
            outputs: list[str] = []
            clears: list[str] = []
            assistant = ConsoleAssistant(
                input_fn=lambda prompt: next(inputs),
                output_fn=outputs.append,
                clear_fn=lambda: clears.append("clear"),
            )

            assistant.run_beginner_mode()

            self.assertGreaterEqual(len(clears), 2)
            self.assertIn("Current: Tab 1/10", outputs[0])
            self.assertIn("Step 1. 프로젝트 선택", outputs[0])
            self.assertNotIn("Step 2. 프로젝트 자동 스캔", outputs[0])
            self.assertIn("Current: Tab 2/10", outputs[1])
            self.assertIn("Step 2. 프로젝트 자동 스캔", outputs[1])
            self.assertIn("초급자 Wizard를 종료합니다", outputs[-1])

    def test_beginner_console_applies_preview_after_step6_approval(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            inputs = iter([str(root), "", "", "", "", "", "1", "종료"])
            outputs: list[str] = []
            prompts: list[str] = []
            assistant = ConsoleAssistant(
                input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
                output_fn=outputs.append,
                clear_fn=lambda: None,
            )

            assistant.run_beginner_mode()

            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertIn("import mlflow", train.read_text())
            self.assertIn("선택 번호 > ", prompts)
            self.assertTrue(any("1번 적용을 승인했습니다" in output for output in outputs))
            step7_output = next(output for output in outputs if "Current: Tab 7/10" in output)
            self.assertIn("적용 완료: 2개", step7_output)
            self.assertIn("등록 상태: 등록 가능", step7_output)

    def test_beginner_console_step4_choice_one_moves_to_preview(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("tensorflow==2.17.0\n")
            (root / "train.py").write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            inputs = iter([str(root), "", "", "", "1", "종료"])
            outputs: list[str] = []
            prompts: list[str] = []
            assistant = ConsoleAssistant(
                input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
                output_fn=outputs.append,
                clear_fn=lambda: None,
            )

            assistant.run_beginner_mode()

            self.assertIn("선택 번호 > ", prompts)
            self.assertTrue(any("Current: Tab 5/10" in output for output in outputs))
            self.assertFalse(any("Current: Tab 1/10" in output for output in outputs[4:]))

    def test_beginner_console_step6_uses_number_selection_for_review(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("tensorflow==2.17.0\n")
            (root / "train.py").write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            inputs = iter([str(root), "", "", "", "", "", "2", "종료"])
            outputs: list[str] = []
            prompts: list[str] = []
            assistant = ConsoleAssistant(
                input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
                output_fn=outputs.append,
                clear_fn=lambda: None,
            )

            assistant.run_beginner_mode()

            self.assertIn("선택 번호 > ", prompts)
            self.assertEqual((root / "requirements.txt").read_text(), "tensorflow==2.17.0\n")
            self.assertTrue(any("Current: Tab 5/10" in output for output in outputs[6:]))

    def test_beginner_wizard_is_read_only_first(self):
        output = build_beginner_wizard("/workspace/my-model")

        self.assertIn("project-scanner", output)
        self.assertIn("read-only scan", output)
        self.assertIn("파일은 수정하지 않았습니다", output)
        self.assertIn("등록 상태", output)
        self.assertIn("문제 수", output)
        self.assertIn("다음 조치", output)
        self.assertIn("수정안 미리보기", output)
        self.assertIn("파일은 수정하지 않았습니다", output)
        self.assertIn("번호만 입력", output)
        self.assertIn("다시 보기", output)
        self.assertIn("취소하기", output)
        self.assertIn("승인 전 상태", output)
        self.assertIn("삭제 작업은 수행하지 않습니다", output)
        self.assertIn("재검증", output)
        self.assertIn("Step 9. 로컬 서빙 테스트", output)
        self.assertIn("Step 10. 분석 리포트 생성", output)

    def test_beginner_wizard_reports_registration_ready_project(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\nmlflow==2.17.0\n")
            (root / "train.py").write_text(
                "import mlflow\n\n"
                "if __name__ == \"__main__\":\n"
                "    with mlflow.start_run():\n"
                "        mlflow.log_metric('accuracy', 0.9)\n"
            )
            (root / "model").mkdir()
            (root / "model" / "model.pkl").write_text("sample")

            output = build_beginner_wizard(str(root))

            self.assertIn("등록 상태: 등록 가능", output)
            self.assertIn("Step 2. 프로젝트 자동 스캔", output)
            self.assertIn("파일 수:", output)
            self.assertIn("모델 artifact 후보: 1개", output)
            self.assertIn("[통과] MLflow 의존성", output)
            self.assertIn("[통과] 모델 산출물", output)
            self.assertIn("Job Template 초안 준비: 가능", output)
            self.assertIn("문제 수: 0개", output)
            self.assertIn("상태: 준비 가능", output)
            self.assertIn("http://127.0.0.1:8000/health", output)
            self.assertIn("최종 결과 요약", output)
            self.assertIn("리포트 저장", output)

    def test_beginner_wizard_lists_step4_issue_details(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 4. 문제 목록 확인", output)
            self.assertIn("필수 확인: 0개", output)
            self.assertIn("보완 권장:", output)
            self.assertIn("Agent 수정 가능:", output)
            self.assertIn("주요 문제:", output)
            self.assertIn("| 번호", output)
            self.assertIn("| 구분", output)
            self.assertIn("| 다음 조치", output)
            self.assertIn("MLflow 패키지 누락", output)
            self.assertIn("requirements.txt", output)
            self.assertIn("가능", output)
            self.assertIn("1. 수정안 미리보기로 이동", output)

    def test_beginner_wizard_shows_step5_dry_run_preview(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 5. 수정안 미리보기", output)
            self.assertIn("dry-run 결과입니다", output)
            self.assertIn("MLflow 의존성 추가", output)
            self.assertIn("+ mlflow", output)
            self.assertIn("MLflow 기록 코드 추가", output)
            self.assertIn("적용하려면 다음 단계에서 1번을 선택합니다", output)

    def test_beginner_wizard_shows_step6_approval_choices(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 6. 사용자 승인", output)
            self.assertIn("적용 범위: Step 5에 표시된 미리보기 항목으로 제한됩니다.", output)
            self.assertIn("선택 방법: 번호만 입력합니다.", output)
            self.assertIn("1. 적용하기 (선택 가능)", output)
            self.assertIn("결과: 파일 수정 있음", output)
            self.assertIn("2. 다시 보기 (선택 가능)", output)
            self.assertIn("3. 취소하기 (선택 가능)", output)

    def test_beginner_wizard_disables_apply_without_preview(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow==2.17.0\n")
            (root / "train.py").write_text(
                "import mlflow\n"
                "if __name__ == \"__main__\":\n"
                "    mlflow.log_param('x', 1)\n"
            )
            (root / "model.onnx").write_text("sample")

            output = build_beginner_wizard(str(root))

            self.assertIn("1. 적용하기 (선택 불가)", output)
            self.assertIn("1번은 수정안이 있을 때만 선택할 수 있습니다.", output)

    def test_beginner_wizard_shows_step7_apply_scope(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 7. 파일 생성 또는 수정", output)
            self.assertIn("Step 6에서 1번 승인 후에만", output)
            self.assertIn("삭제 작업은 수행하지 않습니다.", output)
            self.assertIn("requirements.txt: MLflow 의존성 추가", output)
            self.assertIn("train.py: MLflow 기록 코드 추가", output)

    def test_beginner_wizard_shows_step9_local_serving(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow==2.17.0\n")
            (root / "train.py").write_text("import mlflow\n")
            (root / "model.onnx").write_text("sample")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 9. 로컬 서빙 테스트", output)
            self.assertIn("상태: 준비 가능", output)
            self.assertIn("FastAPI 기본 서버", output)
            self.assertIn("ml-agent serve", output)
            self.assertIn("/predict", output)

    def test_beginner_wizard_shows_step10_report_summary(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow==2.17.0\n")
            (root / "train.py").write_text("import mlflow\n")
            (root / "model.onnx").write_text("sample")

            steps = build_beginner_step_tabs(str(root))
            output = steps[9]

            self.assertIn("Step 10. 분석 리포트 생성", output)
            self.assertIn("최종 결과 요약", output)
            self.assertIn("등록 상태: 등록 가능", output)
            self.assertIn("로컬 서빙: 준비 가능", output)
            self.assertIn("남은 문제: 없음", output)
            self.assertIn("저장 경로:", output)
            self.assertIn("ml-agent-report.json", output)
            self.assertIn("저장 명령: ml-agent report", output)

    def test_heavy_model_sample_can_be_selected_from_step1(self):
        with TemporaryDirectory() as tmpdir:
            sample = create_heavy_model_sample(Path(tmpdir) / "heavy-model", artifact_size_bytes=1024)
            analysis = analyze_project(str(sample))
            output = build_beginner_wizard(str(sample))

            self.assertEqual((sample / "model" / "heavy-model.onnx").stat().st_size, 1024)
            self.assertEqual(analysis.registration_status, "등록 가능")
            self.assertIn("등록 상태: 등록 가능", output)
            self.assertIn("heavy-model.onnx", output)
            self.assertIn("1.0 KiB", output)
            self.assertEqual(analysis.scan.model_artifacts[0].size_bytes, 1024)

    def test_heavy_model_alias_creates_sample_project(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("/sample heavy")
            finally:
                os.chdir(cwd)

            self.assertIsNotNone(message)
            self.assertTrue(Path(path).exists())
            self.assertTrue((Path(path) / "model" / "heavy-model.onnx").exists())

    def test_beginner_intro_lists_existing_sample_projects(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                sample = Path(tmpdir) / ".aiu" / "sample_projects" / "custom-added-model"
                sample.mkdir(parents=True)

                intro = build_beginner_intro()
                samples = list_existing_sample_projects()
                resolved = resolve_existing_sample_project("/sample custom-added-model")
                path, message = resolve_beginner_project_input("/sample custom-added-model")
            finally:
                os.chdir(cwd)

            self.assertIn("/sample custom-added-model", intro)
            self.assertEqual([path.name for path in samples], ["custom-added-model"])
            self.assertEqual(resolved, sample.resolve())
            self.assertEqual(Path(path), sample.resolve())
            self.assertIn("기존 샘플 프로젝트를 선택했습니다", message)

    def test_tensorflow_sample_alias_creates_fixable_project(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("/sample tensorflow")
            finally:
                os.chdir(cwd)

            sample = Path(path)
            analysis = analyze_project(str(sample))

            self.assertIsNotNone(message)
            self.assertTrue((sample / "model" / "tensorflow-sample.keras").exists())
            self.assertIn("tensorflow", (sample / "requirements.txt").read_text())
            self.assertNotIn("mlflow", (sample / "requirements.txt").read_text().lower())
            self.assertEqual(analysis.registration_status, "보완 필요")

            wizard = build_beginner_wizard(str(sample))
            self.assertIn("tensorflow-sample.keras", wizard)
            self.assertIn("MLflow 의존성 추가", wizard)
            self.assertIn("MLflow 기록 코드 추가", wizard)
            self.assertIn("적용하기", wizard)

            dry_run = json.loads(handle_advanced_input(f"ml-agent fix {sample} --dry-run --json"))
            self.assertEqual(len(dry_run["fix_previews"]), 2)
            self.assertNotIn("mlflow", (sample / "requirements.txt").read_text().lower())

            applied = json.loads(handle_advanced_input(f"ml-agent apply {sample} --json"))
            self.assertEqual(applied["command"], "apply")
            self.assertEqual(len(applied["applied_changes"]), 2)
            self.assertIn("mlflow", (sample / "requirements.txt").read_text().lower())
            self.assertIn("import mlflow", (sample / "train.py").read_text())
            self.assertEqual(analyze_project(str(sample)).registration_status, "등록 가능")

    def test_pytorch_korean_alias_creates_registerable_project(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("파이토치")
            finally:
                os.chdir(cwd)

            sample = Path(path)
            analysis = analyze_project(str(sample))

            self.assertIsNotNone(message)
            self.assertTrue((sample / "model" / "pytorch-sample.pt").exists())
            self.assertIn("torch", (sample / "requirements.txt").read_text())
            self.assertEqual(analysis.registration_status, "등록 가능")

    def test_sora_korean_alias_creates_registerable_video_project(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("소라모델")
            finally:
                os.chdir(cwd)

            sample = Path(path)
            analysis = analyze_project(str(sample))
            output = build_beginner_wizard(str(sample))

            self.assertIsNotNone(message)
            self.assertTrue((sample / "model" / "sora-video-sample.onnx").exists())
            self.assertIn("opencv-python", (sample / "requirements.txt").read_text())
            self.assertIn("sora-style-video-generation", (sample / "train.py").read_text())
            self.assertEqual(analysis.registration_status, "등록 가능")
            self.assertIn("sora-video-sample.onnx", output)
            self.assertIn("64.0 MiB", output)

    def test_sora_error_alias_creates_broken_video_project(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("/sample sora-error")
            finally:
                os.chdir(cwd)

            sample = Path(path)
            analysis = analyze_project(str(sample))
            output = build_beginner_wizard(str(sample))

            self.assertIsNotNone(message)
            self.assertIn("오류/수정 흐름", message)
            self.assertTrue((sample / "outputs" / "sora-preview.mp4").exists())
            self.assertNotIn("mlflow", (sample / "requirements.txt").read_text().lower())
            self.assertEqual(analysis.registration_status, "보완 필요")
            issue_codes = {issue.code for issue in analysis.issue_details}
            self.assertIn("MLFLOW_DEPENDENCY_MISSING", issue_codes)
            self.assertIn("MLFLOW_CODE_MISSING", issue_codes)
            self.assertIn("MODEL_ARTIFACT_MISSING", issue_codes)
            self.assertIn("sora-preview.mp4", (sample / "train.py").read_text())
            self.assertIn("모델 파일 후보 없음", output)

    def test_sora_error_korean_alias_is_available_in_intro(self):
        intro = build_beginner_intro()

        self.assertIn("/sample sora-error", intro)

    def test_sample_all_creates_multiple_model_projects(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("/sample all")
            finally:
                os.chdir(cwd)

            root = Path(tmpdir) / ".aiu" / "sample_projects"
            expected = [
                root / "heavy-model",
                root / "tensorflow-model",
                root / "pytorch-model",
                root / "sklearn-model",
                root / "onnx-model",
                root / "sora-video-model",
            ]

            self.assertIsNotNone(message)
            self.assertEqual(Path(path).resolve(), expected[0].resolve())
            self.assertTrue(all(project.exists() for project in expected))
            self.assertFalse((root / "sora-error-model").exists())
            statuses = {project.name: analyze_project(str(project)).registration_status for project in expected}
            self.assertEqual(statuses["tensorflow-model"], "보완 필요")
            self.assertTrue(
                all(status == "등록 가능" for name, status in statuses.items() if name != "tensorflow-model")
            )

    def test_sample_matrix_summarizes_multiple_model_results(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("/sample matrix")
            finally:
                os.chdir(cwd)

            self.assertIsNotNone(message)
            self.assertTrue(Path(path).exists())
            self.assertIn("Step 1 검증을 완료했습니다", message)
            self.assertIn("tensorflow-model: 보완 필요", message)
            self.assertIn("pytorch-model: 등록 가능", message)
            self.assertIn("sora-video-model: 등록 가능", message)
            self.assertIn("issues=0", message)
            self.assertIn("issues=2", message)

    def test_large10_sample_alias_creates_ten_large_model_projects(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("/sample large10")
            finally:
                os.chdir(cwd)

            root = Path(tmpdir) / ".aiu" / "sample_projects"
            projects = sorted(path for path in root.iterdir() if path.is_dir())

            self.assertIsNotNone(message)
            self.assertEqual(len(projects), 10)
            self.assertEqual(Path(path).resolve(), (root / "large-tensorflow-model").resolve())
            self.assertTrue((root / "large-sora-video-model" / "model" / "large-sora-video.onnx").exists())
            self.assertTrue((root / "large-llm-adapter" / "model" / "llm-adapter.safetensors").exists())
            self.assertIn("대형 모델 테스트 샘플 10개", message)
            self.assertIn("large-sora-video-model: 등록 가능", message)

    def test_create_large_model_samples_supports_small_test_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / ".aiu" / "sample_projects"
            sample_paths = create_large_model_samples(root, artifact_size_bytes=1024)

            self.assertEqual(len(sample_paths), 10)
            for sample in sample_paths:
                analysis = analyze_project(str(sample))
                self.assertEqual(analysis.registration_status, "등록 가능")
                self.assertEqual(analysis.scan.model_artifacts[0].size_bytes, 1024)


class ProjectAnalysisTest(unittest.TestCase):
    def test_missing_path_is_not_registerable(self):
        analysis = analyze_project("/path/that/does/not/exist")

        self.assertEqual(analysis.registration_status, "불가")
        self.assertFalse(analysis.exists)
        self.assertIn("프로젝트 경로를 찾을 수 없습니다.", analysis.issues)
        self.assertEqual(analysis.issue_details[0].code, "PROJECT_PATH_NOT_FOUND")

    def test_project_with_missing_mlflow_needs_action(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            analysis = analyze_project(str(root))

            self.assertEqual(analysis.registration_status, "보완 필요")
            self.assertFalse(analysis.has_mlflow_dependency)
            self.assertIn("train.py", analysis.entrypoint_candidates)
            self.assertTrue(any(issue.code == "MLFLOW_DEPENDENCY_MISSING" for issue in analysis.issue_details))


class IntermediateModeTest(unittest.TestCase):
    def test_mlflow_request_gets_mlflow_guidance(self):
        output = handle_intermediate_request("MLflow 설정만 확인해줘")

        self.assertIn("MLflow 설정 검증", output)
        self.assertIn("mlflow-validator", output)
        self.assertIn("dry-run", output)

    def test_job_template_request_gets_template_guidance(self):
        output = handle_intermediate_request("Job Template 초안 만들어줘")

        self.assertIn("Job Template 초안", output)
        self.assertIn("job-template-planner", output)
        self.assertIn("항목별", output)


class AdvancedModeTest(unittest.TestCase):
    def test_fix_defaults_to_dry_run_guidance(self):
        output = handle_advanced_input("ml-agent fix ./project")

        self.assertIn("default=dry-run", output)
        self.assertIn("advanced_apply_required=true", output)

    def test_json_output(self):
        output = handle_advanced_input("ml-agent validate ./project --json")
        payload = json.loads(output)

        self.assertEqual(payload["command"], "validate")
        self.assertEqual(payload["path"], "./project")
        self.assertEqual(payload["exit_code"], 2)
        self.assertIn("agent_profile=ai-ml-onboarding-assistant", payload["details"])
        self.assertEqual(payload["analysis"]["registration_status"], "불가")
        self.assertEqual(payload["analysis"]["issue_details"][0]["code"], "PROJECT_PATH_NOT_FOUND")

    def test_validate_json_contains_step3_analysis(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow==2.17.0\n")
            (root / "train.py").write_text(
                "import mlflow\n"
                "if __name__ == \"__main__\":\n"
                "    mlflow.log_param('x', 1)\n"
            )
            (root / "model.onnx").write_text("sample")

            output = handle_advanced_input(f"ml-agent validate {root} --json")
            payload = json.loads(output)

            self.assertEqual(payload["exit_code"], 0)
            self.assertEqual(payload["analysis"]["registration_status"], "등록 가능")
            self.assertTrue(payload["analysis"]["job_template_ready"])
            self.assertTrue(all(check["status"] == "pass" for check in payload["analysis"]["registration_checks"]))

    def test_validate_json_contains_step2_scan(self):
        with TemporaryDirectory() as tmpdir:
            sample = create_heavy_model_sample(Path(tmpdir) / "heavy-model", artifact_size_bytes=2048)

            output = handle_advanced_input(f"ml-agent validate {sample} --json")
            payload = json.loads(output)

            self.assertEqual(payload["analysis"]["scan"]["model_artifacts"][0]["path"], "model/heavy-model.onnx")
            self.assertEqual(payload["analysis"]["scan"]["model_artifacts"][0]["size_bytes"], 2048)
            self.assertEqual(payload["analysis"]["scan"]["model_artifacts"][0]["size"], "2.0 KiB")
            artifact_check = [
                check
                for check in payload["analysis"]["registration_checks"]
                if check["code"] == "model_artifact"
            ][0]
            self.assertEqual(artifact_check["status"], "pass")
            self.assertIn("2.0 KiB", artifact_check["detail"])

    def test_fix_json_contains_step5_previews(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = handle_advanced_input(f"ml-agent fix {root} --dry-run --json")
            payload = json.loads(output)

            self.assertEqual(payload["command"], "fix")
            self.assertIn("preview_items=2", payload["details"])
            self.assertEqual(len(payload["fix_previews"]), 2)
            self.assertEqual(payload["fix_previews"][0]["code"], "ADD_MLFLOW_DEPENDENCY")
            self.assertTrue(payload["fix_previews"][0]["requires_approval"])

    def test_fix_json_contains_step6_approval_options(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = handle_advanced_input(f"ml-agent fix {root} --dry-run --json")
            payload = json.loads(output)

            self.assertIn("approval_required=true", payload["details"])
            self.assertEqual(payload["approval_options"][0]["key"], "apply")
            self.assertTrue(payload["approval_options"][0]["enabled"])
            self.assertTrue(payload["approval_options"][0]["will_modify_files"])
            self.assertEqual(payload["approval_options"][1]["key"], "review")
            self.assertFalse(payload["approval_options"][1]["will_modify_files"])

    def test_fix_dry_run_does_not_modify_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("scikit-learn==1.5.2\n")
            train.write_text("print('train')\n")

            handle_advanced_input(f"ml-agent fix {root} --dry-run --json")

            self.assertEqual(requirements.read_text(), "scikit-learn==1.5.2\n")
            self.assertEqual(train.read_text(), "print('train')\n")

    def test_apply_modifies_only_previewed_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            untouched = root / "README.md"
            requirements.write_text("scikit-learn==1.5.2\n")
            train.write_text("print('train')\n")
            untouched.write_text("hello\n")

            output = handle_advanced_input(f"ml-agent apply {root} --json")
            payload = json.loads(output)

            self.assertEqual(payload["command"], "apply")
            self.assertIn("applied_changes=2", payload["details"])
            self.assertEqual(len(payload["applied_changes"]), 2)
            self.assertIn("mlflow", requirements.read_text())
            self.assertIn("import mlflow", train.read_text())
            self.assertIn("MLflow tracking template", train.read_text())
            self.assertEqual(untouched.read_text(), "hello\n")

    def test_apply_creates_requirements_when_missing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.py").write_text("print('train')\n")

            output = handle_advanced_input(f"ml-agent apply {root} --json")
            payload = json.loads(output)

            self.assertTrue((root / "requirements.txt").exists())
            self.assertIn("mlflow", (root / "requirements.txt").read_text())
            self.assertTrue(any(change["code"] == "CREATE_REQUIREMENTS" for change in payload["applied_changes"]))

    def test_serve_json_contains_local_serving_plan(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\nscikit-learn\n")
            (root / "train.py").write_text("import mlflow\n")
            (root / "model.onnx").write_text("sample")

            output = handle_advanced_input(f"ml-agent serve {root} --dry-run --json")
            payload = json.loads(output)
            serving = payload["analysis"]["local_serving"]

            self.assertEqual(payload["command"], "serve")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(serving["status"], "준비 가능")
            self.assertEqual(serving["health_endpoint"], "http://127.0.0.1:8000/health")
            self.assertEqual(serving["predict_endpoint"], "http://127.0.0.1:8000/predict")
            self.assertIn("local_serving=준비 가능", payload["details"])

    def test_report_writes_final_analysis_file(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\nscikit-learn\n")
            (root / "train.py").write_text("import mlflow\n")
            (root / "model.onnx").write_text("sample")

            output = handle_advanced_input(f"ml-agent report {root} --json")
            payload = json.loads(output)
            report_path = root / "ml-agent-report.json"
            report = json.loads(report_path.read_text())

            self.assertEqual(payload["command"], "report")
            self.assertTrue(report_path.exists())
            self.assertEqual(report["title"], "AI ML 온보딩 분석 리포트")
            self.assertEqual(report["summary"]["registration_status"], "등록 가능")
            self.assertEqual(report["summary"]["issue_count"], 0)
            self.assertEqual(report["summary"]["mlflow"], "ok")
            self.assertEqual(report["summary"]["local_serving"], "준비 가능")
            self.assertEqual(report["summary"]["health_endpoint"], "http://127.0.0.1:8000/health")

    def test_profile_command_outputs_deep_agent_profile(self):
        output = handle_advanced_input("ml-agent profile")

        self.assertIn("Deep Agent Profile", output)
        self.assertIn("project-scanner", output)
        self.assertIn("적용하기", output)
        self.assertIn("libs_reference", output)

    def test_deepagents_command_outputs_libs_manifest(self):
        output = handle_advanced_input("ml-agent deepagents")

        self.assertIn("DeepAgents Libs", output)
        self.assertIn("libs/deepagents", output)
        self.assertIn("libs/code", output)
        self.assertIn("runtime_import: deepagents", output)

    def test_deepagents_command_outputs_json(self):
        output = handle_advanced_input("ml-agent deepagents --json")
        payload = json.loads(output)

        self.assertEqual(payload["runtime_import"], "deepagents")
        self.assertIn("https://github.com/langchain-ai/deepagents/tree/main/libs", payload["reference"])
        self.assertEqual(payload["source_type"], "directory")
        self.assertIn("deep_agent/vendor/deepagents", payload["source_path"])
        self.assertTrue(any(item["path"] == "libs/deepagents" for item in payload["libs"]))

    def test_deepagents_source_is_committed_under_repo(self):
        source_root = Path(__file__).resolve().parents[1] / "deep_agent" / "vendor" / "deepagents" / "deepagents-main" / "libs"

        self.assertTrue(source_root.exists())
        self.assertTrue((source_root / "deepagents" / "pyproject.toml").exists())
        self.assertTrue((source_root / "code" / "pyproject.toml").exists())

    def test_deepagents_libs_can_be_read_from_source_zip(self):
        with TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "deepagents-main.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "deepagents-main/libs/deepagents/pyproject.toml",
                    '[project]\nname = "deepagents"\ndescription = "Core Deep Agents runtime."\n',
                )
                archive.writestr(
                    "deepagents-main/libs/talon/pyproject.toml",
                    '[project]\nname = "deepagents-talon"\ndescription = "Talon runtime package."\n',
                )

            payload = deepagents_libs_as_dict(str(zip_path))

            self.assertEqual(payload["libs_source"], "zip")
            self.assertEqual(payload["source_zip"], str(zip_path))
            self.assertTrue(any(item["path"] == "libs/talon" for item in payload["libs"]))
            self.assertTrue(any(item["required_now"] for item in payload["libs"]))

    def test_deepagents_command_accepts_source_zip(self):
        with TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "deepagents-main.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "deepagents-main/libs/partners/quickjs/pyproject.toml",
                    '[project]\nname = "langchain-quickjs"\ndescription = "QuickJS sandbox package."\n',
                )

            output = handle_advanced_input(f"ml-agent deepagents --source {zip_path} --json")
            payload = json.loads(output)

            self.assertEqual(payload["libs_source"], "zip")
            self.assertTrue(any(item["path"] == "libs/partners/quickjs" for item in payload["libs"]))

    def test_config_command_outputs_env_summary(self):
        output = handle_advanced_input("ml-agent config")

        self.assertIn("Environment Config", output)
        self.assertIn("qwen_model=qwen3.6", output)
        self.assertIn("skill_store_dir", output)

    def test_prompts_command_outputs_prompt_templates(self):
        output = handle_advanced_input("ml-agent prompts")

        self.assertIn("Prompt templates", output)
        self.assertIn("launch_mode_router", output)
        self.assertIn("mlflow_registration_check", output)

    def test_errors_list_command_outputs_empty_state(self):
        output = handle_advanced_input("ml-agent errors list")

        self.assertIn("error logs:", output)


class DeepAgentProfileTest(unittest.TestCase):
    def test_profile_contains_deepagents_harness_concepts(self):
        profile = build_ml_platform_profile("beginner")

        self.assertEqual(profile.name, "ai-ml-onboarding-assistant")
        self.assertGreaterEqual(len(profile.subagents), 4)
        self.assertTrue(any(rule.mode == "interrupt" for rule in profile.permissions))
        self.assertIn("task", profile.tools)
        self.assertIn("mlflow-registration-check", profile.skills)

    def test_profile_format_references_upstream(self):
        output = format_profile(build_ml_platform_profile("advanced"))

        self.assertIn("https://github.com/langchain-ai/deepagents", output)
        self.assertIn("https://github.com/langchain-ai/deepagents/tree/main/libs", output)
        self.assertIn("permissions:", output)


class WindowsSetupTest(unittest.TestCase):
    def test_tui_subcommand_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["tui"])

        self.assertEqual(args.command, "tui")

    def test_aiu_module_entrypoint_delegates_to_ml_agent(self):
        from aiu.__main__ import main as aiu_main

        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = aiu_main(["config"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Environment Config", stdout.getvalue())

    def test_tui_missing_textual_message_is_actionable(self):
        message = missing_textual_message()

        self.assertIn("Textual", message)
        self.assertIn('pip install ".[tui,deepagents]"', message)
        self.assertIn("Windows Terminal", message)

    def test_tui_command_placeholder_shows_active_agent_mode(self):
        self.assertEqual(command_placeholder_for_mode("Plan"), "")
        self.assertEqual(command_placeholder_for_mode("Build"), "")
        self.assertEqual(command_placeholder_for_mode("Chatbot"), "")

    def test_tui_agent_mode_selector_and_command_parser(self):
        self.assertEqual(format_agent_mode_selector("Plan"), "Plan build chatbot")
        self.assertEqual(format_agent_mode_selector("Build"), "plan Build chatbot")
        self.assertEqual(format_agent_mode_selector("Chatbot"), "plan build Chatbot")
        self.assertEqual(parse_agent_mode_command("/agent plan"), "Plan")
        self.assertEqual(parse_agent_mode_command("/agent build"), "Build")
        self.assertEqual(parse_agent_mode_command("/agent chat"), "Chatbot")
        self.assertEqual(parse_agent_mode_command("/에이전트 챗봇"), "Chatbot")
        self.assertEqual(parse_agent_mode_command("/agent"), "")
        self.assertIsNone(parse_agent_mode_command("agent chat"))

    def test_tui_command_placeholder_mentions_deepagents_model(self):
        self.assertNotIn("DeepAgents", command_placeholder_for_mode("Plan", "qwen3.6"))
        self.assertNotIn("qwen3.6", command_placeholder_for_mode("Build", "qwen3.6"))

    def test_tui_command_placeholder_mentions_model_command(self):
        self.assertNotIn("/path", command_placeholder_for_mode("Plan", "qwen3.6"))
        self.assertNotIn("/model", command_placeholder_for_mode("Build", "gamma"))

    def test_tui_model_selection_placeholder_shows_number_range(self):
        self.assertIn("1-4", model_selection_placeholder(["qwen3.6", "qwen3.5", "gpt20", "gamma"]))
        self.assertIn("Tab/화살표", model_selection_placeholder(["qwen3.6"]))
        self.assertIn("모델명", model_selection_placeholder([]))

    def test_tui_parse_model_commands(self):
        self.assertEqual(parse_model_command("/model qwen3.5"), "qwen3.5")
        self.assertEqual(parse_model_command("/모델 gamma"), "gamma")
        self.assertEqual(parse_model_command("/model"), "")
        self.assertIsNone(parse_model_command("모델 gamma"))

    def test_tui_formats_model_choices_with_current_marker(self):
        output = format_model_choices(["qwen3.6", "gamma"], "gamma")

        self.assertIn("1. qwen3.6", output)
        self.assertIn("2. gamma (현재)", output)
        self.assertIn("번호를 입력", output)

    def test_tui_controller_lists_and_selects_model_from_input_box(self):
        controller = BeginnerTuiController("")

        list_output = controller.submit("/model")
        self.assertIn("1. qwen3.6", list_output)
        self.assertIn("gamma", list_output)
        self.assertTrue(controller.awaiting_model_selection)

        output = controller.submit("/model gamma")

        self.assertEqual(output, "")
        self.assertEqual(controller.qwen_model, "gamma")
        self.assertFalse(controller.awaiting_model_selection)
        self.assertEqual(command_placeholder_for_mode(controller.agent_mode, controller.qwen_model), "")

    def test_tui_controller_selects_model_by_number_after_model_menu(self):
        controller = BeginnerTuiController("")

        controller.submit("/model")
        output = controller.submit("2")

        self.assertEqual(output, "")
        self.assertEqual(controller.qwen_model, "qwen3.5")
        self.assertFalse(controller.awaiting_model_selection)

    def test_tui_controller_cycles_model_selection_for_input_box(self):
        controller = BeginnerTuiController("")

        controller.submit("/model")
        self.assertEqual(controller.highlighted_model, "qwen3.6")
        self.assertEqual(controller.cycle_model_selection(1), "qwen3.5")
        self.assertEqual(controller.cycle_model_selection(1), "gpt20")
        self.assertEqual(controller.cycle_model_selection(-1), "qwen3.5")

    def test_tui_controller_enter_selects_highlighted_model_when_input_empty(self):
        controller = BeginnerTuiController("")

        controller.submit("/model")
        controller.cycle_model_selection(1)
        output = controller.submit("")

        self.assertEqual(output, "")
        self.assertEqual(controller.qwen_model, "qwen3.5")
        self.assertFalse(controller.awaiting_model_selection)

    def test_tui_controller_selects_model_by_number_in_model_command(self):
        controller = BeginnerTuiController("")

        output = controller.submit("/model 4")

        self.assertEqual(output, "")
        self.assertEqual(controller.qwen_model, "gamma")

    def test_tui_controller_rejects_unknown_model(self):
        controller = BeginnerTuiController("")

        output = controller.submit("/model unknown-model")

        self.assertIn("지원하지 않는 모델", output)
        self.assertEqual(controller.qwen_model, "qwen3.6")

    def test_tui_controller_keeps_model_menu_open_on_invalid_number(self):
        controller = BeginnerTuiController("")

        controller.submit("/model")
        output = controller.submit("99")

        self.assertIn("지원하지 않는 모델", output)
        self.assertTrue(controller.awaiting_model_selection)
        self.assertEqual(controller.qwen_model, "qwen3.6")

    def test_tui_detects_fix_request_text(self):
        self.assertTrue(is_fix_request("코드 자동 수정해줘"))
        self.assertTrue(is_fix_request("please fix this project"))
        self.assertFalse(is_fix_request("프로젝트 상태 알려줘"))

    def test_tui_normalizes_direct_folder_path_input(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            self.assertEqual(normalize_input_path(str(root)), root.resolve())
            self.assertEqual(normalize_input_path(f'"{root}"'), root.resolve())

    def test_tui_normalizes_path_command_and_terminal_paste_variants(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spaced = root / "model with space"
            spaced.mkdir()
            escaped_spaced = str(spaced).replace(" ", "\\ ")
            url_spaced = str(spaced).replace(" ", "%20")

            self.assertEqual(strip_path_command(f"/path {spaced}"), (str(spaced), True))
            self.assertEqual(strip_path_command(f"/경로 {spaced}"), (str(spaced), True))
            self.assertEqual(normalize_input_path(f"/path {escaped_spaced}"), spaced.resolve())
            self.assertEqual(normalize_input_path(f"file://{url_spaced}"), spaced.resolve())
            self.assertEqual(normalize_input_path(f"> {spaced}"), spaced.resolve())
            self.assertEqual(path_candidates_from_input(f"\n{spaced}\nignored"), [str(spaced), "ignored"])

    def test_tui_normalizes_file_path_to_parent_project(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train.py"
            train.write_text("print('train')\n")

            self.assertEqual(normalize_input_path(str(train)), root.resolve())

    def test_tui_controller_selects_direct_path_from_input_box(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("tensorflow==2.17.0\n")
            (root / "train.py").write_text("print('train')\n")

            controller = BeginnerTuiController("")
            output = controller.submit(f'"{root}"')

            self.assertEqual(controller.project_path, str(root.resolve()))
            self.assertEqual(controller.index, 0)
            self.assertIn("Current: Tab 1/10", output)
            self.assertNotIn("프로젝트 경로를 선택했습니다", controller.render_log())
            self.assertNotIn("Qwen qwen3.6:", controller.render_log())

    def test_tui_controller_selects_path_command_from_input_box(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("tensorflow==2.17.0\n")
            (root / "train.py").write_text("print('train')\n")

            controller = BeginnerTuiController("")
            output = controller.submit(f"/path {root}")

            self.assertEqual(controller.project_path, str(root.resolve()))
            self.assertIn("Current: Tab 1/10", output)

    def test_tui_controller_reports_missing_path_for_path_command(self):
        controller = BeginnerTuiController("")

        self.assertIn("경로를 함께 입력", controller.submit("/path"))
        self.assertIn("경로를 찾을 수 없습니다", controller.submit("/path /no/such/model"))

    def test_tui_controller_handles_navigation_and_exit(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("tensorflow==2.17.0\n")
            (root / "train.py").write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = BeginnerTuiController(str(root))

            self.assertEqual(controller.index, 0)
            output = controller.submit("다음")
            self.assertEqual(controller.index, 1)
            self.assertIn("Current: Tab 2/10", output)
            self.assertNotIn("> 다음", controller.render_log())
            self.assertNotIn("다음 단계로 이동합니다.", controller.render_log())

            mode_output = controller.submit("/mode beginner")
            self.assertIn("현재 모드가 초급자 모드로 변경되었습니다", mode_output)

            controller.submit("/exit")
            self.assertTrue(controller.exited)

    def test_tui_controller_toggle_agent_does_not_add_log_noise(self):
        controller = BeginnerTuiController("")
        controller.toggle_agent()

        self.assertEqual(controller.agent_mode, "Build")
        self.assertNotIn("현재 Agent 모드", controller.render_log())

        controller.toggle_agent()
        self.assertEqual(controller.agent_mode, "Chatbot")
        controller.toggle_agent()
        self.assertEqual(controller.agent_mode, "Plan")
        controller.previous_agent()
        self.assertEqual(controller.agent_mode, "Chatbot")

    def test_tui_controller_selects_agent_mode_from_input_box(self):
        controller = BeginnerTuiController("")

        output = controller.submit("/agent chat")

        self.assertEqual(output, "")
        self.assertEqual(controller.agent_mode, "Chatbot")
        self.assertEqual(controller.submit("/agent"), "plan build Chatbot")

    def test_tui_controller_plan_mode_does_not_apply_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = BeginnerTuiController(str(root))
            for _ in range(5):
                controller.submit("다음")

            self.assertEqual(controller.index, 5)
            output = controller.submit("1")

            self.assertNotIn("Build 모드에서만", output)
            self.assertEqual(requirements.read_text(), "tensorflow==2.17.0\n")
            self.assertNotIn("import mlflow", train.read_text())

    def test_tui_chat_without_deepagents_config_does_not_modify_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = BeginnerTuiController(str(root))
            controller.select_agent_mode("Chatbot")

            output = controller.submit("코드 자동 수정해줘")

            self.assertIn("DeepAgents runtime", output)
            self.assertIn("QWEN_API_KEY", output)
            self.assertEqual(requirements.read_text(), "tensorflow==2.17.0\n")
            self.assertNotIn("import mlflow", train.read_text())

    def test_tui_build_mode_text_points_to_chatbot_without_modifying_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = BeginnerTuiController(str(root))
            controller.toggle_agent()

            output = controller.submit("코드 자동 수정해줘")

            self.assertNotIn("Build 모드", output)
            self.assertNotIn("Chatbot 모드", output)
            self.assertEqual(requirements.read_text(), "tensorflow==2.17.0\n")
            self.assertNotIn("import mlflow", train.read_text())

    def test_tui_chat_routes_general_requests_through_deepagents_runtime(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("DeepAgents 응답", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir) / "project"
                root.mkdir()
                (root / "requirements.txt").write_text("mlflow\n")
                (root / "train.py").write_text("import mlflow\n")
                (root / "model.onnx").write_text("sample")
                fake = FakeRuntime()
                controller = BeginnerTuiController(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                output = controller.submit("프로젝트 분석해줘")
            finally:
                os.chdir(cwd)

            self.assertIn("DeepAgents 응답", output)
            self.assertIn("최종 등록 상태", output)
            self.assertEqual(fake.calls[-1], ("프로젝트 분석해줘", str(root), "AutoFix"))
            self.assertIn("DeepAgents 응답", controller.render_log())
            self.assertNotIn("Agent:", controller.render_log())
            session_files = list((Path(tmpdir) / ".aiu" / "sessions").glob("chat-session-*.jsonl"))
            self.assertEqual(len(session_files), 1)
            session_payload = json.loads(session_files[0].read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(session_payload["user_message"], "프로젝트 분석해줘")
            self.assertEqual(session_payload["analysis_status"], "등록 가능")

    def test_tui_chat_autofix_applies_fixable_issues_after_deepagents_success(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("DeepAgents 수정 완료", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir) / "project"
                root.mkdir()
                requirements = root / "requirements.txt"
                train = root / "train.py"
                requirements.write_text("tensorflow==2.17.0\n")
                train.write_text("print('train')\n")
                (root / "model.keras").write_text("sample")
                fake = FakeRuntime()
                controller = BeginnerTuiController(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                output = controller.submit("코드 자동 수정해줘")
            finally:
                os.chdir(cwd)

            self.assertIn("DeepAgents 수정 완료", output)
            self.assertIn("자동 수정 결과", output)
            self.assertEqual(fake.calls[-1], ("코드 자동 수정해줘", str(root), "AutoFix"))
            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertIn("import mlflow", train.read_text())
            self.assertEqual(controller.index, 6)
            session_file = next((Path(tmpdir) / ".aiu" / "sessions").glob("chat-session-*.jsonl"))
            session_payload = json.loads(session_file.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(session_payload["selected_model"], "qwen3.6")
            self.assertTrue(session_payload["applied_changes"])

    def test_tui_chat_autofix_leaves_manual_sora_artifact_issue(self):
        class FakeRuntime:
            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                return DeepAgentsRunResult("Sora 오류를 점검했습니다.", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                path, _ = resolve_beginner_project_input("/sample sora-error")
                sample = Path(path)
                controller = BeginnerTuiController(str(sample), deepagents_runtime=FakeRuntime())
                controller.select_agent_mode("Chatbot")

                output = controller.submit("문제 발견하면 자동 수정해줘")
            finally:
                os.chdir(cwd)

            self.assertIn("자동 수정 결과", output)
            self.assertIn("모델 산출물 없음", output)
            self.assertIn("mlflow", (sample / "requirements.txt").read_text().lower())
            final_codes = {issue.code for issue in analyze_project(str(sample)).issue_details}
            self.assertIn("MODEL_ARTIFACT_MISSING", final_codes)
            self.assertNotIn("MLFLOW_DEPENDENCY_MISSING", final_codes)
            self.assertNotIn("MLFLOW_CODE_MISSING", final_codes)

    def test_tui_controller_build_mode_applies_only_after_approval(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = BeginnerTuiController(str(root))
            controller.toggle_agent()
            for _ in range(5):
                controller.submit("다음")

            output = controller.submit("1")

            self.assertEqual(controller.agent_mode, "Build")
            self.assertEqual(controller.index, 6)
            self.assertIn("Current: Tab 7/10", output)
            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertIn("import mlflow", train.read_text())

    def test_windows_command_wrapper_exists(self):
        wrapper = Path(__file__).resolve().parents[1] / "ml-agent.cmd"

        self.assertTrue(wrapper.exists())
        self.assertIn("py -3", wrapper.read_text())

    def test_quickstart_has_windows_execution_section(self):
        quickstart = Path(__file__).resolve().parents[1] / "QUICKSTART.md"
        content = quickstart.read_text(encoding="utf-8")

        self.assertIn("## 12. Windows 10/11 실행 환경", content)
        self.assertIn(".\\ml-agent.cmd init", content)
        self.assertIn("Linux/macOS에서 확인할 때만", content)
        self.assertIn("## 9. 이미지형 TUI 화면", content)

    def test_tui_preview_image_is_documented(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertTrue((root / "docs" / "tui-preview.svg").exists())
        self.assertIn("docs/tui-preview.svg", readme)


class AppConfigTest(unittest.TestCase):
    def test_env_example_contains_qwen_and_skill_store(self):
        env_example = Path(__file__).resolve().parents[1] / ".env.example"
        content = env_example.read_text(encoding="utf-8")

        self.assertIn("QWEN_API_KEY=your-internal-qwen-key", content)
        self.assertIn("QWEN_BASE_URL=http://xxx.xxx.xxx.xxx:port/v1", content)
        self.assertIn("ENABLE_RICH_CONSOLE=true", content)
        self.assertIn("ENABLE_TUI_BACKGROUND=false", content)
        self.assertIn("ENABLE_TUI_INPUT_PANEL=true", content)
        self.assertIn("SESSION_DIR=.aiu/sessions", content)
        self.assertIn("SKILL_STORE_DIR=deep_agent/skills", content)
        self.assertIn("WIKI_DIR=deep_agent/wiki", content)
        self.assertIn("WIKI_PROMPT_DIR=deep_agent/wiki/prompts", content)

    def test_runtime_layout_creates_skill_store(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig.load(root_dir=root)
            directories = ensure_runtime_layout(config)

            self.assertIn(root / "deep_agent" / "skills", directories)
            self.assertIn(root / ".aiu" / "sessions", directories)
            self.assertTrue((root / "deep_agent" / "skills" / "README.md").exists())
            self.assertTrue((root / "deep_agent" / "skills" / "instrumenting-with-mlflow-tracing" / "SKILL.md").exists())
            self.assertTrue((root / "deep_agent" / "skills" / "agent-evaluation" / "SKILL.md").exists())
            self.assertTrue((root / ".aiu" / "registration_packages").exists())
            self.assertTrue((root / "deep_agent" / "wiki").exists())
            self.assertTrue((root / "deep_agent" / "wiki" / "prompts").exists())
            self.assertTrue((root / "deep_agent" / "wiki" / "prompts" / "prompt_templates.md").exists())
            self.assertTrue((root / "deep_agent" / "wiki" / "prompts" / "prompt_templates.json").exists())

    def test_chat_session_store_masks_sensitive_values(self):
        with TemporaryDirectory() as tmpdir:
            config = AppConfig(
                values={
                    "SESSION_DIR": "sessions",
                    "MASK_SENSITIVE_LOGS": "true",
                    "QWEN_API_KEY": "secret-key",
                    "QWEN_BASE_URL": "http://internal-qwen/v1",
                },
                root_dir=Path(tmpdir),
            )

            path = append_chat_session_event(
                config,
                {
                    "user_message": "key secret-key url http://internal-qwen/v1",
                    "agent_response": "ok secret-key",
                },
            )

            content = path.read_text(encoding="utf-8")
            self.assertIn("***", content)
            self.assertNotIn("secret-key", content)
            self.assertNotIn("http://internal-qwen/v1", content)
            self.assertEqual(mask_sensitive_text("secret-key", config), "***")



class PromptAndSkillStoreTest(unittest.TestCase):
    def test_prompt_templates_are_saved(self):
        templates = load_prompt_templates()
        names = {template.name for template in templates}

        self.assertIn("launch_mode_router", names)
        self.assertIn("job_template_draft", names)
        self.assertIn("closed_network_validation", names)
        self.assertIn("error_log_analysis", names)
        self.assertIn("retry_fix_from_error", names)

    def test_default_skills_exist(self):
        root = Path(__file__).resolve().parents[1]

        for skill_name in DEFAULT_SKILLS:
            self.assertTrue((root / "deep_agent" / "skills" / skill_name / "SKILL.md").exists(), skill_name)

    def test_prompt_templates_export_to_wiki(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_file = root / "deep_agent" / "prompts" / "prompt_templates.json"
            prompt_file.parent.mkdir(parents=True)
            prompt_file.write_text(
                json.dumps(
                    {
                        "templates": [
                            {
                                "name": "sample_prompt",
                                "description": "샘플 프롬프트",
                                "prompt": "프로젝트를 분석하세요.",
                                "tags": ["sample", "wiki"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = AppConfig.load(root_dir=root)
            paths = export_prompt_templates_to_wiki(config)

            markdown = root / "deep_agent" / "wiki" / "prompts" / "prompt_templates.md"
            payload = root / "deep_agent" / "wiki" / "prompts" / "prompt_templates.json"
            self.assertEqual(paths, [markdown, payload])
            self.assertIn("## sample_prompt", markdown.read_text(encoding="utf-8"))
            self.assertIn("프로젝트를 분석하세요.", markdown.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(payload.read_text(encoding="utf-8"))["templates"][0]["name"],
                "sample_prompt",
            )

    def test_runtime_layout_updates_default_skills_and_preserves_custom_skills(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig.load(root_dir=root)
            stale_skill = root / "deep_agent" / "skills" / "mlflow-registration-check" / "SKILL.md"
            custom_skill = root / "deep_agent" / "skills" / "custom-user-skill" / "SKILL.md"
            stale_skill.parent.mkdir(parents=True, exist_ok=True)
            custom_skill.parent.mkdir(parents=True, exist_ok=True)
            stale_skill.write_text("old content\n", encoding="utf-8")
            custom_skill.write_text("custom content\n", encoding="utf-8")

            ensure_runtime_layout(config)

            self.assertEqual(stale_skill.read_text(encoding="utf-8"), DEFAULT_SKILLS["mlflow-registration-check"].strip() + "\n")
            self.assertEqual(custom_skill.read_text(encoding="utf-8"), "custom content\n")

    def test_mlflow_skills_are_in_deep_agent_profile(self):
        profile = build_ml_platform_profile("advanced")

        self.assertIn("instrumenting-with-mlflow-tracing", profile.skills)
        self.assertIn("analyze-mlflow-trace", profile.skills)
        self.assertIn("agent-evaluation", profile.skills)
        self.assertIn("searching-mlflow-docs", profile.skills)
        self.assertIn("mlflow-prompt-management", profile.skills)
        self.assertIn("mlflow-prompt-optimization", profile.skills)
        self.assertIn("mlflow-ai-gateway", profile.skills)
        self.assertIn("mlflow-experiment-tracking", profile.skills)
        self.assertIn("mlflow-model-registry-deployment", profile.skills)


class ErrorLogStoreTest(unittest.TestCase):
    def test_error_log_save_list_and_analyze(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig.load(root_dir=root)
            entry = save_error_log(
                "ModuleNotFoundError: No module named mlflow api_key=secret",
                project_path="./project",
                config=config,
            )
            entries = list_error_logs(config)
            analysis = analyze_error_log(entry.id, config)

            self.assertEqual(len(entries), 1)
            self.assertIn("mlflow", entries[0].tags)
            self.assertIn("requirements", entries[0].tags)
            self.assertNotIn("secret", entries[0].message)
            self.assertIn("ml-agent fix ./project --dry-run", analysis.recommended_command)


if __name__ == "__main__":
    unittest.main()
