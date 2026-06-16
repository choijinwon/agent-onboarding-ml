import json
import os
import subprocess
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import zipfile

from deep_agent import cli as ml_agent
from deep_agent.app_config import (
    AppConfig,
    DEFAULT_SKILLS,
    ensure_local_file_access,
    ensure_read_write_directory,
    ensure_runtime_layout,
    grant_windows_read_write,
)
from deep_agent.profile import build_ml_platform_profile, format_profile
from deep_agent.libs import deepagents_libs_as_dict
from deep_agent.path_utils import filesystem_path_candidates, is_windows_absolute_path, resolve_filesystem_path
from deep_agent.runtime import DeepAgentsRunResult, DeepAgentsRuntime, build_deepagents_system_prompt, extract_deepagents_content
from deep_agent.stores.chat_session_store import append_chat_session_event, mask_sensitive_text
from deep_agent.stores.error_log_store import analyze_error_log, list_error_logs, save_error_log
from deep_agent.stores.prompt_store import append_used_prompt_to_wiki, export_prompt_templates_to_wiki, load_prompt_templates
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
    ensure_ai_studio_sample_runtime,
    ensure_standard_ml_dl_template,
    format_beginner_tab,
    handle_logo_command,
    handle_advanced_input,
    handle_intermediate_request,
    list_existing_sample_projects,
    list_existing_work_projects,
    normalize_clipboard_text,
    parse_mode,
    parse_mode_command,
    resolve_existing_sample_project,
    resolve_existing_work_project,
    resolve_beginner_project_input,
    run_model_source,
    sample_projects_root,
)
from deep_agent.tui import (
    BeginnerTuiController,
    build_compacted_runtime_prompt,
    command_placeholder_for_mode,
    choose_folder_with_dialog,
    compact_chat_entries,
    discover_selectable_folders,
    extract_agent_response_choices,
    format_agent_response_choices,
    folder_selection_placeholder,
    format_agent_mode_selector,
    format_folder_choices,
    format_chat_card,
    format_thinking_animation,
    format_model_choices,
    format_sample_choices,
    format_tui_chatbot_screen,
    format_tui_help_screen,
    format_tui_model_info,
    is_chat_apply_approved,
    is_chat_coding_request,
    is_fix_request,
    missing_textual_message,
    model_selection_placeholder,
    agent_response_choice_placeholder,
    sample_selection_placeholder,
    normalize_input_path,
    normalize_pasted_input,
    normalize_path_text,
    parse_agent_mode_command,
    parse_folder_command,
    parse_model_command,
    path_candidates_from_input,
    is_right_click_event,
    should_use_autofix_chat,
    strip_shell_path_prefix,
    strip_path_command,
    truncate_for_tui,
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
        self.assertIn("read files, search the codebase", build_prompt)
        self.assertIn("create or edit files inside the project root", build_prompt)
        self.assertIn("AutoFix mode", autofix_prompt)
        self.assertIn("apply_ml_fixes automatically", autofix_prompt)
        self.assertIn("ml-platform-onboarding-orchestrator", build_prompt)
        self.assertIn("model-project-standardization", build_prompt)
        self.assertIn("job-template-draft", build_prompt)
        self.assertIn("analysis-reporting", build_prompt)
        self.assertIn("error-log-repair", build_prompt)

    def test_deepagents_prompt_maps_windows_project_to_virtual_root(self):
        prompt = build_deepagents_system_prompt(
            r"C:\Users\choi\AI ML\model",
            "Build",
            filesystem_backend_enabled=True,
        )

        self.assertIn("use virtual paths starting with /", prompt)
        self.assertIn("Do not pass Windows absolute paths", prompt)
        self.assertIn("project_path=/ means the selected project root", prompt)

    def test_deepagents_runtime_uses_selected_project_tool_wrappers(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "runtime.py").read_text(encoding="utf-8")

        self.assertIn("build_project_tools(selected_project_path)", source)
        self.assertIn('if not value or value in {".", "/"}:', source)
        self.assertIn('analyze_selected_ml_project.__name__ = "analyze_ml_project"', source)

    def test_extract_deepagents_content_reads_last_message(self):
        class Message:
            content = "마지막 응답"

        self.assertEqual(extract_deepagents_content({"messages": [Message()]}), "마지막 응답")
        self.assertEqual(extract_deepagents_content({"output": "출력"}), "출력")

    def test_chat_runtime_does_not_reuse_cached_agent_and_sets_timeout(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "runtime.py").read_text(encoding="utf-8")

        self.assertIn('use_cache = agent_mode != "Chat"', source)
        self.assertIn('if use_cache and cache_key in self._agent_cache:', source)
        self.assertIn('if use_cache:', source)
        self.assertIn('timeout=self.app_config.get_int("DEV_COMMAND_TIMEOUT", default=120)', source)

    def test_deepagents_runtime_configures_local_filesystem_backend_and_permissions(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "runtime.py").read_text(encoding="utf-8")

        self.assertIn("ensure_local_file_access(resolved_project)", source)
        self.assertIn("FilesystemBackend(root_dir=str(resolved_project), virtual_mode=True)", source)
        self.assertIn('write_mode = "allow" if agent_mode in {"Build", "AutoFix"} else "deny"', source)
        self.assertIn('FilesystemPermission(operations=["read"], paths=["/**"], mode="allow")', source)
        self.assertIn('FilesystemPermission(operations=["write"], paths=["/**"], mode=write_mode)', source)


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
        self.assertEqual(parse_mode("mid"), MODE_INTERMEDIATE)
        self.assertEqual(parse_mode("미드"), MODE_INTERMEDIATE)


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
        self.assertIn("STEPS 1-10", first_tab)
        self.assertIn("CURRENT STEP", first_tab)
        self.assertIn("[PLAN] | BUILD | CHAT", first_tab)
        self.assertIn("Active agent: Plan read-only", first_tab)
        self.assertIn("esc interrupt", first_tab)
        self.assertIn("> Step 01. 프로젝트 선택", first_tab)
        self.assertIn("Step 10. 리포트", first_tab)
        self.assertIn("Enter=다음", first_tab)

        apply_tab = format_beginner_tab(6, len(steps), steps[6])
        self.assertIn("PLAN | [BUILD] | CHAT", apply_tab)
        self.assertIn("Active agent: Build approval", apply_tab)

    def test_launch_screen_uses_terminal_console_layout(self):
        self.assertIn("AI ML Onboarding Console", LAUNCH_SCREEN)
        self.assertIn("# Launch workflow", LAUNCH_SCREEN)
        self.assertIn("Plan(read-only)", LAUNCH_SCREEN)
        self.assertIn("esc interrupt", LAUNCH_SCREEN)

    def test_logo_command_outputs_copyable_console_logo(self):
        output = handle_logo_command()

        self.assertEqual(output, LAUNCH_SCREEN)
        self.assertIn("AI ML Onboarding Console", output)

    def test_logo_command_can_copy_console_logo(self):
        with patch("deep_agent.cli.copy_text_to_clipboard", return_value=(True, "test-clipboard")) as copy:
            output = handle_logo_command(copy_to_clipboard=True)

        copy.assert_called_once_with(LAUNCH_SCREEN)
        self.assertIn("AI ML Onboarding Console", output)
        self.assertIn("logo copied to clipboard: test-clipboard", output)

    def test_clipboard_copy_normalizes_korean_multiline_text(self):
        raw = "\u1112\u1161\u11ab\u1100\u1173\u11af\r\n멀티\x00 input"

        output = normalize_clipboard_text(raw)

        self.assertEqual(output, "한글\n멀티 input")

    def test_clipboard_copy_repairs_common_korean_mojibake(self):
        raw = "íê¸ ë³µì¬"

        output = normalize_clipboard_text(raw)

        self.assertEqual(output, "한글 복사")

    def test_clipboard_copy_uses_utf8_and_normalized_text(self):
        calls = []

        def fake_run(command, *, input="", text=False, encoding=None, check=False, capture_output=False):
            calls.append((command, input, text, encoding, check, capture_output))
            return subprocess.CompletedProcess(command, 0)

        with patch("deep_agent.cli.shutil.which", return_value="/usr/bin/pbcopy"):
            with patch("deep_agent.cli.sys.platform", "darwin"):
                with patch("deep_agent.cli.subprocess.run", side_effect=fake_run):
                    copied, detail = ml_agent.copy_text_to_clipboard("\u1112\u1161\u11ab\u1100\u1173\u11af\r\n복사")

        self.assertTrue(copied)
        self.assertEqual(detail, "pbcopy")
        self.assertEqual(calls[0][1], "한글\n복사")
        self.assertEqual(calls[0][3], "utf-8")

    def test_logo_subcommand_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["logo", "--copy"])

        self.assertEqual(args.command, "logo")
        self.assertTrue(args.copy)

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
            self.assertIn("PLAN", output)
            self.assertIn("BUILD", output)
            self.assertIn("CHAT", output)
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
            self.assertEqual(train.read_text(), "print('train')\n")
            self.assertTrue((root / "run_model.py").exists())
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
            ensure_ai_studio_sample_runtime(root)
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
            self.assertIn("필수 확인: 1개", output)
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
            self.assertIn("AI Studio MLflow 등록 스캐폴드 생성", output)
            self.assertIn("적용하려면 다음 단계에서 1번을 선택합니다", output)

    def test_beginner_wizard_shows_step6_approval_choices(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 6. 사용자 승인", output)
            self.assertIn("선택하세요. 번호만 입력하면 됩니다.", output)
            self.assertIn("1. 승인 후 생성/수정", output)
            self.assertIn("Step 5의 미리보기 항목만 파일에 반영합니다.", output)
            self.assertIn("2. 수정안 다시 보기", output)
            self.assertIn("3. 취소", output)
            self.assertIn("아직 파일은 수정되지 않았습니다.", output)

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
            (root / "config.json").write_text("{}\n")
            (root / "ai_studio.env").write_text("MLFLOW_TRACKING_URL=\n")
            (root / "input_example.json").write_text("{}\n")
            (root / "mlflow_ai_studio_logging.py").write_text("import mlflow\n")
            (root / "run_model.py").write_text("from mlflow_ai_studio_logging import main\n")
            (root / "aiu_custom").mkdir()
            (root / "aiu_custom" / "model_wrapper.py").write_text("class ModelWrapper: pass\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("자동 수정할 항목이 없습니다.", output)
            self.assertIn("Step 6은 자동 스킵됩니다.", output)
            self.assertIn("파일을 생성하거나 수정하지 않습니다.", output)

    def test_beginner_wizard_shows_step7_apply_scope(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 7. 파일 생성 또는 수정", output)
            self.assertIn("Step 6에서 1번을 선택한 경우에만", output)
            self.assertIn("삭제 작업은 수행하지 않습니다.", output)
            self.assertIn("1. 승인 후 생성/수정", output)
            self.assertIn("requirements.txt: MLflow 의존성 추가", output)
            self.assertIn(".: AI Studio MLflow 등록 스캐폴드 생성", output)

    def test_beginner_wizard_step7_saves_ai_studio_env_values(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = BeginnerTuiController(str(root))
            controller.activate_launch_mode(MODE_BEGINNER)
            for _ in range(5):
                controller.submit("다음")
            controller.submit("1")

            self.assertEqual(controller.index, 6)
            output = controller.submit("1")
            self.assertIn("AI Studio 환경설정 입력", output)
            self.assertTrue(controller.awaiting_ai_studio_env)

            controller.submit("http://mlflow.local:5000")
            controller.submit("ai-user")
            controller.submit("ai-pass")
            controller.submit("sora-exp")
            output = controller.submit("sora-model")

            self.assertFalse(controller.awaiting_ai_studio_env)
            self.assertIn("AI Studio 환경설정이 저장되었습니다", output)
            env_source = (root / "ai_studio.env").read_text(encoding="utf-8")
            self.assertIn("MLFLOW_TRACKING_URL=http://mlflow.local:5000", env_source)
            self.assertIn("MLFLOW_TRACKING_USERNAME=ai-user", env_source)
            self.assertIn("MLFLOW_TRACKING_PASSWORD=ai-pass", env_source)
            self.assertIn("MLFLOW_EXPERIMENT_NAME=sora-exp", env_source)
            self.assertIn("MLFLOW_REGISTER_MODEL_NAME=sora-model", env_source)

    def test_beginner_wizard_shows_step9_local_serving(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow==2.17.0\n")
            (root / "train.py").write_text("import mlflow\n")
            ensure_ai_studio_sample_runtime(root)
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
            ensure_ai_studio_sample_runtime(root)
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

    def test_beginner_intro_lists_work_model_projects(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                work_model = Path(tmpdir) / "work" / "user-model"
                work_model.mkdir(parents=True)
                (work_model / "requirements.txt").write_text("scikit-learn==1.5.0\n")
                (work_model / "train.py").write_text("print('train')\n")
                (work_model / "model").mkdir()
                (work_model / "model" / "model.pkl").write_text("artifact")

                intro = build_beginner_intro()
                work_projects = list_existing_work_projects()
                resolved_by_command = resolve_existing_work_project("/work user-model")
                resolved_by_name = resolve_existing_work_project("user-model")
                path, message = resolve_beginner_project_input("user-model")
            finally:
                os.chdir(cwd)

            self.assertIn("work/에 있는 모델 프로젝트", intro)
            self.assertIn("/work user-model", intro)
            self.assertEqual([path.name for path in work_projects], ["user-model"])
            self.assertEqual(resolved_by_command, work_model.resolve())
            self.assertEqual(resolved_by_name, work_model.resolve())
            self.assertEqual(Path(path), work_model.resolve())
            self.assertIn("work 디렉토리 모델 프로젝트를 선택했습니다", message)

    def test_beginner_number_one_runs_multiple_samples_without_command(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("1")
            finally:
                os.chdir(cwd)

            self.assertTrue(Path(path).exists())
            self.assertIn("샘플 모델 실행을 완료했습니다", message)
            self.assertIn("총", message)
            self.assertIn("성공", message)

    def test_beginner_number_two_runs_local_train_sample_without_command(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("2")
            finally:
                os.chdir(cwd)

            self.assertTrue(Path(path).exists())
            self.assertIn("샘플 모델 실행을 완료했습니다", message)
            self.assertIn("standard-pytorch", path)
            self.assertTrue(any((Path(path) / "saved_model").glob("local_model*.pt")))

    def test_beginner_number_six_creates_sora_error_sample_without_command(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("6")
            finally:
                os.chdir(cwd)

            sample = Path(path)
            self.assertTrue(sample.exists())
            self.assertIn("오류/수정 흐름", message)
            self.assertTrue((sample / "outputs" / "sora-preview.mp4").exists())

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
            self.assertIn("적용하기", wizard)

            dry_run = json.loads(handle_advanced_input(f"ml-agent fix {sample} --dry-run --json"))
            self.assertEqual(len(dry_run["fix_previews"]), 1)
            self.assertNotIn("mlflow", (sample / "requirements.txt").read_text().lower())

            applied = json.loads(handle_advanced_input(f"ml-agent apply {sample} --json"))
            self.assertEqual(applied["command"], "apply")
            self.assertEqual(len(applied["applied_changes"]), 1)
            self.assertIn("mlflow", (sample / "requirements.txt").read_text().lower())
            self.assertNotIn("import mlflow", (sample / "train.py").read_text())
            self.assertTrue((sample / "run_model.py").exists())
            self.assertTrue((sample / "config.json").exists())
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

    def test_root_sora_runner_creates_executable_sample(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sora-root-model"
            script = Path(__file__).resolve().parents[1] / "run_sora_model.py"

            result = subprocess.run(
                [sys.executable, str(script), "--target", str(target), "--create-only"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Sora model sample ready", result.stdout)
            self.assertTrue((target / "model" / "sora-video-sample.onnx").exists())
            self.assertTrue((target / "run_model.py").exists())
            self.assertTrue((target / "ai_studio.env").exists())
            self.assertTrue((target / "config.json").exists())
            self.assertEqual(analyze_project(str(target)).registration_status, "등록 가능")

    def test_root_run_model_prepares_sora_artifact_for_ai_studio(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "sora_model" / "model"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "sora-video-sample.onnx"
            model_path.write_bytes(b"sora")
            script = Path(__file__).resolve().parents[1] / "run_model.py"

            result = subprocess.run(
                [sys.executable, str(script), "--model", str(model_path), "--prepare-only"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[1/6] AI Studio process files check", result.stdout)
            self.assertIn("[4/6] Resolve and prepare local model", result.stdout)
            self.assertIn("local model prepared", result.stdout)
            self.assertTrue((root / "saved_model" / "local_model.onnx").exists())
            self.assertTrue((root / "ai_studio.env").exists())
            self.assertTrue((root / "config.json").exists())
            self.assertTrue((root / "input_example.json").exists())
            self.assertTrue((root / "requirements.txt").exists())
            self.assertTrue((root / "aiu_custom" / "model_wrapper.py").exists())
            self.assertTrue((root / "mlflow_ai_studio_logging.py").exists())
            source = script.read_text(encoding="utf-8")
            self.assertIn("AI Studio", source)
            self.assertIn("MLFLOW_TRACKING_URL", source)
            self.assertIn("Windows 10/11", source)

    def test_root_run_model_setup_env_screen_saves_mlflow_values(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = Path(__file__).resolve().parents[1] / "run_model.py"
            user_input = "\n".join(
                [
                    "http://mlflow.local:5000",
                    "ai-user",
                    "ai-pass",
                    "sora-exp",
                    "sora-model",
                ]
            ) + "\n"

            result = subprocess.run(
                [sys.executable, str(script), "--setup-env"],
                cwd=root,
                input=user_input,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("AI Studio MLflow 환경설정", result.stdout)
            env_source = (root / "ai_studio.env").read_text(encoding="utf-8")
            self.assertIn("MLFLOW_TRACKING_URL=http://mlflow.local:5000", env_source)
            self.assertIn("MLFLOW_TRACKING_USERNAME=ai-user", env_source)
            self.assertIn("MLFLOW_TRACKING_PASSWORD=ai-pass", env_source)
            self.assertIn("MLFLOW_EXPERIMENT_NAME=sora-exp", env_source)
            self.assertIn("MLFLOW_REGISTER_MODEL_NAME=sora-model", env_source)
            source = script.read_text(encoding="utf-8")
            self.assertIn('os.environ["MLFLOW_TRACKING_URI"] = tracking_url', source)

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

        self.assertIn("Sora 오류 샘플 생성", intro)

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
            self.assertIn("issues=1", message)

    def test_beginner_sample_run_prepares_multiple_models_and_selects_first_success(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                path, message = resolve_beginner_project_input("/sample run --kind all")
            finally:
                os.chdir(cwd)

            self.assertIsNotNone(message)
            self.assertTrue(Path(path).exists())
            self.assertIn("샘플 모델 실행을 완료했습니다", message)
            self.assertIn("성공", message)
            self.assertIn("초급자 Wizard는 첫 번째 성공 샘플 경로", message)
            self.assertTrue((Path(path) / "saved_model").exists())

    def test_beginner_tui_sample_run_is_available_after_launch_mode_selection(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                controller = BeginnerTuiController("")
                controller.submit("1")
                output = controller.submit("/sample run --kind tensorflow --register --dry-run")
            finally:
                os.chdir(cwd)

            self.assertIn("Step 1. 프로젝트 선택", output)
            self.assertIn("샘플 모델 실행을 완료했습니다", controller.latest_message)
            self.assertIn("tensorflow-model", controller.latest_message)
            self.assertTrue(Path(controller.project_path).exists())

    def test_beginner_intro_lists_sample_run_commands(self):
        intro = build_beginner_intro()

        self.assertIn("1. 여러 기본 샘플 모델 생성 후 실행", intro)
        self.assertIn("2. 로컬 학습 가능한 표준 PyTorch 샘플 실행", intro)

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
            self.assertNotIn("train.py", analysis.entrypoint_candidates)
            self.assertFalse(analysis.entrypoint_candidates)
            self.assertTrue(any(issue.code == "MLFLOW_DEPENDENCY_MISSING" for issue in analysis.issue_details))


class IntermediateModeTest(unittest.TestCase):
    def test_mlflow_request_gets_mlflow_guidance(self):
        output = handle_intermediate_request("MLflow 설정만 확인해줘")

        self.assertIn("MLflow 설정 검증", output)
        self.assertIn("mlflow-validator", output)
        self.assertIn("dry-run", output)
        self.assertIn("run_model.py 설정 변수", output)
        self.assertIn("ModelWrapper 구현", output)
        self.assertIn("mlflow.pyfunc.log_model", output)
        self.assertIn("requirements.txt serve/mlflow", output)
        self.assertIn("환경변수", output)

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
            (root / "requirements.txt").write_text("mlflow==2.17.0\nfastapi\nuvicorn\n")
            (root / "train.py").write_text(
                "import mlflow\n"
                "if __name__ == \"__main__\":\n"
                "    mlflow.log_param('x', 1)\n"
            )
            ensure_ai_studio_sample_runtime(root)
            (root / "model.onnx").write_text("sample")

            output = handle_advanced_input(f"ml-agent validate {root} --json")
            payload = json.loads(output)

            self.assertEqual(payload["exit_code"], 0)
            self.assertEqual(payload["analysis"]["registration_status"], "등록 가능")
            self.assertTrue(payload["analysis"]["job_template_ready"])
            checks = {check["code"]: check for check in payload["analysis"]["registration_checks"]}
            for code in (
                "project_path",
                "dependencies",
                "mlflow_dependency",
                "mlflow_code",
                "entrypoint",
                "model_artifact",
                "job_template",
                "run_model_config",
                "model_wrapper",
                "pyfunc_log_model",
                "serving_requirements",
            ):
                self.assertEqual(checks[code]["status"], "pass", code)
            self.assertEqual(checks["mlflow_environment"]["status"], "warn")
            self.assertIn("file:./mlruns", checks["mlflow_environment"]["detail"])

    def test_mlflow_config_checks_include_requested_items(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\nfastapi\nuvicorn\n")
            (root / "train.py").write_text("import mlflow\n")
            ensure_ai_studio_sample_runtime(root)
            (root / "model.onnx").write_text("sample")

            analysis = analyze_project(str(root))
            checks = {check.code: check for check in analysis.registration_checks}

            self.assertIn("run_model_config", checks)
            self.assertIn("model_wrapper", checks)
            self.assertIn("pyfunc_log_model", checks)
            self.assertIn("serving_requirements", checks)
            self.assertIn("mlflow_environment", checks)
            self.assertEqual(checks["run_model_config"].status, "pass")
            self.assertEqual(checks["model_wrapper"].status, "pass")
            self.assertEqual(checks["pyfunc_log_model"].status, "pass")
            self.assertEqual(checks["serving_requirements"].status, "pass")

    def test_predict_py_model_wrapper_satisfies_wrapper_check(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\n")
            (root / "run_model.py").write_text("--env-file --config --model --mode --register AI_STUDIO_CONFIG_PATH AI_STUDIO_LOCAL_MODEL_PATH")
            (root / "predict.py").write_text(
                "import mlflow.pyfunc\n\n"
                "class ModelWrapper(mlflow.pyfunc.PythonModel):\n"
                "    def load_context(self, context):\n"
                "        pass\n"
                "    def predict(self, context, model_input):\n"
                "        return model_input\n"
            )
            (root / "mlflow_ai_studio_logging.py").write_text("mlflow.pyfunc.log_model(python_model=ModelWrapper())\n")
            (root / "model.onnx").write_text("sample")

            analysis = analyze_project(str(root))
            wrapper = next(check for check in analysis.registration_checks if check.code == "model_wrapper")

            self.assertEqual(wrapper.status, "pass")
            self.assertIn("predict.py", wrapper.detail)

    def test_pyfunc_log_model_missing_parameters_warns(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\n")
            ensure_ai_studio_sample_runtime(root)
            (root / "mlflow_ai_studio_logging.py").write_text(
                "import mlflow\n"
                "from aiu_custom import ModelWrapper\n"
                "mlflow.pyfunc.log_model(python_model=ModelWrapper())\n"
            )
            (root / "model.onnx").write_text("sample")

            analysis = analyze_project(str(root))
            check = next(check for check in analysis.registration_checks if check.code == "pyfunc_log_model")

            self.assertEqual(check.status, "warn")
            self.assertIn("registered_model_name", check.detail)
            self.assertIn("pip_requirements", check.detail)

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

    def test_sample_list_outputs_basic_and_large_catalog(self):
        output = handle_advanced_input("aiu sample list --json")
        payload = json.loads(output)

        self.assertIn("sample_root", payload)
        self.assertGreaterEqual(len(payload["basic"]), 6)
        self.assertEqual(len(payload["large10"]), 10)
        basic_kinds = {item["kind"] for item in payload["basic"]}
        large_kinds = {item["kind"] for item in payload["large10"]}
        self.assertIn("tensorflow", basic_kinds)
        self.assertIn("pytorch", basic_kinds)
        self.assertIn("large_tabular_ensemble", large_kinds)
        self.assertIn("large_llm_adapter", large_kinds)

    def test_sample_create_tensorflow_generates_register_scaffold(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                output = handle_advanced_input("aiu sample create --kind tensorflow --json")
            finally:
                os.chdir(cwd)

            payload = json.loads(output)
            sample = Path(payload["created"][0])

            self.assertEqual(payload["status"], "ok")
            self.assertTrue((sample / "run_model.py").exists())
            self.assertTrue((sample / "config.json").exists())
            self.assertTrue((sample / "ai_studio.env").exists())
            self.assertTrue((sample / "input_example.json").exists())
            self.assertIn("--register", (sample / "run_model.py").read_text(encoding="utf-8"))

    def test_sample_run_all_prepares_multiple_models(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                output = handle_advanced_input("aiu sample run --kind all --json")
            finally:
                os.chdir(cwd)

            payload = json.loads(output)

            self.assertEqual(payload["command"], "sample run")
            self.assertEqual(payload["status"], "ok")
            self.assertGreaterEqual(payload["count"], 6)
            self.assertEqual(payload["pass_count"], payload["count"])
            self.assertEqual(payload["fail_count"], 0)
            self.assertTrue(all(result["status"] == "pass" for result in payload["results"]))
            self.assertTrue(all("local model prepared" in result["stdout"] for result in payload["results"]))

    def test_sample_run_register_dry_run_for_single_model(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                output = handle_advanced_input("aiu sample run --kind tensorflow --register --dry-run --json")
            finally:
                os.chdir(cwd)

            payload = json.loads(output)
            result = payload["results"][0]

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["run_mode"], "register")
            self.assertTrue(payload["dry_run"])
            self.assertEqual(result["status"], "pass")
            self.assertIn("--register --dry-run", result["command"])
            self.assertIn("dry-run register command", result["stdout"])

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
            self.assertEqual(payload["fix_previews"][1]["code"], "CREATE_AI_STUDIO_MLFLOW_SCAFFOLD")

    def test_fix_json_contains_step6_approval_options(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = handle_advanced_input(f"ml-agent fix {root} --dry-run --json")
            payload = json.loads(output)

            self.assertIn("approval_required=true", payload["details"])
            self.assertEqual(payload["approval_options"][0]["key"], "apply")
            self.assertEqual(payload["approval_options"][0]["label"], "승인 후 생성/수정")
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
            self.assertEqual(train.read_text(), "print('train')\n")
            self.assertTrue((root / "config.json").exists())
            self.assertTrue((root / "ai_studio.env").exists())
            self.assertTrue((root / "input_example.json").exists())
            self.assertTrue((root / "aiu_custom" / "model_wrapper.py").exists())
            self.assertTrue((root / "mlflow_ai_studio_logging.py").exists())
            self.assertTrue((root / "run_model.py").exists())
            self.assertEqual(untouched.read_text(), "hello\n")

    def test_apply_creates_ai_studio_mlflow_scaffold(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            requirements.write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")
            (root / "model.pkl").write_text("sample")

            output = handle_advanced_input(f"ml-agent apply {root} --json")
            payload = json.loads(output)

            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            logging_source = (root / "mlflow_ai_studio_logging.py").read_text(encoding="utf-8")
            generated_run_model = (root / "run_model.py").read_text(encoding="utf-8")
            env_source = (root / "ai_studio.env").read_text(encoding="utf-8")
            wrapper_source = (root / "aiu_custom" / "predict.py").read_text(encoding="utf-8")
            wrapper_alias_source = (root / "aiu_custom" / "model_wrapper.py").read_text(encoding="utf-8")
            requirements_text = requirements.read_text(encoding="utf-8")
            self.assertEqual(config["mlflow_tracking_url"], "${MLFLOW_TRACKING_URL}")
            self.assertEqual(config["model"]["source_path"], "")
            self.assertEqual(config["model"]["artifact_path"], "ai_studio")
            self.assertEqual(config["execution"]["entrypoint"], "run_model.py")
            self.assertIn("train.py", config["execution"]["blocked_entrypoints"])
            self.assertIn("AI_STUDIO_LOCAL_MODEL_PATH", logging_source)
            self.assertIn("mlflow.pyfunc.log_model", logging_source)
            self.assertIn('os.getenv("AI_STUDIO_CONFIG_PATH"', logging_source)
            self.assertIn("python_model=ModelWrapper()", logging_source)
            self.assertIn("registered_model_name=registered_model_name", logging_source)
            self.assertIn("prepare_local_model", generated_run_model)
            self.assertIn("local model prepared", generated_run_model)
            self.assertIn("train_sklearn_diabetes_model", generated_run_model)
            self.assertIn("load_diabetes", generated_run_model)
            self.assertIn("ElasticNet(alpha=0.001, l1_ratio=0.5", generated_run_model)
            self.assertIn("mlflow.data.from_pandas", generated_run_model)
            self.assertIn("mlflow.pyfunc.log_model", generated_run_model)
            self.assertIn("from aiu_custom.predict import ModelWrapper", generated_run_model)
            self.assertIn("MLFLOW_TRACKING_URL=", env_source)
            self.assertIn("class ModelWrapper", wrapper_source)
            self.assertIn("from .predict import ModelWrapper", wrapper_alias_source)
            self.assertIn("cloudpickle", requirements_text)
            self.assertIn("scikit-learn", requirements_text)
            self.assertIn("joblib", requirements_text)
            self.assertIn("fastapi", requirements_text)
            self.assertIn("uvicorn", requirements_text)
            self.assertTrue((root / "serving_app.py").exists())
            self.assertIn("FastAPI", (root / "serving_app.py").read_text(encoding="utf-8"))
            self.assertTrue(any(change["code"] == "CREATE_AI_STUDIO_MLFLOW_SCAFFOLD" for change in payload["applied_changes"]))

            result = subprocess.run(
                [sys.executable, "run_model.py", "--prepare-only"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "saved_model" / "local_model.pkl").exists())
            self.assertIn("local model prepared", result.stdout)

    def test_register_dry_run_writes_local_mlflow_registration_plan(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            requirements.write_text("mlflow\nscikit-learn\n")
            (root / "train.py").write_text("import mlflow\n")
            (root / "model.pkl").write_text("sample")
            ensure_ai_studio_sample_runtime(root)

            output = handle_advanced_input(f"aiu register {root} --dry-run --json")
            payload = json.loads(output)
            plan_path = root / "mlflow-registration-plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["command"], "register")
            self.assertEqual(payload["status"], "ok")
            self.assertTrue(plan_path.exists())
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["tracking_mode"], "local-file-store")
            self.assertEqual(plan["tracking_uri"], "file:./mlruns")
            self.assertIn("python run_model.py --env-file ai_studio.env --register --dry-run", plan["command"])
            self.assertIn("register_command=python run_model.py --env-file ai_studio.env --register --dry-run", payload["details"])

    def test_run_model_register_dry_run_prepares_model_without_mlflow_import(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "run_model.py").write_text(run_model_source(), encoding="utf-8")
            (root / "config.json").write_text(json.dumps({"model": {"save_path": "saved_model"}}), encoding="utf-8")
            source = root / "my_model.onnx"
            source.write_text("model", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "run_model.py", "--model", str(source), "--register", "--dry-run"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "saved_model" / "local_model.onnx").exists())
            self.assertTrue((root / "mlruns").exists())
            self.assertIn("dry-run register command", result.stdout)
            self.assertIn("file:./mlruns", result.stdout)
            self.assertIn("local mlruns directory", result.stdout)

    def test_run_model_source_maps_empty_tracking_url_to_root_mlruns(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "run_model.py").write_text(run_model_source(), encoding="utf-8")
            (root / "config.json").write_text(json.dumps({"model": {"save_path": "saved_model"}}), encoding="utf-8")
            (root / "ai_studio.env").write_text("MLFLOW_TRACKING_URL=\n", encoding="utf-8")
            source = root / "my_model.onnx"
            source.write_text("model", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "run_model.py", "--env-file", "ai_studio.env", "--model", str(source), "--register", "--dry-run"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "mlruns").is_dir())
            self.assertIn("mlflow tracking default: file:./mlruns", result.stdout)

    def test_run_model_source_prepares_explicit_local_model_without_mlflow(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "run_model.py").write_text(run_model_source(), encoding="utf-8")
            (root / "config.json").write_text(json.dumps({"model": {"save_path": "saved_model"}}), encoding="utf-8")
            source = root / "my_model.onnx"
            source.write_text("model", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "run_model.py", "--model", str(source), "--prepare-only"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "saved_model" / "local_model.onnx").exists())
            self.assertEqual((root / "saved_model" / "local_model.onnx").read_text(encoding="utf-8"), "model")

    def test_standard_ml_dl_template_creates_expected_structure(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            change = ensure_standard_ml_dl_template(root, "pytorch")

            self.assertEqual(change.status, "applied")
            self.assertTrue((root / "aiu_custom" / "model_wrapper.py").exists())
            self.assertTrue((root / "config" / "model_config.json").exists())
            self.assertTrue((root / "config" / "train_config.json").exists())
            self.assertTrue((root / "config" / "mlflow_config.json").exists())
            self.assertTrue((root / "saved_model").is_dir())
            self.assertTrue((root / "run_model.py").exists())
            self.assertTrue((root / "train.py").exists())
            self.assertIn("torch", (root / "requirements.txt").read_text(encoding="utf-8"))

    def test_run_model_source_supports_train_mode_prepare_only(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_standard_ml_dl_template(root, "sklearn")

            result = subprocess.run(
                [sys.executable, "run_model.py", "--mode", "train", "--prepare-only"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("trained model artifact", result.stdout)
            self.assertIn("local model prepared", result.stdout)
            self.assertTrue(any((root / "saved_model").glob("local_model*.joblib")))

    def test_sample_create_standard_template_json(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                output = handle_advanced_input("aiu sample create --kind standard --framework onnx --json")
            finally:
                os.chdir(cwd)

            payload = json.loads(output)
            created = Path(payload["created"][0])
            self.assertEqual(payload["status"], "ok")
            self.assertTrue((created / "run_model.py").exists())
            self.assertTrue((created / "config" / "model_config.json").exists())
            self.assertIn("onnx", (created / "requirements.txt").read_text(encoding="utf-8"))

    def test_verify_run_without_mlflow_writes_failure_report(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_ai_studio_sample_runtime(root)
            (root / "requirements.txt").write_text("mlflow\n")
            (root / "model.onnx").write_text("sample")

            output = handle_advanced_input(f"aiu verify-run {root} --skip-serving --json")
            payload = json.loads(output)
            report = json.loads((root / "mlflow-run-verification.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["command"], "verify-run")
            self.assertEqual(payload["status"], "error")
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["run_model"]["exit_code"], 1)
            self.assertIn("run_model.py 실행 실패", report["errors"])

    def test_verify_run_with_fake_mlflow_confirms_run_and_prediction(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_ai_studio_sample_runtime(root)
            (root / "requirements.txt").write_text("mlflow\ncloudpickle\npandas\n")
            (root / "model.onnx").write_text("sample")
            self.write_fake_mlflow_runtime(root)
            sys.path.insert(0, str(root))
            try:
                output = handle_advanced_input(f"aiu verify-run {root} --skip-serving --json")
            finally:
                sys.path.remove(str(root))
                for name in list(sys.modules):
                    if name == "mlflow" or name.startswith("mlflow.") or name == "pandas":
                        sys.modules.pop(name, None)

            payload = json.loads(output)
            report = json.loads((root / "mlflow-run-verification.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["run_model"]["exit_code"], 0)
            self.assertTrue(report["mlflow_run"]["run_id"])
            self.assertGreaterEqual(report["metric_report"]["param_count"], 1)
            self.assertEqual(report["model_test"]["status"], "pass")
            self.assertEqual(report["serving"]["status"], "skipped")

    def write_fake_mlflow_runtime(self, root: Path) -> None:
        mlflow_dir = root / "mlflow"
        mlflow_dir.mkdir()
        (root / "pandas.py").write_text(
            "class DataFrame:\n"
            "    def __init__(self, data, columns=None):\n"
            "        self.data = data\n"
            "        self.columns = columns or []\n"
            "    def __len__(self):\n"
            "        return len(self.data)\n"
            "    def to_dict(self, orient='dict'):\n"
            "        if orient == 'records':\n"
            "            return [dict(zip(self.columns, row)) for row in self.data]\n"
            "        return {'data': self.data, 'columns': self.columns}\n",
            encoding="utf-8",
        )
        (root / "joblib.py").write_text(
            "def dump(model, path):\n"
            "    open(path, 'w', encoding='utf-8').write('fake joblib model')\n"
            "def load(path):\n"
            "    class _Model:\n"
            "        def predict(self, model_input): return [0 for _ in range(len(model_input))]\n"
            "    return _Model()\n",
            encoding="utf-8",
        )
        (mlflow_dir / "__init__.py").write_text(
            "import os, shutil, time, uuid\n"
            "from pathlib import Path\n"
            "from . import pyfunc, tracking\n"
            "_tracking_uri = 'file:./mlruns'\n"
            "_experiment_name = 'default'\n"
            "def set_tracking_uri(uri):\n"
            "    global _tracking_uri; _tracking_uri = uri\n"
            "def set_experiment(name):\n"
            "    global _experiment_name; _experiment_name = name\n"
            "def _mlruns_root():\n"
            "    raw = _tracking_uri.replace('file:', '', 1)\n"
            "    return Path(raw)\n"
            "class _Run:\n"
            "    def __enter__(self):\n"
            "        run_id = uuid.uuid4().hex\n"
            "        run_dir = _mlruns_root() / '0' / run_id\n"
            "        (run_dir / 'params').mkdir(parents=True, exist_ok=True)\n"
            "        (run_dir / 'metrics').mkdir(parents=True, exist_ok=True)\n"
            "        (run_dir / 'tags').mkdir(parents=True, exist_ok=True)\n"
            "        (run_dir / 'artifacts').mkdir(parents=True, exist_ok=True)\n"
            "        (run_dir / 'meta.yaml').write_text('run_id: ' + run_id + '\\n', encoding='utf-8')\n"
            "        os.environ['FAKE_MLFLOW_RUN_DIR'] = str(run_dir)\n"
            "        return self\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        os.environ.pop('FAKE_MLFLOW_RUN_DIR', None)\n"
            "def start_run(): return _Run()\n"
            "def _run_dir(): return Path(os.environ['FAKE_MLFLOW_RUN_DIR'])\n"
            "def log_params(params):\n"
            "    for key, value in params.items(): (_run_dir() / 'params' / key).write_text(str(value), encoding='utf-8')\n"
            "def log_artifact(path):\n"
            "    target = _run_dir() / 'artifacts' / Path(path).name\n"
            "    shutil.copy2(path, target)\n"
            "def log_text(text, artifact_file):\n"
            "    target = _run_dir() / 'artifacts' / artifact_file\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            "    target.write_text(text, encoding='utf-8')\n",
            encoding="utf-8",
        )
        (mlflow_dir / "pyfunc.py").write_text(
            "import os\n"
            "from pathlib import Path\n"
            "class PythonModel: pass\n"
            "def log_model(**kwargs):\n"
            "    run_dir = Path(os.environ['FAKE_MLFLOW_RUN_DIR'])\n"
            "    artifact_path = kwargs.get('artifact_path', 'model')\n"
            "    target = run_dir / 'artifacts' / artifact_path\n"
            "    target.mkdir(parents=True, exist_ok=True)\n"
            "    (target / 'MLmodel').write_text('fake model', encoding='utf-8')\n"
            "class _LoadedModel:\n"
            "    def predict(self, model_input):\n"
            "        return model_input.to_dict(orient='records')\n"
            "def load_model(model_uri): return _LoadedModel()\n",
            encoding="utf-8",
        )
        (mlflow_dir / "tracking.py").write_text(
            "from pathlib import Path\n"
            "import mlflow\n"
            "class _Experiment:\n"
            "    experiment_id = '0'\n"
            "class _Info:\n"
            "    def __init__(self, run_dir):\n"
            "        self.run_id = run_dir.name; self.experiment_id = run_dir.parent.name\n"
            "        self.artifact_uri = str(run_dir / 'artifacts'); self.start_time = int(run_dir.stat().st_mtime * 1000)\n"
            "class _Data:\n"
            "    def __init__(self, run_dir):\n"
            "        self.params = {p.name: p.read_text(encoding='utf-8') for p in (run_dir / 'params').iterdir()}\n"
            "        self.metrics = {}\n"
            "        self.tags = {}\n"
            "class _Run:\n"
            "    def __init__(self, run_dir): self.info = _Info(run_dir); self.data = _Data(run_dir)\n"
            "class MlflowClient:\n"
            "    def search_experiments(self): return [_Experiment()]\n"
            "    def search_runs(self, experiment_ids, order_by=None, max_results=1):\n"
            "        root = Path(mlflow._tracking_uri.replace('file:', '', 1)) / '0'\n"
            "        runs = [p for p in root.iterdir() if (p / 'meta.yaml').exists()] if root.exists() else []\n"
            "        runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)\n"
            "        return [_Run(p) for p in runs[:max_results]]\n",
            encoding="utf-8",
        )

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
            ensure_ai_studio_sample_runtime(root)
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
            self.assertTrue((root / "serving_app.py").exists())

    def test_serve_command_writes_job_template_yml_with_serving_values(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\nfastapi\nuvicorn\nscikit-learn\n")
            (root / "train.py").write_text("import mlflow\n")
            ensure_ai_studio_sample_runtime(root)
            (root / "model.onnx").write_text("sample")

            output = handle_advanced_input(f"aiu serve {root} --json")
            payload = json.loads(output)
            template_path = root / "job_template.yml"
            template = template_path.read_text(encoding="utf-8")

            self.assertEqual(payload["status"], "ok")
            self.assertTrue(template_path.exists())
            self.assertIn(f"job_template_file={template_path}", payload["details"])
            self.assertIn("kind: JobTemplate", template)
            self.assertIn("command: \"python run_model.py --env-file ai_studio.env --register\"", template)
            self.assertIn("app: \"serving_app:app\"", template)
            self.assertIn("health_endpoint: \"http://127.0.0.1:8000/health\"", template)
            self.assertIn("predict_endpoint: \"http://127.0.0.1:8000/predict\"", template)
            self.assertIn("serve_command: \"python -m uvicorn serving_app:app --host 127.0.0.1 --port 8000\"", template)
            self.assertIn("local_store: ./mlruns", template)

    def test_serve_command_creates_serving_app_when_missing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\nfastapi\nuvicorn\n")
            (root / "train.py").write_text("import mlflow\n")
            ensure_ai_studio_sample_runtime(root)
            (root / "serving_app.py").unlink()
            (root / "model.onnx").write_text("sample")

            output = handle_advanced_input(f"aiu serve {root} --json")
            payload = json.loads(output)

            self.assertEqual(payload["status"], "ok")
            self.assertTrue((root / "serving_app.py").exists())
            self.assertIn("serving_app=applied", payload["details"])
            checks = {check["code"]: check for check in payload["analysis"]["local_serving"]["checks"]}
            self.assertEqual(checks["serving_app"]["status"], "pass")

    def test_verify_run_writes_job_template_with_serving_smoke_status(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_ai_studio_sample_runtime(root)
            (root / "requirements.txt").write_text("mlflow\ncloudpickle\npandas\n")
            (root / "model.onnx").write_text("sample")
            self.write_fake_mlflow_runtime(root)
            sys.path.insert(0, str(root))
            try:
                output = handle_advanced_input(f"aiu verify-run {root} --json")
            finally:
                sys.path.remove(str(root))
                for name in list(sys.modules):
                    if name == "mlflow" or name.startswith("mlflow.") or name == "pandas":
                        sys.modules.pop(name, None)

            payload = json.loads(output)
            template_path = root / "job_template.yml"
            template = template_path.read_text(encoding="utf-8")

            self.assertIn(f"job_template_file={template_path}", payload["details"])
            self.assertTrue(template_path.exists())
            self.assertIn("smoke_test:", template)
            self.assertIn("status: skipped", template)
            self.assertIn("fastapi serving dependency missing", template)

    def test_report_writes_final_analysis_file(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\nscikit-learn\n")
            (root / "train.py").write_text("import mlflow\n")
            ensure_ai_studio_sample_runtime(root)
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
    def beginner_tui(self, project_path: str = "", **kwargs):
        return BeginnerTuiController(project_path, selected_launch_mode=MODE_BEGINNER, **kwargs)

    def test_tui_subcommand_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["tui"])

        self.assertEqual(args.command, "tui")

    def test_sample_and_register_subcommands_are_registered(self):
        parser = build_parser()
        sample_args = parser.parse_args(["sample", "create", "--kind", "tensorflow", "--json"])
        sample_run_args = parser.parse_args(["sample", "run", "--kind", "all", "--register", "--dry-run", "--timeout", "30", "--json"])
        register_args = parser.parse_args(["register", ".", "--dry-run", "--json"])
        verify_args = parser.parse_args(["verify-run", ".", "--skip-serving", "--json"])

        self.assertEqual(sample_args.command, "sample")
        self.assertEqual(sample_args.action, "create")
        self.assertEqual(sample_args.kind, "tensorflow")
        self.assertTrue(sample_args.json)
        self.assertEqual(sample_run_args.action, "run")
        self.assertEqual(sample_run_args.kind, "all")
        self.assertTrue(sample_run_args.register)
        self.assertTrue(sample_run_args.dry_run)
        self.assertEqual(sample_run_args.timeout, 30)
        self.assertEqual(register_args.command, "register")
        self.assertTrue(register_args.dry_run)
        self.assertEqual(verify_args.command, "verify-run")
        self.assertTrue(verify_args.skip_serving)

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

    def test_read_write_directory_helper_creates_writable_folder(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "runtime" / "sessions"

            result = ensure_read_write_directory(target)

            self.assertEqual(result, target)
            self.assertTrue(target.exists())
            probe = target / "write-check.txt"
            probe.write_text("ok", encoding="utf-8")
            self.assertEqual(probe.read_text(encoding="utf-8"), "ok")

    def test_local_file_access_helper_keeps_existing_folder_writable(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "project"
            target.mkdir()

            result = ensure_local_file_access(target)

            self.assertEqual(result, target)
            probe = target / "access-check.txt"
            probe.write_text("ok", encoding="utf-8")
            self.assertEqual(probe.read_text(encoding="utf-8"), "ok")

    def test_windows_read_write_permission_uses_icacls_for_current_user(self):
        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "runtime"
            target.mkdir()
            completed = subprocess.CompletedProcess(args=[], returncode=0)
            with patch.dict(os.environ, {"USERNAME": "aiu-user"}, clear=False):
                with patch("deep_agent.app_config.subprocess.run", return_value=completed) as run:
                    success = grant_windows_read_write(target)

            self.assertTrue(success)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["icacls", str(target), "/grant"])
            self.assertIn("aiu-user:(OI)(CI)M", command)
            self.assertIn("/T", command)
            self.assertIn("/C", command)

    def test_tui_initial_screen_requires_launch_mode_selection(self):
        controller = BeginnerTuiController("")
        output = controller.render_log()

        self.assertIsNone(controller.selected_launch_mode)
        self.assertEqual(controller.project_path, "")
        self.assertIn("사용자 모드를 선택하세요", output)
        self.assertIn("1. 초급자 모드", output)
        self.assertIn("2. 중급자 모드", output)
        self.assertIn("3. 고급자 모드", output)

    def test_tui_help_screen_lists_core_commands(self):
        output = format_tui_help_screen(MODE_BEGINNER, "Chatbot", "/tmp/project", "qwen3.6")

        self.assertIn("HELP   AI ML Onboarding Console", output)
        self.assertIn("/help", output)
        self.assertIn("Enter", output)
        self.assertIn("Shift+Enter", output)
        self.assertIn("Ctrl+Enter", output)
        self.assertIn("/sample large10", output)
        self.assertIn("/open", output)
        self.assertIn("/folder", output)
        self.assertIn("/file", output)
        self.assertIn("/model qwen3.6", output)
        self.assertIn("PLAN | BUILD | [CHAT]", output)

    def test_tui_help_command_works_before_launch_mode_selection(self):
        controller = BeginnerTuiController("")

        output = controller.submit("/help")

        self.assertIn("HELP   AI ML Onboarding Console", output)
        self.assertIn("모드 미선택", output)
        self.assertIn("/mode beginner", output)
        self.assertFalse(controller.should_show_thinking("/help"))

    def test_tui_help_command_works_in_chat_mode_without_runtime_call(self):
        class FakeRuntime:
            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                raise AssertionError("help must not call runtime")

        controller = self.beginner_tui("", deepagents_runtime=FakeRuntime())
        controller.select_agent_mode("Chatbot")

        output = controller.submit("/help")

        self.assertIn("HELP   AI ML Onboarding Console", output)
        self.assertIn("chat", output)
        self.assertIn("/sample tensorflow", output)

    def test_tui_chatbot_dragged_folder_selects_project_without_runtime_call(self):
        class FakeRuntime:
            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                raise AssertionError("dragged folder path must not call runtime")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "AI ML" / "model"
            root.mkdir(parents=True)
            (root / "requirements.txt").write_text("mlflow\n")
            controller = self.beginner_tui("", deepagents_runtime=FakeRuntime())
            controller.select_agent_mode("Chatbot")

            output = controller.submit(f"cd '{root}'")

        self.assertEqual(controller.project_path, str(root.resolve()))
        self.assertIn("프로젝트 폴더를 선택했습니다", output)
        self.assertIn(str(root.resolve()), output)
        self.assertFalse(controller.should_show_thinking(f"cd '{root}'"))

    def test_tui_intermediate_chatbot_dragged_windows_folder_selects_project(self):
        old_value = os.environ.get("AIU_WINDOWS_DRIVE_C")
        with TemporaryDirectory() as tmpdir:
            drive_root = Path(tmpdir)
            project = drive_root / "Users" / "choi" / "AI ML" / "model"
            project.mkdir(parents=True)
            (project / "requirements.txt").write_text("mlflow\n")
            os.environ["AIU_WINDOWS_DRIVE_C"] = str(drive_root)
            try:
                controller = BeginnerTuiController("")
                controller.submit("2")

                output = controller.submit(r"cd 'C:\Users\choi\AI ML\model'")
            finally:
                if old_value is None:
                    os.environ.pop("AIU_WINDOWS_DRIVE_C", None)
                else:
                    os.environ["AIU_WINDOWS_DRIVE_C"] = old_value

        self.assertEqual(controller.project_path, str(project.resolve()))
        self.assertIn("프로젝트 폴더를 선택했습니다", output)
        self.assertFalse(controller.should_show_thinking(r"cd 'C:\Users\choi\AI ML\model'"))

    def test_tui_does_not_accept_project_path_before_launch_mode(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            controller = BeginnerTuiController("")

            output = controller.submit(str(root))

            self.assertIsNone(controller.selected_launch_mode)
            self.assertEqual(controller.project_path, "")
            self.assertIn("먼저 모드를 선택하세요", output)

    def test_tui_launch_mode_selects_beginner_aliases(self):
        for value in ["1", "초급자", "beginner"]:
            controller = BeginnerTuiController("")
            output = controller.submit(value)

            self.assertEqual(controller.selected_launch_mode, MODE_BEGINNER)
            self.assertEqual(controller.agent_mode, "Plan")
            self.assertIn("Step 1. 프로젝트 선택", output)

    def test_tui_launch_mode_selects_intermediate_aliases(self):
        for value in ["2", "미드", "중급자", "intermediate"]:
            controller = BeginnerTuiController("")
            output = controller.submit(value)

            self.assertEqual(controller.selected_launch_mode, MODE_INTERMEDIATE)
            self.assertEqual(controller.agent_mode, "Chatbot")
            self.assertIn("CHAT MODE", output)
            self.assertIn("실행 모드: 중급자 모드", output)
            self.assertIn("Agent 모델: qwen3.6", output)

    def test_tui_launch_mode_selects_advanced_aliases(self):
        for value in ["3", "고급자", "advanced"]:
            controller = BeginnerTuiController("")
            output = controller.submit(value)

            self.assertEqual(controller.selected_launch_mode, MODE_ADVANCED)
            self.assertEqual(controller.agent_mode, "Build")
            self.assertIn("사용 가능한 명령어", output)
            self.assertIn("analyze", output)

    def test_tui_intermediate_mode_handles_chat_input(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                controller = BeginnerTuiController("")
                controller.submit("2")

                output = controller.submit("MLflow 설정만 확인해줘")
            finally:
                os.chdir(cwd)

        self.assertIn("DeepAgents runtime", output)
        self.assertIn("CHAT MODE", output)

    def test_tui_intermediate_chatbot_routes_through_runtime(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("중급자 Chatbot 응답", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir)
                (root / "requirements.txt").write_text("mlflow\n")
                (root / "train.py").write_text("import mlflow\n")
                (root / "model.onnx").write_text("sample")
                fake = FakeRuntime()
                controller = BeginnerTuiController("", deepagents_runtime=fake)
                controller.submit("2")
                controller.project_path = str(root)

                output = controller.submit("프로젝트 분석해줘")
            finally:
                os.chdir(cwd)

            self.assertIn("중급자 Chatbot 응답", output)
            self.assertEqual(fake.calls[-1], ("프로젝트 분석해줘", str(root), "AutoFix"))

    def test_tui_extracts_agent_response_numbered_choices(self):
        response = "다음 중 선택하세요.\n1. MLflow 설정 점검\n2) run_model.py 생성\n- [3] 로컬 서빙 테스트"

        choices = extract_agent_response_choices(response)

        self.assertEqual(choices, ["MLflow 설정 점검", "run_model.py 생성", "로컬 서빙 테스트"])
        self.assertIn("1. MLflow 설정 점검", format_agent_response_choices(choices))
        self.assertIn("1-3", agent_response_choice_placeholder(choices))

    def test_tui_chatbot_allows_selecting_agent_response_choice(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                if len(self.calls) == 1:
                    return DeepAgentsRunResult("선택하세요.\n1. MLflow 설정 점검\n2. run_model.py 생성", True)
                return DeepAgentsRunResult("선택 항목 실행 완료", True)

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
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                first = controller.submit("검증 항목 보여줘")
                second = controller.submit("2")
            finally:
                os.chdir(cwd)

        self.assertIn("선택하세요", first)
        self.assertTrue(controller.awaiting_agent_response_choice is False)
        self.assertIn("선택 항목 실행 완료", second)
        self.assertIn("선택 항목: run_model.py 생성", fake.calls[-1][0])
        self.assertIn("Agent 응답 선택", controller.render_log())

    def test_tui_agent_response_choice_prepare_clears_selection_before_worker(self):
        controller = self.beginner_tui("")
        controller.select_agent_mode("Chatbot")
        controller._capture_agent_response_choices("1. MLflow 설정 점검\n2. run_model.py 생성")

        prompt, display = controller.prepare_agent_response_choice_submission("2")

        self.assertFalse(controller.awaiting_agent_response_choice)
        self.assertEqual(controller.agent_response_choices, [])
        self.assertEqual(display, "선택: run_model.py 생성")
        self.assertIn("선택 항목: run_model.py 생성", prompt)

    def test_tui_submit_queue_uses_agent_choice_prepare_before_chat_worker(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertIn("prepare_agent_response_choice_submission(value)", source)
        self.assertIn("display_value = value", source)
        self.assertIn("submit_value = value", source)
        self.assertIn("self.controller.set_thinking(display_value)", source)
        self.assertIn("self._start_submit_worker(submit_value, request_id)", source)

    def test_tui_chatbot_direct_message_clears_agent_response_choice(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                if len(self.calls) == 1:
                    return DeepAgentsRunResult("1. A안\n2. B안", True)
                return DeepAgentsRunResult("다시 요청 처리", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                controller = self.beginner_tui("", deepagents_runtime=FakeRuntime())
                controller.select_agent_mode("Chatbot")

                controller.submit("선택지 줘")
                output = controller.submit("원하는 항목이 없어서 다른 방식으로 해줘")
            finally:
                os.chdir(cwd)

        self.assertIn("다시 요청 처리", output)
        self.assertFalse(controller.awaiting_agent_response_choice)

    def test_tui_direct_agent_mode_words_switch_modes(self):
        controller = self.beginner_tui("")

        controller.submit("chatbot")
        self.assertEqual(controller.agent_mode, "Chatbot")

        controller.submit("build")
        self.assertEqual(controller.agent_mode, "Build")

        controller.submit("plan")
        self.assertEqual(controller.agent_mode, "Plan")

    def test_tui_intermediate_mode_answers_greeting(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                controller = BeginnerTuiController("")
                controller.submit("2")

                output = controller.submit("하이")
            finally:
                os.chdir(cwd)

        self.assertIn("안녕하세요", output)
        self.assertIn("AI ML 온보딩 Agent", output)

    def test_tui_chatbot_input_sets_thinking_before_submit(self):
        controller = BeginnerTuiController("")
        controller.submit("2")

        self.assertTrue(controller.should_show_thinking("하이"))
        controller.set_thinking("하이")

        self.assertIn("AI thinking", controller.render_log())
        self.assertIn("0s", controller.render_log())
        self.assertIn("YOU    하이", controller.render_log())
        self.assertIn("AGENT  response", controller.render_log())

    def test_tui_thinking_animation_shows_elapsed_seconds(self):
        self.assertIn("AI thinking [=", format_thinking_animation(0))
        self.assertIn("0s", format_thinking_animation(0))
        self.assertIn("5s", format_thinking_animation(5))

        controller = BeginnerTuiController("")
        controller.submit("2")
        controller.set_thinking("하이", elapsed_seconds=3)

        self.assertIn("AI thinking", controller.render_log())
        self.assertIn("3s", controller.render_log())

    def test_chat_card_uses_clean_rich_sections(self):
        rendered = format_chat_card("모델 검증해줘", "검증 결과입니다.", [])

        self.assertIn("YOU    모델 검증해줘", rendered)
        self.assertIn("AGENT  response", rendered)
        self.assertIn("         검증 결과입니다.", rendered)
        self.assertIn("----------------------------------------------------------------------------", rendered)
        self.assertNotIn("+----------------", rendered)

    def test_tui_truncates_long_chat_response_for_scroll_performance(self):
        long_text = "\n".join(f"line {index}" for index in range(250))

        rendered = format_chat_card("긴 응답", long_text, [])

        self.assertIn("화면 성능을 위해 일부 응답을 접었습니다", rendered)
        self.assertIn("전체 내용은 sessions/wiki 로그에 저장", rendered)
        self.assertIn("line 179", rendered)
        self.assertNotIn("line 249", rendered)

    def test_tui_prunes_chat_log_entries_for_scroll_performance(self):
        controller = self.beginner_tui("")
        controller.select_agent_mode("Chatbot")

        for index in range(20):
            controller._append_chat_log(f"질문 {index}", f"응답 {index}", [])

        self.assertLessEqual(len(controller.log_lines), 8)
        self.assertIn("질문 19", controller.render_log())
        self.assertNotIn("질문 0", controller.render_log())

    def test_tui_render_trim_keeps_screen_under_budget(self):
        clipped = truncate_for_tui("x" * 9000, max_chars=1000, max_lines=1000)

        self.assertIn("화면 성능", clipped)
        self.assertLess(len(clipped), 1300)

    def test_tui_chatbot_response_replaces_thinking_log_entry(self):
        class FakeRuntime:
            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                return DeepAgentsRunResult("완료 응답", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir) / "project"
                root.mkdir()
                (root / "requirements.txt").write_text("mlflow\n")
                (root / "train.py").write_text("import mlflow\n")
                (root / "model.onnx").write_text("sample")
                controller = self.beginner_tui(str(root), deepagents_runtime=FakeRuntime())
                controller.select_agent_mode("Chatbot")
                controller.set_thinking("분석해줘")

                controller.submit("분석해줘")
            finally:
                os.chdir(cwd)

            rendered = controller.render_log()
            self.assertIn("YOU    분석해줘", rendered)
            self.assertIn("AGENT  response", rendered)
            self.assertIn("완료 응답", rendered)
            self.assertNotIn("AI thinking", rendered)

    def test_tui_chatbot_error_is_visible_in_log(self):
        controller = BeginnerTuiController("")
        controller.submit("2")
        controller.set_thinking("분석해줘")

        controller._append_chat_error("분석해줘", RuntimeError("boom"))

        rendered = controller.render_log()
        self.assertIn("YOU    분석해줘", rendered)
        self.assertIn("AGENT  response", rendered)
        self.assertIn("Chatbot 응답 처리 중 오류가 발생했습니다: boom", rendered)

    def test_tui_chatbot_worker_always_releases_busy_state(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertIn("def _submit_value_in_thread", source)
        self.assertIn("def _start_thinking_animation", source)
        self.assertIn("def _stop_thinking_animation", source)
        self.assertIn("def _tick_thinking_animation", source)
        self.assertIn("self._stop_thinking_animation()", source)
        self.assertIn("except BaseException as exc", source)
        self.assertIn("finally:", source)
        self.assertIn("self.call_from_thread(lambda: self._finish_submit(request_id, value))", source)
        self.assertIn("self._chatbot_busy = False", source)

    def test_tui_thinking_is_only_for_chatbot_text(self):
        controller = BeginnerTuiController("")
        self.assertFalse(controller.should_show_thinking("하이"))

        controller.submit("1")
        self.assertFalse(controller.should_show_thinking("/path /tmp/project"))
        self.assertFalse(controller.should_show_thinking("/model"))
        self.assertFalse(controller.should_show_thinking("다음"))

        controller.select_agent_mode("Chatbot")
        self.assertTrue(controller.should_show_thinking("프로젝트 분석해줘"))

        advanced = BeginnerTuiController("")
        advanced.submit("3")
        self.assertFalse(advanced.should_show_thinking("analyze ."))
        advanced.submit("chatbot")
        self.assertTrue(advanced.should_show_thinking("하이"))

    def test_tui_advanced_mode_handles_cli_style_input(self):
        controller = BeginnerTuiController("")
        controller.submit("3")

        output = controller.submit("analyze .")

        self.assertIn("analyze:", output)
        self.assertIn("exit_code:", output)
        self.assertIn("사용 가능한 명령어", output)

    def test_tui_advanced_chatbot_mode_handles_chat_input(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                controller = BeginnerTuiController("")
                controller.submit("3")
                controller.submit("chatbot")

                output = controller.submit("하이")
            finally:
                os.chdir(cwd)

        self.assertIn("안녕하세요", output)
        self.assertNotIn("unknown command", output.lower())

    def test_tui_chatbot_screen_shows_project_model_info(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "sample.onnx").write_bytes(b"1234")
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model": {"source_path": "model/sample.onnx", "save_path": "saved_model"},
                        "training": {"epochs": 3, "learning_rate": 0.01, "batch_size": 8, "optimizer": "Adam"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            model_info = format_tui_model_info(str(root))
            output = format_tui_chatbot_screen(str(root), "qwen3.6", MODE_BEGINNER)

            self.assertIn("- 프로젝트 모델: model/sample.onnx", model_info)
            self.assertIn("- 모델 크기: 4 B", output)
            self.assertIn("- 모델 후보: 1개", output)
            self.assertIn("- 모델 파라미터:", output)
            self.assertIn("training.epochs=3", output)
            self.assertIn("training.learning_rate=0.01", output)
            self.assertIn("training.batch_size=8", output)
            self.assertIn("training.optimizer=Adam", output)
            self.assertIn("- Agent 모델: qwen3.6", output)

    def test_tui_chatbot_controller_render_shows_selected_project_model_info(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\n")
            (root / "model.onnx").write_bytes(b"model")
            controller = self.beginner_tui(str(root))

            output = controller.submit("chatbot")

            self.assertIn("CHAT MODE", output)
            self.assertIn("프로젝트 모델: model.onnx", output)
            self.assertIn("모델 크기: 5 B", output)

    def test_analyze_project_loads_model_parameter_values(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("mlflow\n")
            (root / "run_model.py").write_text("print('run')\n")
            (root / "mlflow_ai_studio_logging.py").write_text("import mlflow\n")
            (root / "model.onnx").write_bytes(b"12345")
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "source_path": "model.onnx",
                            "save_path": "saved_model",
                            "artifact_path": "ai_studio",
                        },
                        "training": {
                            "epochs": 5,
                            "learning_rate": 0.001,
                            "batch_size": 64,
                            "optimizer": "SGD",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            analysis = analyze_project(str(root))
            payload = analysis.as_dict()

            self.assertEqual(analysis.model_parameters["model.source_path"], "model.onnx")
            self.assertEqual(analysis.model_parameters["training.epochs"], 5)
            self.assertEqual(analysis.model_parameters["training.learning_rate"], 0.001)
            self.assertEqual(analysis.model_parameters["artifact.primary_path"], "model.onnx")
            self.assertEqual(analysis.model_parameters["artifact.primary_size"], "5 B")
            self.assertEqual(payload["model_parameters"]["training.batch_size"], 64)

    def test_tui_command_placeholder_shows_active_agent_mode(self):
        self.assertEqual(command_placeholder_for_mode("Plan"), "")
        self.assertEqual(command_placeholder_for_mode("Build"), "")
        self.assertEqual(command_placeholder_for_mode("Chatbot"), "")

    def test_tui_agent_mode_selector_and_command_parser(self):
        self.assertEqual(format_agent_mode_selector("Plan"), "[PLAN] | BUILD | CHAT")
        self.assertEqual(format_agent_mode_selector("Build"), "PLAN | [BUILD] | CHAT")
        self.assertEqual(format_agent_mode_selector("Chatbot"), "PLAN | BUILD | [CHAT]")
        self.assertEqual(parse_agent_mode_command("/agent plan"), "Plan")
        self.assertEqual(parse_agent_mode_command("/agent build"), "Build")
        self.assertEqual(parse_agent_mode_command("/agent chat"), "Chatbot")
        self.assertEqual(parse_agent_mode_command("plan"), "Plan")
        self.assertEqual(parse_agent_mode_command("build"), "Build")
        self.assertEqual(parse_agent_mode_command("chatbot"), "Chatbot")
        self.assertEqual(parse_agent_mode_command("쳇봇모드"), "Chatbot")
        self.assertEqual(parse_agent_mode_command("/에이전트 챗봇"), "Chatbot")
        self.assertEqual(parse_agent_mode_command("/agent"), "")
        self.assertIsNone(parse_agent_mode_command("agent chat"))

    def test_tui_chat_autofix_classifier_keeps_general_questions_fast(self):
        self.assertFalse(should_use_autofix_chat("PLAN BUILD CHAT 차이를 알려줘"))
        self.assertTrue(should_use_autofix_chat("프로젝트 분석해줘"))
        self.assertTrue(should_use_autofix_chat("오류 로그 보고 자동 수정해줘"))
        self.assertTrue(is_chat_apply_approved("승인하고 코드 수정해줘"))
        self.assertTrue(is_chat_apply_approved("apply 승인 후 반영"))
        self.assertFalse(is_chat_apply_approved("분석만 해줘"))
        self.assertTrue(is_chat_coding_request("파일 디렉토리 분석해서 기능 구현해줘"))
        self.assertTrue(is_chat_coding_request("add feature after reading the codebase"))
        self.assertFalse(is_chat_coding_request("PLAN BUILD CHAT 차이를 알려줘"))

    def test_tui_command_placeholder_mentions_deepagents_model(self):
        self.assertNotIn("DeepAgents", command_placeholder_for_mode("Plan", "qwen3.6"))
        self.assertNotIn("qwen3.6", command_placeholder_for_mode("Build", "qwen3.6"))

    def test_tui_command_placeholder_mentions_model_command(self):
        self.assertNotIn("/path", command_placeholder_for_mode("Plan", "qwen3.6"))
        self.assertNotIn("/model", command_placeholder_for_mode("Build", "gamma"))

    def test_tui_input_does_not_manually_insert_printable_characters(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertNotIn("insert_text_at_cursor(event.character)", source)

    def test_tui_input_supports_ctrl_space_gap(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertIn('"ctrl+space"', source)
        self.assertIn('insert_text_at_cursor("    ")', source)

    def test_tui_input_uses_multiline_text_area(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertIn("TextArea", source)
        self.assertIn("RichLog", source)
        self.assertIn("class LogView(RichLog)", source)
        self.assertIn("def replace_text", source)
        self.assertIn("if text == self._last_rendered_text:", source)
        self.assertIn("self.scroll_end(animate=False)", source)
        self.assertIn("overflow-y: auto", source)
        self.assertIn("LogView(id=\"log\", wrap=True", source)
        self.assertNotIn('Binding("enter", "submit_input"', source)
        self.assertIn("def action_submit_input", source)
        self.assertIn("async def _on_key", source)
        self.assertIn("await super()._on_key(event)", source)
        self.assertNotIn("def on_key(self, event) -> None:\n            self._handle_submit_keys(event)", source)
        self.assertIn("def _handle_submit_keys", source)
        self.assertIn('event.key in {"ctrl+enter", "ctrl+j"}', source)
        self.assertIn('event.key == "shift+enter"', source)
        self.assertIn('event.key == "enter"', source)
        self.assertIn('insert_text_at_cursor("\\n")', source)
        self.assertIn("normalize_pasted_input(event.text)", source)
        self.assertIn("self.app._submit_command_input()", source)
        self.assertIn("SendButton", source)
        self.assertIn('SendButton("SEND  Enter", id="send")', source)
        self.assertIn("FileButton", source)
        self.assertIn('FileButton("FILE", id="file")', source)
        self.assertIn("SampleButton", source)
        self.assertIn('SampleButton("SAMPLE", id="sample")', source)
        self.assertIn("CancelButton", source)
        self.assertIn('CancelButton("CANCEL", id="cancel"', source)
        self.assertIn('Vertical(id="input-area")', source)
        self.assertIn("#input-area", source)
        self.assertIn("dock: bottom", source)
        self.assertIn("def on_button_pressed", source)
        self.assertIn('event.button.id == "file"', source)
        self.assertIn("self._open_folder_picker()", source)
        self.assertIn("self.controller.start_folder_selection(base)", source)
        self.assertIn('event.button.id == "sample"', source)
        self.assertIn("self._open_sample_picker()", source)
        self.assertIn("self.controller.start_sample_selection()", source)
        self.assertIn('event.button.id == "cancel"', source)
        self.assertIn("self._cancel_chatbot_request()", source)
        self.assertIn('event.button.id != "send"', source)
        self.assertNotIn("Input.Submitted", source)

    def test_tui_sample_button_selection_creates_selected_sample(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                controller = self.beginner_tui("")
                controller.submit("1")

                message = controller.start_sample_selection()
                self.assertIn("샘플 모델을 선택하세요", message)
                self.assertIn("TensorFlow", format_sample_choices())
                self.assertIn("1-11", sample_selection_placeholder())

                output = controller.select_sample("4")
            finally:
                os.chdir(cwd)

            self.assertIn("pytorch-model", controller.project_path)
            self.assertIn("Step 1", output)
            self.assertTrue((Path(controller.project_path) / "model" / "pytorch-sample.pt").exists())

    def test_tui_chatbot_cancel_button_ignores_late_response(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertIn("self._cancelled_chatbot_requests", source)
        self.assertIn("self._active_chatbot_request_id", source)
        self.assertIn("def _cancel_chatbot_request", source)
        self.assertIn("늦게 도착한 응답은 표시하지 않습니다", source)
        self.assertIn("request_id in self._cancelled_chatbot_requests", source)
        self.assertIn("cancel.disabled = not self._chatbot_busy", source)

    def test_tui_paste_normalization_keeps_multiline_but_cleans_indent(self):
        raw = "\n        첫 줄\n            둘째 줄\n\n\n        셋째 줄   \n"

        output = normalize_pasted_input(raw)

        self.assertEqual(output, "첫 줄\n    둘째 줄\n\n셋째 줄")

    def test_tui_paste_normalization_preserves_korean_and_removes_terminal_markers(self):
        raw = "\x1b[200~\u1112\u1161\u11ab\u1100\u1173\u11af \x1b[31m테스트\x1b[0m\x1b[201~"

        output = normalize_pasted_input(raw)

        self.assertEqual(output, "한글 테스트")

    def test_tui_paste_normalization_repairs_korean_mojibake_for_chat_input(self):
        raw = "íë¡ì í¸ ë¶ìí´ì¤"

        output = normalize_pasted_input(raw)

        self.assertEqual(output, "프로젝트 분석해줘")

    def test_tui_paste_prevents_textual_default_duplicate_insert(self):
        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(source.count('if hasattr(event, "prevent_default"):'), 2)
        self.assertIn("event.prevent_default()", source)

    def test_tui_right_click_copies_current_screen(self):
        class Event:
            button = 3
            button_name = ""

        class LeftEvent:
            button = 1
            button_name = ""

        source = (Path(__file__).resolve().parents[1] / "deep_agent" / "tui.py").read_text(encoding="utf-8")

        self.assertTrue(is_right_click_event(Event()))
        self.assertFalse(is_right_click_event(LeftEvent()))
        self.assertIn("def on_mouse_down", source)
        self.assertIn("action_copy_current_screen", source)
        self.assertIn("copy_to_clipboard", source)
        self.assertIn("normalize_clipboard_text(self.controller.render_log())", source)
        self.assertIn("copy_text_to_clipboard(text)", source)
        self.assertIn("UTF-8", source)

    def test_tui_model_selection_placeholder_shows_number_range(self):
        self.assertIn("1-4", model_selection_placeholder(["qwen3.6", "qwen3.5", "gpt20", "gamma"]))
        self.assertIn("Tab/화살표", model_selection_placeholder(["qwen3.6"]))
        self.assertIn("모델명", model_selection_placeholder([]))

    def test_tui_folder_selection_placeholder_shows_number_range(self):
        self.assertIn("1-2", folder_selection_placeholder([Path("/tmp/a"), Path("/tmp/b")]))
        self.assertIn("Tab/화살표", folder_selection_placeholder([Path("/tmp/a")]))
        self.assertIn("폴더 경로", folder_selection_placeholder([]))
        self.assertIn("Open Folder", folder_selection_placeholder([Path("/tmp/a")]))

    def test_tui_parse_model_commands(self):
        self.assertEqual(parse_model_command("/model qwen3.5"), "qwen3.5")
        self.assertEqual(parse_model_command("/모델 gamma"), "gamma")
        self.assertEqual(parse_model_command("/model"), "")
        self.assertIsNone(parse_model_command("모델 gamma"))

    def test_tui_parse_folder_commands(self):
        self.assertEqual(parse_folder_command("/folder"), "")
        self.assertEqual(parse_folder_command("/폴더 /tmp/models"), "/tmp/models")
        self.assertEqual(parse_folder_command("/dir ./work"), "./work")
        self.assertEqual(parse_folder_command("/open"), "")
        self.assertEqual(parse_folder_command("/open ./work"), "./work")
        self.assertEqual(parse_folder_command("/file /tmp/models"), "/tmp/models")
        self.assertEqual(parse_folder_command("/파일 /tmp/models"), "/tmp/models")
        self.assertEqual(parse_folder_command("/열기 /tmp/models"), "/tmp/models")
        self.assertIsNone(parse_folder_command("folder ./work"))

    def test_tui_formats_model_choices_with_current_marker(self):
        output = format_model_choices(["qwen3.6", "gamma"], "gamma")

        self.assertIn("1. qwen3.6", output)
        self.assertIn("2. gamma (현재)", output)
        self.assertIn("번호를 입력", output)

    def test_tui_formats_folder_choices_with_current_marker(self):
        folders = [Path("/tmp/model-a"), Path("/tmp/model-b")]
        output = format_folder_choices(folders, folders[1])

        self.assertIn("1. /tmp/model-a", output)
        self.assertIn("2. /tmp/model-b (선택)", output)
        self.assertIn("번호를 입력", output)

    def test_tui_controller_lists_and_selects_model_from_input_box(self):
        controller = self.beginner_tui("")

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
        controller = self.beginner_tui("")

        controller.submit("/model")
        output = controller.submit("2")

        self.assertEqual(output, "")
        self.assertEqual(controller.qwen_model, "qwen3.5")
        self.assertFalse(controller.awaiting_model_selection)

    def test_tui_controller_cycles_model_selection_for_input_box(self):
        controller = self.beginner_tui("")

        controller.submit("/model")
        self.assertEqual(controller.highlighted_model, "qwen3.6")
        self.assertEqual(controller.cycle_model_selection(1), "qwen3.5")
        self.assertEqual(controller.cycle_model_selection(1), "gpt20")
        self.assertEqual(controller.cycle_model_selection(-1), "qwen3.5")

    def test_tui_controller_enter_selects_highlighted_model_when_input_empty(self):
        controller = self.beginner_tui("")

        controller.submit("/model")
        controller.cycle_model_selection(1)
        output = controller.submit("")

        self.assertEqual(output, "")
        self.assertEqual(controller.qwen_model, "qwen3.5")
        self.assertFalse(controller.awaiting_model_selection)

    def test_tui_controller_selects_model_by_number_in_model_command(self):
        controller = self.beginner_tui("")

        output = controller.submit("/model 4")

        self.assertEqual(output, "")
        self.assertEqual(controller.qwen_model, "gamma")

    def test_tui_controller_rejects_unknown_model(self):
        controller = self.beginner_tui("")

        output = controller.submit("/model unknown-model")

        self.assertIn("지원하지 않는 모델", output)
        self.assertEqual(controller.qwen_model, "qwen3.6")

    def test_tui_controller_keeps_model_menu_open_on_invalid_number(self):
        controller = self.beginner_tui("")

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

    def test_tui_normalizes_dragged_folder_path_variants(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spaced = root / "AI ML" / "model"
            spaced.mkdir(parents=True)
            encoded = str(spaced).replace(" ", "%20")

            self.assertEqual(normalize_input_path(f"'{spaced}' "), spaced.resolve())
            self.assertEqual(normalize_input_path(f"file://localhost{encoded}/"), spaced.resolve())
            self.assertEqual(normalize_input_path(f"open \"{spaced}\""), spaced.resolve())
            self.assertEqual(
                normalize_input_path(f"PS C:\\Users\\choi>\nfile://localhost{encoded}\n"),
                spaced.resolve(),
            )

    def test_tui_file_button_folder_dialog_returns_selected_folder(self):
        with TemporaryDirectory() as tmpdir:
            selected = Path(tmpdir) / "selected model"
            selected.mkdir()

            class FakeTk:
                def withdraw(self):
                    return None

                def attributes(self, *args):
                    return None

                def destroy(self):
                    return None

            fake_tkinter = types.ModuleType("tkinter")
            fake_filedialog = types.SimpleNamespace(askdirectory=lambda initialdir=None: str(selected))
            fake_tkinter.Tk = FakeTk
            fake_tkinter.filedialog = fake_filedialog
            old_tkinter = sys.modules.get("tkinter")
            old_filedialog = sys.modules.get("tkinter.filedialog")
            sys.modules["tkinter"] = fake_tkinter
            sys.modules["tkinter.filedialog"] = fake_filedialog
            try:
                self.assertEqual(choose_folder_with_dialog(), selected.resolve())
            finally:
                if old_tkinter is None:
                    sys.modules.pop("tkinter", None)
                else:
                    sys.modules["tkinter"] = old_tkinter
                if old_filedialog is None:
                    sys.modules.pop("tkinter.filedialog", None)
                else:
                    sys.modules["tkinter.filedialog"] = old_filedialog

    def test_tui_preserves_windows_paths_with_backslashes_and_spaces(self):
        self.assertEqual(
            normalize_path_text(r"C:\Users\choi\AI ML\model"),
            r"C:\Users\choi\AI ML\model",
        )
        self.assertEqual(
            normalize_path_text(r'"C:\Users\choi\AI ML\model"'),
            r"C:\Users\choi\AI ML\model",
        )
        self.assertEqual(
            normalize_path_text("file:///C:/Users/choi/AI%20ML/model"),
            "C:/Users/choi/AI ML/model",
        )
        self.assertEqual(
            normalize_path_text("//C:/Users/choi/AI%20ML/model"),
            "C:/Users/choi/AI ML/model",
        )
        self.assertEqual(
            normalize_path_text(r"\\C:\Users\choi\AI ML\model"),
            r"C:\Users\choi\AI ML\model",
        )
        self.assertEqual(
            normalize_path_text(r"\\?\C:\Users\choi\AI ML\model"),
            r"C:\Users\choi\AI ML\model",
        )
        self.assertTrue(is_windows_absolute_path(r"C:\Users\choi\AI ML\model"))
        self.assertTrue(is_windows_absolute_path(r"\\mlserver\models\team-a\model"))

    def test_tui_extracts_windows_paths_from_multiline_shell_paste(self):
        pasted = "\n".join(
            [
                "PS C:\\Users\\choi>",
                r"cd 'C:\Users\choi\AI ML\model'",
                "다음 줄은 설명입니다",
            ]
        )

        self.assertEqual(strip_shell_path_prefix(r"cd 'C:\Users\choi\AI ML\model'"), r"'C:\Users\choi\AI ML\model'")
        self.assertIn(r"'C:\Users\choi\AI ML\model'", path_candidates_from_input(pasted))

    def test_tui_resolves_windows_path_from_shell_prefixed_multiline_input(self):
        old_value = os.environ.get("AIU_WINDOWS_DRIVE_C")
        with TemporaryDirectory() as tmpdir:
            drive_root = Path(tmpdir)
            project = drive_root / "Users" / "choi" / "AI ML" / "model"
            project.mkdir(parents=True)
            (project / "requirements.txt").write_text("mlflow\n")
            os.environ["AIU_WINDOWS_DRIVE_C"] = str(drive_root)
            try:
                self.assertEqual(
                    normalize_input_path("\n" + r"cd 'C:\Users\choi\AI ML\model'" + "\n"),
                    project.resolve(),
                )
            finally:
                if old_value is None:
                    os.environ.pop("AIU_WINDOWS_DRIVE_C", None)
                else:
                    os.environ["AIU_WINDOWS_DRIVE_C"] = old_value

    def test_tui_expands_windows_env_path_variants(self):
        old_value = os.environ.get("AIU_TEST_HOME")
        os.environ["AIU_TEST_HOME"] = r"C:\Users\choi"
        try:
            self.assertEqual(
                normalize_path_text(r"%AIU_TEST_HOME%\model"),
                r"C:\Users\choi\model",
            )
            self.assertEqual(
                normalize_path_text(r"$env:AIU_TEST_HOME\model"),
                r"C:\Users\choi\model",
            )
        finally:
            if old_value is None:
                os.environ.pop("AIU_TEST_HOME", None)
            else:
                os.environ["AIU_TEST_HOME"] = old_value

    def test_tui_resolves_windows_absolute_path_with_drive_mapping(self):
        old_value = os.environ.get("AIU_WINDOWS_DRIVE_C")
        with TemporaryDirectory() as tmpdir:
            drive_root = Path(tmpdir)
            project = drive_root / "Users" / "choi" / "AI ML" / "model"
            project.mkdir(parents=True)
            (project / "requirements.txt").write_text("mlflow\n")
            os.environ["AIU_WINDOWS_DRIVE_C"] = str(drive_root)
            try:
                self.assertEqual(normalize_input_path(r"C:\Users\choi\AI ML\model"), project.resolve())
                self.assertEqual(normalize_input_path(r"\\C:\Users\choi\AI ML\model"), project.resolve())
                self.assertEqual(normalize_input_path("//C:/Users/choi/AI%20ML/model"), project.resolve())
                self.assertEqual(normalize_input_path("file:///C:/Users/choi/AI%20ML/model"), project.resolve())
            finally:
                if old_value is None:
                    os.environ.pop("AIU_WINDOWS_DRIVE_C", None)
                else:
                    os.environ["AIU_WINDOWS_DRIVE_C"] = old_value

    def test_tui_resolves_windows_absolute_path_with_drive_root_mapping(self):
        old_drive = os.environ.get("AIU_WINDOWS_DRIVE_C")
        old_root = os.environ.get("AIU_WINDOWS_DRIVE_ROOT")
        with TemporaryDirectory() as tmpdir:
            drives_root = Path(tmpdir) / "drives"
            project = drives_root / "C" / "Users" / "choi" / "AI ML" / "model"
            project.mkdir(parents=True)
            (project / "requirements.txt").write_text("mlflow\n")
            os.environ.pop("AIU_WINDOWS_DRIVE_C", None)
            os.environ["AIU_WINDOWS_DRIVE_ROOT"] = str(drives_root)
            try:
                self.assertIn(project, filesystem_path_candidates(r"C:\Users\choi\AI ML\model"))
                self.assertEqual(normalize_input_path(r"C:\Users\choi\AI ML\model"), project.resolve())
                self.assertEqual(resolve_filesystem_path(r"\\?\C:\Users\choi\AI ML\model"), project.resolve())
            finally:
                if old_drive is None:
                    os.environ.pop("AIU_WINDOWS_DRIVE_C", None)
                else:
                    os.environ["AIU_WINDOWS_DRIVE_C"] = old_drive
                if old_root is None:
                    os.environ.pop("AIU_WINDOWS_DRIVE_ROOT", None)
                else:
                    os.environ["AIU_WINDOWS_DRIVE_ROOT"] = old_root

    def test_tui_resolves_windows_unc_path_with_share_mapping(self):
        old_root = os.environ.get("AIU_UNC_ROOT")
        old_specific = os.environ.get("AIU_UNC_MLSERVER_MODELS")
        with TemporaryDirectory() as tmpdir:
            unc_root = Path(tmpdir) / "unc"
            project = unc_root / "mlserver" / "models" / "team-a" / "model"
            project.mkdir(parents=True)
            (project / "requirements.txt").write_text("mlflow\n")
            os.environ["AIU_UNC_ROOT"] = str(unc_root)
            try:
                self.assertEqual(normalize_input_path(r"\\mlserver\models\team-a\model"), project.resolve())
                self.assertEqual(normalize_input_path("//mlserver/models/team-a/model"), project.resolve())
            finally:
                if old_root is None:
                    os.environ.pop("AIU_UNC_ROOT", None)
                else:
                    os.environ["AIU_UNC_ROOT"] = old_root
                if old_specific is None:
                    os.environ.pop("AIU_UNC_MLSERVER_MODELS", None)
                else:
                    os.environ["AIU_UNC_MLSERVER_MODELS"] = old_specific

    def test_tui_resolves_windows_unc_path_with_specific_share_mapping(self):
        old_value = os.environ.get("AIU_UNC_MLSERVER_MODELS")
        with TemporaryDirectory() as tmpdir:
            share_root = Path(tmpdir) / "models-share"
            project = share_root / "team-a" / "model"
            project.mkdir(parents=True)
            (project / "requirements.txt").write_text("mlflow\n")
            os.environ["AIU_UNC_MLSERVER_MODELS"] = str(share_root)
            try:
                self.assertEqual(normalize_input_path(r"\\mlserver\models\team-a\model"), project.resolve())
            finally:
                if old_value is None:
                    os.environ.pop("AIU_UNC_MLSERVER_MODELS", None)
                else:
                    os.environ["AIU_UNC_MLSERVER_MODELS"] = old_value

    def test_analyze_project_accepts_windows_absolute_path_with_drive_mapping(self):
        old_value = os.environ.get("AIU_WINDOWS_DRIVE_C")
        with TemporaryDirectory() as tmpdir:
            drive_root = Path(tmpdir)
            project = drive_root / "work" / "model"
            project.mkdir(parents=True)
            (project / "requirements.txt").write_text("mlflow\n")
            (project / "train.py").write_text("import mlflow\n")
            os.environ["AIU_WINDOWS_DRIVE_C"] = str(drive_root)
            try:
                analysis = analyze_project(r"C:\work\model")
            finally:
                if old_value is None:
                    os.environ.pop("AIU_WINDOWS_DRIVE_C", None)
                else:
                    os.environ["AIU_WINDOWS_DRIVE_C"] = old_value

        self.assertTrue(analysis.exists)
        self.assertTrue(analysis.is_directory)
        self.assertEqual(Path(analysis.path), project.resolve())

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

            controller = self.beginner_tui("")
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

            controller = self.beginner_tui("")
            output = controller.submit(f"/path {root}")

            self.assertEqual(controller.project_path, str(root.resolve()))
            self.assertIn("Current: Tab 1/10", output)

    def test_tui_controller_selects_folder_from_input_box(self):
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            first = base / "first-model"
            second = base / "second-model"
            first.mkdir()
            second.mkdir()
            (first / "requirements.txt").write_text("tensorflow==2.17.0\n")
            (second / "requirements.txt").write_text("mlflow\n")
            (second / "run_model.py").write_text("print('run')\n")

            folders = discover_selectable_folders(str(base))
            self.assertEqual([folder.name for folder in folders], ["first-model", "second-model"])

            controller = self.beginner_tui("")
            output = controller.submit(f"/open {base}")

            self.assertTrue(controller.awaiting_folder_selection)
            self.assertIn("파일/폴더를 열 프로젝트 위치로 선택하세요", output)
            self.assertIn("second-model", output)

            output = controller.submit("2")

            self.assertFalse(controller.awaiting_folder_selection)
            self.assertEqual(controller.project_path, str(second.resolve()))
            self.assertIn("Current: Tab 1/10", output)

    def test_tui_file_button_discovers_work_folder_samples_first(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                work_sample = Path(tmpdir) / "work" / "sora-work-model"
                work_sample.mkdir(parents=True)
                (work_sample / "requirements.txt").write_text("mlflow\n")
                (work_sample / "run_model.py").write_text("print('run')\n")

                folders = discover_selectable_folders()
            finally:
                os.chdir(cwd)

            self.assertGreaterEqual(len(folders), 1)
            self.assertEqual(folders[0], work_sample.resolve())

    def test_tui_controller_reports_missing_path_for_path_command(self):
        controller = self.beginner_tui("")

        self.assertIn("경로를 함께 입력", controller.submit("/path"))
        self.assertIn("경로를 찾을 수 없습니다", controller.submit("/path /no/such/model"))

    def test_tui_controller_handles_navigation_and_exit(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("tensorflow==2.17.0\n")
            (root / "train.py").write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = self.beginner_tui(str(root))

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
        controller = self.beginner_tui("")
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
        controller = self.beginner_tui("")

        output = controller.submit("/agent chat")

        self.assertIn("CHAT MODE", output)
        self.assertIn("실행 모드: 초급자 모드", output)
        self.assertEqual(controller.agent_mode, "Chatbot")
        self.assertEqual(controller.submit("/agent"), "PLAN | BUILD | [CHAT]")

    def test_tui_controller_step6_approval_applies_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = self.beginner_tui(str(root))
            for _ in range(5):
                controller.submit("다음")

            self.assertEqual(controller.index, 5)
            output = controller.submit("1")

            self.assertEqual(controller.agent_mode, "Plan")
            self.assertIn("Current: Tab 7/10", output)
            self.assertIn("Build에서 수정 적용 후 Plan으로 자동 전환", controller.latest_message)
            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertNotIn("import mlflow", train.read_text())
            self.assertTrue((root / "run_model.py").exists())

    def test_tui_controller_step6_skips_when_no_fix_preview(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ensure_ai_studio_sample_runtime(root)
            (root / "requirements.txt").write_text("mlflow\nfastapi\nuvicorn\n")
            (root / "train.py").write_text("import mlflow\n")
            (root / "model.onnx").write_text("sample")
            controller = self.beginner_tui(str(root))
            for _ in range(5):
                controller.submit("다음")

            output = controller.submit("1")

            self.assertEqual(controller.index, 6)
            self.assertEqual(controller.agent_mode, "Plan")
            self.assertIn("Current: Tab 7/10", output)
            self.assertIn("자동 수정할 항목이 없어 Step 6을 스킵했습니다", controller.latest_message)

    def test_tui_chat_without_deepagents_config_does_not_modify_files(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir)
                requirements = root / "requirements.txt"
                train = root / "train.py"
                requirements.write_text("tensorflow==2.17.0\n")
                train.write_text("print('train')\n")
                (root / "model.keras").write_text("sample")
                controller = self.beginner_tui(str(root))
                controller.select_agent_mode("Chatbot")

                output = controller.submit("코드 자동 수정해줘")
            finally:
                os.chdir(cwd)

            self.assertIn("DeepAgents runtime", output)
            self.assertIn("QWEN_API_KEY", output)
            self.assertEqual(requirements.read_text(), "tensorflow==2.17.0\n")
            self.assertNotIn("import mlflow", train.read_text())

    def test_tui_chatbot_mode_answers_greeting_without_deepagents_config(self):
        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                controller = self.beginner_tui("")
                controller.select_agent_mode("Chatbot")

                output = controller.submit("하이")
            finally:
                os.chdir(cwd)

        self.assertIn("안녕하세요", output)
        self.assertNotIn("QWEN_API_KEY", output)
        self.assertIn("안녕하세요", controller.render_log())

    def test_tui_build_mode_text_points_to_chatbot_without_modifying_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = self.beginner_tui(str(root))
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
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                output = controller.submit("프로젝트 분석해줘")
            finally:
                os.chdir(cwd)

            self.assertIn("DeepAgents 응답", output)
            self.assertIn("최종 등록 상태", output)
            self.assertEqual(fake.calls[-1], ("프로젝트 분석해줘", str(root), "AutoFix"))
            self.assertIn("YOU    프로젝트 분석해줘", controller.render_log())
            self.assertIn("AGENT  response", controller.render_log())
            self.assertIn("DeepAgents 응답", controller.render_log())
            session_files = list((Path(tmpdir) / ".aiu" / "sessions").glob("chat-session-*.jsonl"))
            self.assertEqual(len(session_files), 1)
            session_payload = json.loads(session_files[0].read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(session_payload["user_message"], "프로젝트 분석해줘")
            self.assertEqual(session_payload["analysis_status"], "등록 가능")

    def test_tui_chat_general_question_uses_fast_chat_mode_without_analysis(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("PLAN은 읽기 전용, BUILD는 승인 후 수정, CHAT은 대화입니다.", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir) / "project"
                root.mkdir()
                (root / "requirements.txt").write_text("tensorflow==2.17.0\n")
                fake = FakeRuntime()
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                output = controller.submit("PLAN BUILD CHAT 차이를 알려줘")
            finally:
                os.chdir(cwd)

            self.assertIn("PLAN은 읽기 전용", output)
            self.assertNotIn("최종 등록 상태", output)
            self.assertEqual(fake.calls[-1], ("PLAN BUILD CHAT 차이를 알려줘", str(root), "Chat"))
            session_file = next((Path(tmpdir) / ".aiu" / "sessions").glob("chat-session-*.jsonl"))
            session_payload = json.loads(session_file.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(session_payload["analysis_status"], "not_analyzed")
            self.assertEqual(session_payload["remaining_issues"], [])
            used_prompt_file = Path(tmpdir) / "deep_agent" / "wiki" / "prompts" / "used_prompts.jsonl"
            used_prompt_payload = json.loads(used_prompt_file.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(used_prompt_payload["user_prompt"], "PLAN BUILD CHAT 차이를 알려줘")
            self.assertEqual(used_prompt_payload["agent_mode"], "Chat")
            self.assertIn(
                "PLAN BUILD CHAT 차이를 알려줘",
                (Path(tmpdir) / "deep_agent" / "wiki" / "prompts" / "used_prompts.md").read_text(encoding="utf-8"),
            )

    def test_compacted_runtime_prompt_includes_summary_recent_and_current_request(self):
        prompt = build_compacted_runtime_prompt(
            "현재 질문",
            "이전 요약",
            [{"user_message": "전 질문", "agent_response": "전 답변"}],
        )

        self.assertIn("[압축 요약]", prompt)
        self.assertIn("이전 요약", prompt)
        self.assertIn("[최근 대화]", prompt)
        self.assertIn("전 질문", prompt)
        self.assertIn("[현재 요청]", prompt)
        self.assertIn("현재 질문", prompt)

    def test_tui_manual_compact_saves_summary_and_next_prompt_uses_it(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("응답", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir) / "project"
                root.mkdir()
                fake = FakeRuntime()
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")
                controller._save_chat_session("이전 질문 1", "이전 답변 1", [], None)
                controller._save_chat_session("이전 질문 2", "이전 답변 2", [], None)

                compact_output = controller.submit("/compact")
                output = controller.submit("다음 질문")
            finally:
                os.chdir(cwd)

            summary_file = Path(tmpdir) / ".aiu" / "sessions" / "chat-context-summary.json"
            self.assertIn("챗봇 컨텍스트를 압축했습니다", compact_output)
            self.assertTrue(summary_file.exists())
            self.assertIn("응답", output)
            self.assertIn("[압축 요약]", fake.calls[-1][0])
            self.assertIn("이전 질문", fake.calls[-1][0])
            self.assertIn("다음 질문", fake.calls[-1][0])

    def test_tui_auto_compacts_chat_context_before_runtime_call(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("자동 압축 후 응답", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                Path(".env").write_text(
                    "CHAT_CONTEXT_COMPACT_AFTER=3\n"
                    "CHAT_CONTEXT_RECENT_MESSAGES=1\n",
                    encoding="utf-8",
                )
                root = Path(tmpdir) / "project"
                root.mkdir()
                fake = FakeRuntime()
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")
                controller._save_chat_session("질문 1", "답변 1", [], None)
                controller._save_chat_session("질문 2", "답변 2", [], None)
                controller._save_chat_session("질문 3", "답변 3", [], None)

                controller.submit("현재 분석")
            finally:
                os.chdir(cwd)

            self.assertIn("[압축 요약]", fake.calls[-1][0])
            self.assertIn("질문 1", fake.calls[-1][0])
            self.assertEqual(len(controller.chat_context_entries), 2)
            self.assertGreaterEqual(controller.chat_context_compacted_count, 2)

    def test_tui_chat_coding_request_routes_to_build_runtime(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("디렉토리 분석 후 코딩 변경 완료", True)

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir) / "project"
                root.mkdir()
                (root / "requirements.txt").write_text("mlflow\n")
                (root / "app.py").write_text("print('hello')\n")
                fake = FakeRuntime()
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                output = controller.submit("파일 디렉토리 분석해서 기능 구현해줘")
            finally:
                os.chdir(cwd)

        self.assertIn("디렉토리 분석 후 코딩 변경 완료", output)
        self.assertEqual(fake.calls[-1], ("파일 디렉토리 분석해서 기능 구현해줘", str(root), "Build"))

    def test_tui_chat_coding_request_requires_project_path(self):
        controller = self.beginner_tui("")
        controller.select_agent_mode("Chatbot")

        output = controller.submit("코딩 기능으로 파일 수정해줘")

        self.assertIn("프로젝트 폴더를 선택", output)
        self.assertIn("FILE", output)

    def test_tui_chat_autofix_requires_policy_approval_before_applying(self):
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
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                output = controller.submit("코드 자동 수정해줘")
                approved = controller.submit("1")
            finally:
                os.chdir(cwd)

            self.assertIn("DeepAgents 수정 완료", output)
            self.assertIn("수정 정책: 승인 기반 자동수정", output)
            self.assertTrue(controller.awaiting_chat_code_policy is False)
            self.assertIn("자동 수정 결과", approved)
            self.assertEqual(fake.calls[-1], ("코드 자동 수정해줘", str(root), "AutoFix"))
            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertNotIn("import mlflow", train.read_text())
            self.assertTrue((root / "run_model.py").exists())
            self.assertEqual(controller.index, 6)
            self.assertIn("YOU    코드 자동 수정해줘", controller.render_log())
            self.assertIn("BUILD  changes", controller.render_log())
            session_file = next((Path(tmpdir) / ".aiu" / "sessions").glob("chat-session-*.jsonl"))
            session_payload = json.loads(session_file.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(session_payload["selected_model"], "qwen3.6")
            self.assertTrue(session_payload["applied_changes"])

    def test_tui_chat_approved_apply_falls_back_to_local_fixable_changes(self):
        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                self.calls.append((prompt, project_path, agent_mode))
                return DeepAgentsRunResult("DeepAgents runtime 실행 실패: test", False, error="test")

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
                controller = self.beginner_tui(str(root), deepagents_runtime=fake)
                controller.select_agent_mode("Chatbot")

                output = controller.submit("승인하고 코드 수정해줘")
            finally:
                os.chdir(cwd)

            self.assertEqual(fake.calls[-1], ("승인하고 코드 수정해줘", str(root), "Build"))
            self.assertIn("DeepAgents runtime 실행 실패", output)
            self.assertIn("자동 수정 결과", output)
            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertNotIn("import mlflow", train.read_text())
            self.assertTrue((root / "run_model.py").exists())

    def test_tui_chat_policy_blocks_delete_request_without_modifying_files(self):
        class FakeRuntime:
            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                return DeepAgentsRunResult("삭제 요청을 확인했습니다.", True)

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "project"
            root.mkdir()
            target = root / "train.py"
            target.write_text("print('keep')\n")
            controller = self.beginner_tui(str(root), deepagents_runtime=FakeRuntime())
            controller.select_agent_mode("Chatbot")

            output = controller.submit("train.py 삭제해줘")

            self.assertIn("수정 정책: 승인 기반 자동수정", output)
            self.assertIn("수정 차단", output)
            self.assertIn("파일 삭제 요청", output)
            self.assertEqual(target.read_text(), "print('keep')\n")

    def test_tui_chat_policy_review_required_preview_for_train_code_change(self):
        class FakeRuntime:
            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                return DeepAgentsRunResult("MLflow 코드를 점검했습니다.", True)

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "project"
            root.mkdir()
            ensure_ai_studio_sample_runtime(root)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("mlflow\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = self.beginner_tui(str(root), deepagents_runtime=FakeRuntime())
            controller.select_agent_mode("Chatbot")

            output = controller.submit("MLflow 기록 코드 수정해줘")
            preview = controller.submit("2")

            self.assertIn("검토 필요", output)
            self.assertIn("MLflow 기록 코드 추가", preview)
            self.assertNotIn("import mlflow", train.read_text())

    def test_tui_chat_approved_standard_template_creates_ml_dl_structure(self):
        class FakeRuntime:
            def invoke(self, prompt, *, project_path="", agent_mode="Plan"):
                return DeepAgentsRunResult("표준 템플릿 생성을 진행합니다.", False, error="offline")

        with TemporaryDirectory() as tmpdir:
            cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                root = Path(tmpdir) / "project"
                root.mkdir()
                controller = self.beginner_tui(str(root), deepagents_runtime=FakeRuntime())
                controller.select_agent_mode("Chatbot")

                output = controller.submit("승인하고 pytorch 표준 템플릿 만들어줘")
            finally:
                os.chdir(cwd)

            self.assertIn("표준 템플릿", output)
            self.assertTrue((root / "aiu_custom" / "model_wrapper.py").exists())
            self.assertTrue((root / "config" / "train_config.json").exists())
            self.assertTrue((root / "run_model.py").exists())
            self.assertTrue((root / "train.py").exists())
            self.assertIn("torch", (root / "requirements.txt").read_text(encoding="utf-8"))

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
                controller = self.beginner_tui(str(sample), deepagents_runtime=FakeRuntime())
                controller.select_agent_mode("Chatbot")

                output = controller.submit("문제 발견하면 자동 수정해줘")
                approved = controller.submit("1")
            finally:
                os.chdir(cwd)

            self.assertIn("수정 정책: 승인 기반 자동수정", output)
            self.assertIn("자동 수정 결과", approved)
            self.assertIn("모델 산출물 없음", approved)
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
            controller = self.beginner_tui(str(root))
            controller.toggle_agent()
            for _ in range(5):
                controller.submit("다음")

            output = controller.submit("1")

            self.assertEqual(controller.agent_mode, "Plan")
            self.assertEqual(controller.index, 6)
            self.assertIn("Current: Tab 7/10", output)
            self.assertIn("Build에서 수정 적용 후 Plan으로 자동 전환", controller.latest_message)
            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertNotIn("import mlflow", train.read_text())
            self.assertTrue((root / "run_model.py").exists())

    def test_tui_beginner_step6_approval_one_applies_without_manual_build_switch(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requirements = root / "requirements.txt"
            train = root / "train.py"
            requirements.write_text("tensorflow==2.17.0\n")
            train.write_text("print('train')\n")
            (root / "model.keras").write_text("sample")
            controller = self.beginner_tui(str(root))
            for _ in range(5):
                controller.submit("다음")

            output = controller.submit("1")

            self.assertEqual(controller.agent_mode, "Plan")
            self.assertEqual(controller.index, 6)
            self.assertIn("Current: Tab 7/10", output)
            self.assertIn("Build에서 수정 적용 후 Plan으로 자동 전환", controller.latest_message)
            self.assertIn("mlflow", requirements.read_text().lower())
            self.assertNotIn("import mlflow", train.read_text())
            self.assertTrue((root / "run_model.py").exists())

    def test_tui_beginner_next_on_step10_returns_to_step1(self):
        controller = self.beginner_tui("")
        controller.submit("10")

        output = controller.submit("다음")

        self.assertEqual(controller.index, 0)
        self.assertIn("Current: Tab 1/10", output)
        self.assertIn("Step 1. 프로젝트 선택", output)

    def test_windows_command_wrapper_exists(self):
        wrapper = Path(__file__).resolve().parents[1] / "ml-agent.cmd"

        self.assertTrue(wrapper.exists())
        self.assertIn("py -3", wrapper.read_text())

    def test_quickstart_has_windows_execution_section(self):
        quickstart = Path(__file__).resolve().parents[1] / "QUICKSTART.md"
        content = quickstart.read_text(encoding="utf-8")

        self.assertIn("## 12. Windows 10/11 실행 환경", content)
        self.assertIn(".\\ml-agent.cmd init", content)
        self.assertIn("현재 사용자 읽기/쓰기 권한", content)
        self.assertIn("Linux/macOS에서 확인할 때만", content)
        self.assertIn("## 9. 이미지형 TUI 화면", content)

    def test_tui_preview_image_is_documented(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")

        self.assertTrue((root / "docs" / "tui-preview.svg").exists())
        self.assertIn("docs/tui-preview.svg", readme)
        self.assertIn("읽기/쓰기 권한", readme)


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
        self.assertIn("DEV_COMMAND_TIMEOUT=120", content)
        self.assertIn("AIU_WINDOWS_DRIVE_C=", content)
        self.assertIn("AIU_UNC_ROOT=", content)

    def test_app_config_get_int_uses_default_for_blank_or_invalid_values(self):
        blank = AppConfig(values={"DEV_COMMAND_TIMEOUT": ""}, root_dir=Path.cwd())
        invalid = AppConfig(values={"DEV_COMMAND_TIMEOUT": "abc"}, root_dir=Path.cwd())
        configured = AppConfig(values={"DEV_COMMAND_TIMEOUT": "45"}, root_dir=Path.cwd())

        self.assertEqual(blank.get_int("DEV_COMMAND_TIMEOUT", default=120), 120)
        self.assertEqual(invalid.get_int("DEV_COMMAND_TIMEOUT", default=120), 120)
        self.assertEqual(configured.get_int("DEV_COMMAND_TIMEOUT", default=120), 45)

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

    def test_default_skills_include_ml_platform_onboarding_flow(self):
        expected = {
            "ml-platform-onboarding-orchestrator",
            "model-project-standardization",
            "ai-studio-runtime-template",
            "local-serving-validation",
            "job-template-draft",
            "analysis-reporting",
            "error-log-repair",
            "closed-network-validation",
        }

        self.assertTrue(expected.issubset(set(DEFAULT_SKILLS)))
        for name in expected:
            content = DEFAULT_SKILLS[name]
            self.assertIn("## When To Use", content)
            self.assertIn("##", content)

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

    def test_used_prompt_is_appended_to_wiki(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig.load(root_dir=root)
            paths = append_used_prompt_to_wiki(
                config,
                {
                    "project_path": "/tmp/project",
                    "user_prompt": "프로젝트 분석해줘",
                    "system_prompt": "QWEN_API_KEY=secret-value",
                    "agent_mode": "AutoFix",
                    "launch_mode": "beginner",
                    "selected_model": "qwen3.6",
                    "response_summary": "분석 완료",
                },
            )

            jsonl_path = root / "deep_agent" / "wiki" / "prompts" / "used_prompts.jsonl"
            markdown_path = root / "deep_agent" / "wiki" / "prompts" / "used_prompts.md"
            self.assertEqual(paths, [jsonl_path, markdown_path])
            payload = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(payload["user_prompt"], "프로젝트 분석해줘")
            self.assertEqual(payload["agent_mode"], "AutoFix")
            self.assertIn("프로젝트 분석해줘", markdown_path.read_text(encoding="utf-8"))
            self.assertIn("분석 완료", markdown_path.read_text(encoding="utf-8"))

    def test_cli_start_exports_prompt_templates_to_wiki(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_file = root / "deep_agent" / "prompts" / "prompt_templates.json"
            prompt_file.parent.mkdir(parents=True)
            prompt_file.write_text(
                json.dumps(
                    {
                        "templates": [
                            {
                                "name": "startup_prompt",
                                "description": "시작 시 저장",
                                "prompt": "실행 시작 시 wiki에 남깁니다.",
                                "tags": ["startup"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cwd = Path.cwd()
            try:
                os.chdir(root)
                stdout = StringIO()
                with redirect_stdout(stdout):
                    exit_code = ml_agent.main(["config"])
            finally:
                os.chdir(cwd)

            markdown = root / "deep_agent" / "wiki" / "prompts" / "prompt_templates.md"
            payload = root / "deep_agent" / "wiki" / "prompts" / "prompt_templates.json"
            self.assertEqual(exit_code, 0)
            self.assertTrue(markdown.exists())
            self.assertTrue(payload.exists())
            self.assertIn("startup_prompt", markdown.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(payload.read_text(encoding="utf-8"))["templates"][0]["name"],
                "startup_prompt",
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
