import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app_config import AppConfig, ensure_runtime_layout
from deep_agent_profile import build_ml_platform_profile, format_profile
from error_log_store import analyze_error_log, list_error_logs, save_error_log
from prompt_store import load_prompt_templates
from ml_agent import (
    MODE_ADVANCED,
    MODE_BEGINNER,
    MODE_INTERMEDIATE,
    build_beginner_wizard,
    handle_advanced_input,
    handle_intermediate_request,
    parse_mode,
    parse_mode_command,
)


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
    def test_beginner_wizard_is_read_only_first(self):
        output = build_beginner_wizard("/workspace/my-model")

        self.assertIn("project-scanner", output)
        self.assertIn("read-only scan", output)
        self.assertIn("수정안 미리보기", output)
        self.assertIn("Human-in-the-loop", output)
        self.assertIn("삭제 작업은 수행하지 않습니다", output)
        self.assertIn("재검증", output)


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
        self.assertIn("Human-in-the-loop", output)

    def test_json_output(self):
        output = handle_advanced_input("ml-agent validate ./project --json")
        payload = json.loads(output)

        self.assertEqual(payload["command"], "validate")
        self.assertEqual(payload["path"], "./project")
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("agent_profile=ai-ml-onboarding-assistant", payload["details"])

    def test_profile_command_outputs_deep_agent_profile(self):
        output = handle_advanced_input("ml-agent profile")

        self.assertIn("Deep Agent Profile", output)
        self.assertIn("project-scanner", output)
        self.assertIn("Human-in-the-loop", output)

    def test_config_command_outputs_env_summary(self):
        output = handle_advanced_input("ml-agent config")

        self.assertIn("Environment Config", output)
        self.assertIn("qwen_model=qwen3.5", output)
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
        self.assertIn("permissions:", output)


class WindowsSetupTest(unittest.TestCase):
    def test_windows_command_wrapper_exists(self):
        wrapper = Path(__file__).resolve().parents[1] / "ml-agent.cmd"

        self.assertTrue(wrapper.exists())
        self.assertIn("py -3", wrapper.read_text())


class AppConfigTest(unittest.TestCase):
    def test_env_example_contains_qwen_and_skill_store(self):
        env_example = Path(__file__).resolve().parents[1] / ".env.example"
        content = env_example.read_text(encoding="utf-8")

        self.assertIn("QWEN_API_KEY=your-internal-qwen-key", content)
        self.assertIn("QWEN_BASE_URL=http://xxx.xxx.xxx.xxx:port/v1", content)
        self.assertIn("SKILL_STORE_DIR=skills", content)

    def test_runtime_layout_creates_skill_store(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = AppConfig.load(root_dir=root)
            directories = ensure_runtime_layout(config)

            self.assertIn(root / "skills", directories)
            self.assertTrue((root / "skills" / "README.md").exists())
            self.assertTrue((root / "registered").exists())


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

        self.assertTrue((root / "skills" / "mlflow-registration-check" / "SKILL.md").exists())
        self.assertTrue((root / "skills" / "job-template-draft" / "SKILL.md").exists())
        self.assertTrue((root / "skills" / "closed-network-validation" / "SKILL.md").exists())
        self.assertTrue((root / "skills" / "error-log-repair" / "SKILL.md").exists())


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
