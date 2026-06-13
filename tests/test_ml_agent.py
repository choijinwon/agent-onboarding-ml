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
    analyze_project,
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
        self.assertIn("등록 상태", output)
        self.assertIn("수정안 미리보기", output)
        self.assertIn("적용하기", output)
        self.assertIn("다시 보기", output)
        self.assertIn("취소하기", output)
        self.assertIn("삭제 작업은 수행하지 않습니다", output)
        self.assertIn("재검증", output)

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
            self.assertIn("MLflow 의존성: 확인됨", output)
            self.assertIn("Job Template 초안 준비: 가능", output)


class ProjectAnalysisTest(unittest.TestCase):
    def test_missing_path_is_not_registerable(self):
        analysis = analyze_project("/path/that/does/not/exist")

        self.assertEqual(analysis.registration_status, "불가")
        self.assertFalse(analysis.exists)
        self.assertIn("프로젝트 경로를 찾을 수 없습니다.", analysis.issues)

    def test_project_with_missing_mlflow_needs_action(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            analysis = analyze_project(str(root))

            self.assertEqual(analysis.registration_status, "보완 필요")
            self.assertFalse(analysis.has_mlflow_dependency)
            self.assertIn("train.py", analysis.entrypoint_candidates)


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

    def test_profile_command_outputs_deep_agent_profile(self):
        output = handle_advanced_input("ml-agent profile")

        self.assertIn("Deep Agent Profile", output)
        self.assertIn("project-scanner", output)
        self.assertIn("적용하기", output)

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
            self.assertTrue((root / "registration_packages").exists())


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
