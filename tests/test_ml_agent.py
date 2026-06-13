import json
import unittest

from deep_agent_profile import build_ml_platform_profile, format_profile
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
        self.assertIn("agent_profile=ml-platform-console-assistant", payload["details"])

    def test_profile_command_outputs_deep_agent_profile(self):
        output = handle_advanced_input("ml-agent profile")

        self.assertIn("Deep Agent Profile", output)
        self.assertIn("project-scanner", output)
        self.assertIn("Human-in-the-loop", output)


class DeepAgentProfileTest(unittest.TestCase):
    def test_profile_contains_deepagents_harness_concepts(self):
        profile = build_ml_platform_profile("beginner")

        self.assertEqual(profile.name, "ml-platform-console-assistant")
        self.assertGreaterEqual(len(profile.subagents), 4)
        self.assertTrue(any(rule.mode == "interrupt" for rule in profile.permissions))
        self.assertIn("task", profile.tools)
        self.assertIn("mlflow-registration-check", profile.skills)

    def test_profile_format_references_upstream(self):
        output = format_profile(build_ml_platform_profile("advanced"))

        self.assertIn("https://github.com/langchain-ai/deepagents", output)
        self.assertIn("permissions:", output)


if __name__ == "__main__":
    unittest.main()
