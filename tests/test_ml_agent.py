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
    create_heavy_model_sample,
    handle_advanced_input,
    handle_intermediate_request,
    parse_mode,
    parse_mode_command,
    resolve_beginner_project_input,
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
        self.assertIn("문제 수", output)
        self.assertIn("쉬운 설명", output)
        self.assertIn("권장 조치", output)
        self.assertIn("수정안 미리보기", output)
        self.assertIn("파일은 수정하지 않았습니다", output)
        self.assertIn("적용하기", output)
        self.assertIn("다시 보기", output)
        self.assertIn("취소하기", output)
        self.assertIn("승인 전 상태", output)
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
            self.assertIn("문제 수: 0개", output)

    def test_beginner_wizard_lists_step4_issue_details(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 4. 문제 목록 확인", output)
            self.assertIn("MLflow 패키지 누락", output)
            self.assertIn("대상: requirements.txt", output)
            self.assertIn("Agent 수정 가능: 가능", output)
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
            self.assertIn("적용하려면 다음 단계에서 '적용하기'를 선택해야 합니다", output)

    def test_beginner_wizard_shows_step6_approval_choices(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 6. 사용자 승인", output)
            self.assertIn("적용 범위: Step 5에 표시된 미리보기 항목으로 제한됩니다.", output)
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
            self.assertIn("적용하기는 수정안이 있을 때만 선택할 수 있습니다.", output)

    def test_beginner_wizard_shows_step7_apply_scope(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "requirements.txt").write_text("scikit-learn==1.5.2\n")
            (root / "train.py").write_text("print('train')\n")

            output = build_beginner_wizard(str(root))

            self.assertIn("Step 7. 파일 생성 또는 수정", output)
            self.assertIn("'적용하기' 승인 후에만", output)
            self.assertIn("삭제 작업은 수행하지 않습니다.", output)
            self.assertIn("requirements.txt: MLflow 의존성 추가", output)
            self.assertIn("train.py: MLflow 기록 코드 추가", output)

    def test_heavy_model_sample_can_be_selected_from_step1(self):
        with TemporaryDirectory() as tmpdir:
            sample = create_heavy_model_sample(Path(tmpdir) / "heavy-model", artifact_size_bytes=1024)
            analysis = analyze_project(str(sample))
            output = build_beginner_wizard(str(sample))

            self.assertEqual((sample / "model" / "heavy-model.onnx").stat().st_size, 1024)
            self.assertEqual(analysis.registration_status, "등록 가능")
            self.assertIn("등록 상태: 등록 가능", output)
            self.assertIn("heavy-model.onnx", output)

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
