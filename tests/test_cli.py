import json
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from geneval.cli import cli
from geneval.schemas import BatchResult, MetricEvaluation, ProfileResult


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def profiles_file(tmp_path):
    data = {
        "profiles": {
            "test_profile": {
                "description": "Test profile",
                "metrics": ["faithfulness", "answer_relevancy"],
                "weights": {"faithfulness": 0.7, "answer_relevancy": 0.3},
                "criteria": {"faithfulness": 0.8, "answer_relevancy": 0.9},
                "composite_threshold": 0.85,
            }
        },
        "policies": {
            "ci_gate": {
                "profile": "test_profile",
                "overrides": {"criteria": {"faithfulness": 0.95}},
            }
        },
    }
    path = tmp_path / "profiles.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return str(path)


@pytest.fixture
def data_file(tmp_path):
    data = {
        "test_cases": [
            {
                "id": "tc1",
                "user_input": "What is X?",
                "response": "X is Y.",
                "reference": "X is Y.",
                "retrieved_contexts": "Context about X.",
            }
        ]
    }
    path = tmp_path / "test_data.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return str(path)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "llm_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"providers": {"openai": {"enabled": True, "default": True, "model": "gpt-4o-mini"}}}, f)
    return str(path)


class TestProfilesListCommand:
    def test_list_profiles(self, runner, profiles_file):
        result = runner.invoke(cli, ["profiles", "list", "--profiles", profiles_file])
        assert result.exit_code == 0
        assert "test_profile" in result.output
        assert "rag_default" in result.output

    def test_list_builtin_only(self, runner):
        result = runner.invoke(cli, ["profiles", "list"])
        assert result.exit_code == 0
        assert "rag_default" in result.output
        assert "strict" in result.output
        assert "smoke_test" in result.output


class TestProfilesShowCommand:
    def test_show_profile(self, runner, profiles_file):
        result = runner.invoke(cli, ["profiles", "show", "test_profile", "--profiles", profiles_file])
        assert result.exit_code == 0
        assert "test_profile" in result.output
        assert "faithfulness" in result.output

    def test_show_unknown_profile(self, runner):
        result = runner.invoke(cli, ["profiles", "show", "nonexistent"])
        assert result.exit_code == 2


class TestProfilesValidateCommand:
    def test_validate_valid(self, runner, profiles_file):
        result = runner.invoke(cli, ["profiles", "validate", profiles_file])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid(self, runner, tmp_path):
        bad_data = {
            "profiles": {
                "bad": {
                    "metrics": ["faithfulness"],
                    "weights": {"faithfulness": 0.5},
                    "criteria": {"faithfulness": 0.8},
                }
            }
        }
        path = str(tmp_path / "bad.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(bad_data, f)

        result = runner.invoke(cli, ["profiles", "validate", path])
        assert result.exit_code == 2
        assert "error" in result.output.lower() or "sum to" in result.output.lower()


class TestEvaluateCommand:
    @patch("geneval.cli.GenEvalFramework")
    def test_evaluate_with_profile_json(self, mock_framework_class, runner, profiles_file, data_file, config_file):
        mock_framework = Mock()
        mock_framework_class.return_value = mock_framework

        mock_batch_result = BatchResult(
            profile_name="test_profile",
            policy_name=None,
            overall_passed=True,
            case_results=[
                ProfileResult(
                    profile_name="test_profile",
                    policy_name=None,
                    overall_passed=True,
                    composite_score=0.9,
                    composite_threshold=0.85,
                    composite_passed=True,
                    metric_results=[
                        MetricEvaluation(
                            name="faithfulness", score=0.9, threshold=0.8, passed=True, weight=0.7, weighted_score=0.63, adapter="ragas"
                        ),
                    ],
                    metadata={},
                )
            ],
            summary={"faithfulness": {"mean": 0.9}},
            pass_rate=1.0,
            metadata={},
        )
        mock_framework.evaluate_profile_batch.return_value = mock_batch_result

        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--profile",
                "test_profile",
                "--data",
                data_file,
                "--config",
                config_file,
                "--profiles",
                profiles_file,
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["overall_passed"] is True

    @patch("geneval.cli.GenEvalFramework")
    def test_evaluate_failed_exit_code_1(self, mock_framework_class, runner, profiles_file, data_file, config_file):
        mock_framework = Mock()
        mock_framework_class.return_value = mock_framework

        mock_batch_result = BatchResult(
            profile_name="test_profile",
            policy_name=None,
            overall_passed=False,
            case_results=[],
            summary={},
            pass_rate=0.0,
            metadata={},
        )
        mock_framework.evaluate_profile_batch.return_value = mock_batch_result

        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--profile",
                "test_profile",
                "--data",
                data_file,
                "--config",
                config_file,
                "--profiles",
                profiles_file,
            ],
        )

        assert result.exit_code == 1

    @patch("geneval.cli.GenEvalFramework")
    def test_evaluate_table_format(self, mock_framework_class, runner, profiles_file, data_file, config_file):
        mock_framework = Mock()
        mock_framework_class.return_value = mock_framework

        mock_batch_result = BatchResult(
            profile_name="test_profile",
            policy_name=None,
            overall_passed=True,
            case_results=[
                ProfileResult(
                    profile_name="test_profile",
                    policy_name=None,
                    overall_passed=True,
                    composite_score=0.9,
                    composite_threshold=0.85,
                    composite_passed=True,
                    metric_results=[
                        MetricEvaluation(
                            name="faithfulness", score=0.9, threshold=0.8, passed=True, weight=0.7, weighted_score=0.63, adapter="ragas"
                        ),
                    ],
                    metadata={},
                )
            ],
            summary={"faithfulness": {"mean": 0.9}},
            pass_rate=1.0,
            metadata={},
        )
        mock_framework.evaluate_profile_batch.return_value = mock_batch_result

        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--profile",
                "test_profile",
                "--data",
                data_file,
                "--config",
                config_file,
                "--profiles",
                profiles_file,
                "--format",
                "table",
            ],
        )

        assert result.exit_code == 0
        assert "PASSED" in result.output or "passed" in result.output.lower()

    def test_evaluate_no_profile_or_policy(self, runner, data_file, config_file):
        result = runner.invoke(cli, ["evaluate", "--data", data_file, "--config", config_file])
        assert result.exit_code == 2

    @patch("geneval.cli.GenEvalFramework")
    def test_evaluate_config_error_exit_code_2(self, mock_framework_class, runner, profiles_file, data_file):
        mock_framework_class.side_effect = RuntimeError("Bad config")

        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--profile",
                "test_profile",
                "--data",
                data_file,
                "--config",
                "/nonexistent/config.yaml",
                "--profiles",
                profiles_file,
            ],
        )

        assert result.exit_code == 2
