from unittest.mock import Mock

import pytest
import yaml

from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from geneval.profile_manager import ProfileManager
from geneval.schemas import BatchResult, MetricResult, Output, ProfileResult


def _write_yaml(data: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _valid_profile_data():
    return {
        "profiles": {
            "test_profile": {
                "description": "A test profile",
                "metrics": ["faithfulness", "answer_relevancy"],
                "weights": {"faithfulness": 0.7, "answer_relevancy": 0.3},
                "criteria": {"faithfulness": 0.8, "answer_relevancy": 0.9},
                "composite_threshold": 0.85,
            }
        }
    }


def _valid_profile_with_policy():
    data = _valid_profile_data()
    data["policies"] = {
        "ci_gate": {
            "profile": "test_profile",
            "overrides": {
                "criteria": {"faithfulness": 0.95},
                "composite_threshold": 0.9,
            },
        },
        "nightly": {"profile": "test_profile"},
    }
    return data


class TestProfileManagerLoading:
    def test_load_from_yaml(self, tmp_path):
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(_valid_profile_data(), path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.get_profile("test_profile")
        assert profile["description"] == "A test profile"
        assert profile["metrics"] == ["faithfulness", "answer_relevancy"]

    def test_load_builtin_profiles(self):
        pm = ProfileManager()
        profile = pm.get_profile("rag_default")
        assert "faithfulness" in profile["metrics"]
        assert profile["weights"]["faithfulness"] == 0.4

    def test_user_profiles_override_builtins(self, tmp_path):
        data = {
            "profiles": {
                "smoke_test": {
                    "description": "Custom smoke test",
                    "metrics": ["faithfulness"],
                    "weights": {"faithfulness": 1.0},
                    "criteria": {"faithfulness": 0.6},
                    "composite_threshold": 0.6,
                }
            }
        }
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.get_profile("smoke_test")
        assert profile["description"] == "Custom smoke test"
        assert profile["weights"]["faithfulness"] == 1.0

    def test_list_profiles(self, tmp_path):
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(_valid_profile_data(), path)

        pm = ProfileManager(profiles_path=path)
        names = pm.list_profiles()
        assert "test_profile" in names
        assert "rag_default" in names
        assert "strict" in names
        assert "smoke_test" in names

    def test_profile_not_found(self):
        pm = ProfileManager()
        with pytest.raises(ProfileNotFoundError, match="nonexistent"):
            pm.get_profile("nonexistent")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ProfileManager(profiles_path="/nonexistent/path.yaml")

    def test_get_profile_returns_copy(self):
        pm = ProfileManager()
        profile = pm.get_profile("rag_default")
        profile["description"] = "MUTATED"
        original = pm.get_profile("rag_default")
        assert original["description"] != "MUTATED"


class TestProfileManagerValidation:
    def test_weights_must_sum_to_one(self, tmp_path):
        data = _valid_profile_data()
        data["profiles"]["test_profile"]["weights"] = {"faithfulness": 0.5, "answer_relevancy": 0.3}
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        with pytest.raises(ProfileValidationError, match="weights sum to"):
            ProfileManager(profiles_path=path)

    def test_weights_tolerance(self, tmp_path):
        data = _valid_profile_data()
        data["profiles"]["test_profile"]["weights"] = {"faithfulness": 0.7000000001, "answer_relevancy": 0.3}
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        assert pm.get_profile("test_profile") is not None

    def test_missing_weight_for_metric(self, tmp_path):
        data = _valid_profile_data()
        del data["profiles"]["test_profile"]["weights"]["answer_relevancy"]
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        with pytest.raises(ProfileValidationError, match="answer_relevancy"):
            ProfileManager(profiles_path=path)

    def test_missing_criteria_for_metric(self, tmp_path):
        data = _valid_profile_data()
        del data["profiles"]["test_profile"]["criteria"]["answer_relevancy"]
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        with pytest.raises(ProfileValidationError, match="answer_relevancy"):
            ProfileManager(profiles_path=path)

    def test_unknown_metric(self, tmp_path):
        data = _valid_profile_data()
        data["profiles"]["test_profile"]["metrics"].append("hallucination_score")
        data["profiles"]["test_profile"]["weights"]["hallucination_score"] = 0.0
        data["profiles"]["test_profile"]["criteria"]["hallucination_score"] = 0.5
        data["profiles"]["test_profile"]["weights"]["faithfulness"] = 0.7
        data["profiles"]["test_profile"]["weights"]["answer_relevancy"] = 0.3
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        with pytest.raises(UnknownMetricError, match="hallucination_score"):
            ProfileManager(profiles_path=path)

    def test_composite_threshold_defaults(self, tmp_path):
        data = _valid_profile_data()
        del data["profiles"]["test_profile"]["composite_threshold"]
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.get_profile("test_profile")
        # default = weighted average of criteria: 0.7*0.8 + 0.3*0.9 = 0.83
        assert abs(profile["composite_threshold"] - 0.83) < 1e-6


class TestProfileManagerPolicies:
    def test_resolve_policy(self, tmp_path):
        data = _valid_profile_with_policy()
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.resolve_policy("ci_gate")
        assert profile["criteria"]["faithfulness"] == 0.95
        assert profile["criteria"]["answer_relevancy"] == 0.9
        assert profile["composite_threshold"] == 0.9

    def test_resolve_policy_no_overrides(self, tmp_path):
        data = _valid_profile_with_policy()
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.resolve_policy("nightly")
        assert profile["criteria"]["faithfulness"] == 0.8
        assert profile["composite_threshold"] == 0.85

    def test_policy_references_unknown_profile(self, tmp_path):
        data = _valid_profile_data()
        data["policies"] = {"bad_policy": {"profile": "nonexistent"}}
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        with pytest.raises(ProfileNotFoundError, match="nonexistent"):
            pm.resolve_policy("bad_policy")

    def test_policy_not_found(self, tmp_path):
        data = _valid_profile_with_policy()
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        with pytest.raises(ProfileNotFoundError, match="nonexistent_policy"):
            pm.resolve_policy("nonexistent_policy")

    def test_policy_cannot_override_metrics(self, tmp_path):
        data = _valid_profile_with_policy()
        data["policies"]["ci_gate"]["overrides"]["metrics"] = ["faithfulness"]
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.resolve_policy("ci_gate")
        assert profile["metrics"] == ["faithfulness", "answer_relevancy"]

    def test_policy_cannot_override_weights(self, tmp_path):
        data = _valid_profile_with_policy()
        data["policies"]["ci_gate"]["overrides"]["weights"] = {"faithfulness": 1.0}
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.resolve_policy("ci_gate")
        assert profile["weights"]["faithfulness"] == 0.7


class TestProfileManagerScoring:
    def test_compute_profile_result_all_pass(self):
        pm = ProfileManager()
        profile = {
            "metrics": ["faithfulness", "answer_relevancy"],
            "weights": {"faithfulness": 0.7, "answer_relevancy": 0.3},
            "criteria": {"faithfulness": 0.8, "answer_relevancy": 0.9},
            "composite_threshold": 0.85,
        }
        scores = {"faithfulness": (0.9, "ragas", None), "answer_relevancy": (0.95, "ragas", None)}

        result = pm._compute_profile_result("test_profile", None, profile, scores)

        assert result.overall_passed is True
        assert result.composite_passed is True
        assert abs(result.composite_score - (0.7 * 0.9 + 0.3 * 0.95)) < 1e-6
        assert len(result.metric_results) == 2
        assert all(mr.passed for mr in result.metric_results)

    def test_compute_profile_result_individual_fail(self):
        pm = ProfileManager()
        profile = {
            "metrics": ["faithfulness", "answer_relevancy"],
            "weights": {"faithfulness": 0.7, "answer_relevancy": 0.3},
            "criteria": {"faithfulness": 0.8, "answer_relevancy": 0.9},
            "composite_threshold": 0.5,
        }
        scores = {"faithfulness": (0.9, "ragas", None), "answer_relevancy": (0.5, "ragas", None)}

        result = pm._compute_profile_result("test_profile", None, profile, scores)

        assert result.overall_passed is False
        assert result.composite_passed is True
        failed = [mr for mr in result.metric_results if not mr.passed]
        assert len(failed) == 1
        assert failed[0].name == "answer_relevancy"

    def test_compute_profile_result_composite_fail(self):
        pm = ProfileManager()
        profile = {
            "metrics": ["faithfulness", "answer_relevancy"],
            "weights": {"faithfulness": 0.7, "answer_relevancy": 0.3},
            "criteria": {"faithfulness": 0.5, "answer_relevancy": 0.5},
            "composite_threshold": 0.95,
        }
        scores = {"faithfulness": (0.8, "ragas", None), "answer_relevancy": (0.8, "ragas", None)}

        result = pm._compute_profile_result("test_profile", None, profile, scores)

        assert result.overall_passed is False
        assert result.composite_passed is False
        assert all(mr.passed for mr in result.metric_results)

    def test_compute_profile_result_with_policy(self):
        pm = ProfileManager()
        profile = {
            "metrics": ["faithfulness"],
            "weights": {"faithfulness": 1.0},
            "criteria": {"faithfulness": 0.8},
            "composite_threshold": 0.8,
        }
        scores = {"faithfulness": (0.85, "ragas", "High faithfulness")}

        result = pm._compute_profile_result("test_profile", "ci_gate", profile, scores)

        assert result.policy_name == "ci_gate"
        assert result.metric_results[0].details == "High faithfulness"

    def test_compute_profile_result_errored_metric(self):
        pm = ProfileManager()
        profile = {
            "metrics": ["faithfulness"],
            "weights": {"faithfulness": 1.0},
            "criteria": {"faithfulness": 0.8},
            "composite_threshold": 0.8,
        }
        scores = {"faithfulness": (None, "ragas", None)}

        result = pm._compute_profile_result("test_profile", None, profile, scores)

        assert result.overall_passed is False
        assert result.metric_results[0].passed is False
        assert result.metric_results[0].score == 0.0


class TestProfileManagerEvaluate:
    def _make_mock_framework(self):
        framework = Mock()
        framework.llm_info = {"provider": "openai", "model": "gpt-4o-mini"}

        def evaluate_side_effect(**kwargs):
            metric_str = kwargs["metrics"][0]
            parts = metric_str.split(".", 1)
            metric_name = parts[1] if len(parts) > 1 else parts[0]
            mock_output = Mock(spec=Output)
            mock_output.metrics = [Mock(spec=MetricResult, name=metric_name, score=0.9, tool_name="ragas", details="good")]
            mock_output.metrics[0].name = metric_name
            mock_output.metrics[0].score = 0.9
            mock_output.metrics[0].details = "good"
            return {metric_str: (parts[0] if len(parts) > 1 else "ragas", mock_output)}

        framework.evaluate.side_effect = evaluate_side_effect
        return framework

    def test_evaluate_profile(self, tmp_path):
        data = _valid_profile_data()
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        framework = self._make_mock_framework()

        result = pm.evaluate(
            framework=framework,
            profile_name="test_profile",
            question="What is X?",
            response="X is Y.",
            reference="X is Y.",
            retrieval_context="X is Y per the docs.",
        )

        assert isinstance(result, ProfileResult)
        assert result.profile_name == "test_profile"
        assert len(result.metric_results) == 2

    def test_evaluate_policy(self, tmp_path):
        data = _valid_profile_with_policy()
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        framework = self._make_mock_framework()

        result = pm.evaluate(
            framework=framework,
            policy_name="ci_gate",
            question="What is X?",
            response="X is Y.",
            reference="X is Y.",
            retrieval_context="X is Y per the docs.",
        )

        assert isinstance(result, ProfileResult)
        assert result.policy_name == "ci_gate"

    def test_evaluate_batch(self, tmp_path):
        profiles_path = str(tmp_path / "profiles.yaml")
        _write_yaml(_valid_profile_data(), profiles_path)

        test_data = {
            "test_cases": [
                {
                    "id": "tc1",
                    "user_input": "Q1?",
                    "response": "A1.",
                    "reference": "A1.",
                    "retrieved_contexts": "Context 1.",
                },
                {
                    "id": "tc2",
                    "user_input": "Q2?",
                    "response": "A2.",
                    "reference": "A2.",
                    "retrieved_contexts": "Context 2.",
                },
            ]
        }
        data_path = str(tmp_path / "test_data.yaml")
        _write_yaml(test_data, data_path)

        pm = ProfileManager(profiles_path=profiles_path)
        framework = self._make_mock_framework()

        result = pm.evaluate_batch(
            framework=framework,
            profile_name="test_profile",
            data_path=data_path,
        )

        assert isinstance(result, BatchResult)
        assert len(result.case_results) == 2
        assert result.pass_rate == 1.0

    def test_evaluate_falls_back_to_second_adapter(self, tmp_path):
        data = _valid_profile_data()
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        framework = Mock()
        framework.llm_info = {"provider": "openai", "model": "gpt-4o-mini"}

        call_count = {"n": 0}

        def evaluate_side_effect(**kwargs):
            call_count["n"] += 1
            metric_str = kwargs["metrics"][0]
            parts = metric_str.split(".", 1)
            adapter = parts[0]
            metric_name = parts[1]
            if adapter == "ragas":
                raise RuntimeError("RAGAS unavailable")
            mock_output = Mock(spec=Output)
            mock_output.metrics = [Mock(spec=MetricResult)]
            mock_output.metrics[0].name = metric_name
            mock_output.metrics[0].score = 0.85
            mock_output.metrics[0].details = "fallback"
            return {metric_str: (adapter, mock_output)}

        framework.evaluate.side_effect = evaluate_side_effect

        result = pm.evaluate(
            framework=framework,
            profile_name="test_profile",
            question="Q?",
            response="A.",
            reference="A.",
            retrieval_context="Ctx.",
        )

        assert isinstance(result, ProfileResult)
        for mr in result.metric_results:
            assert mr.adapter == "deepeval"
            assert mr.score == 0.85
