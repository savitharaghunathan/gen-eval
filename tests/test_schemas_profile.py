import json

from geneval.schemas import BatchResult, MetricEvaluation, ProfileResult


class TestMetricEvaluation:
    def test_creation(self):
        me = MetricEvaluation(
            name="faithfulness",
            score=0.85,
            threshold=0.8,
            passed=True,
            weight=0.7,
            weighted_score=0.595,
            adapter="ragas",
            details="High faithfulness detected",
        )
        assert me.name == "faithfulness"
        assert me.score == 0.85
        assert me.passed is True
        assert me.weighted_score == 0.595

    def test_details_optional(self):
        me = MetricEvaluation(
            name="faithfulness",
            score=0.85,
            threshold=0.8,
            passed=True,
            weight=0.7,
            weighted_score=0.595,
            adapter="ragas",
        )
        assert me.details is None

    def test_json_serialization(self):
        me = MetricEvaluation(
            name="faithfulness",
            score=0.85,
            threshold=0.8,
            passed=True,
            weight=0.7,
            weighted_score=0.595,
            adapter="ragas",
            details=None,
        )
        data = json.loads(me.model_dump_json())
        assert data["name"] == "faithfulness"
        assert data["score"] == 0.85


class TestProfileResult:
    def _make_metric_eval(self, name="faithfulness", score=0.85, threshold=0.8, weight=0.7):
        return MetricEvaluation(
            name=name,
            score=score,
            threshold=threshold,
            passed=score >= threshold,
            weight=weight,
            weighted_score=weight * score,
            adapter="ragas",
        )

    def test_all_passed(self):
        pr = ProfileResult(
            profile_name="rigorous_rag",
            policy_name=None,
            overall_passed=True,
            composite_score=0.87,
            composite_threshold=0.85,
            composite_passed=True,
            metric_results=[self._make_metric_eval()],
            metadata={"version": "0.1.0"},
        )
        assert pr.overall_passed is True
        assert pr.profile_name == "rigorous_rag"

    def test_policy_name_optional(self):
        pr = ProfileResult(
            profile_name="rigorous_rag",
            overall_passed=True,
            composite_score=0.87,
            composite_threshold=0.85,
            composite_passed=True,
            metric_results=[self._make_metric_eval()],
            metadata={},
        )
        assert pr.policy_name is None

    def test_json_roundtrip(self):
        pr = ProfileResult(
            profile_name="rigorous_rag",
            policy_name="ci_gate",
            overall_passed=True,
            composite_score=0.87,
            composite_threshold=0.85,
            composite_passed=True,
            metric_results=[self._make_metric_eval()],
            metadata={"version": "0.1.0"},
        )
        data = json.loads(pr.model_dump_json())
        assert data["profile_name"] == "rigorous_rag"
        assert data["policy_name"] == "ci_gate"
        assert len(data["metric_results"]) == 1

    def test_failed_result(self):
        pr = ProfileResult(
            profile_name="strict",
            policy_name=None,
            overall_passed=False,
            composite_score=0.6,
            composite_threshold=0.85,
            composite_passed=False,
            metric_results=[self._make_metric_eval(score=0.5, threshold=0.8)],
            metadata={},
        )
        assert pr.overall_passed is False
        assert pr.composite_passed is False


class TestBatchResult:
    def _make_profile_result(self, passed=True):
        me = MetricEvaluation(
            name="faithfulness",
            score=0.9 if passed else 0.5,
            threshold=0.8,
            passed=passed,
            weight=1.0,
            weighted_score=0.9 if passed else 0.5,
            adapter="ragas",
        )
        return ProfileResult(
            profile_name="rag_default",
            policy_name=None,
            overall_passed=passed,
            composite_score=0.9 if passed else 0.5,
            composite_threshold=0.7,
            composite_passed=passed,
            metric_results=[me],
            metadata={},
        )

    def test_all_cases_pass(self):
        br = BatchResult(
            profile_name="rag_default",
            policy_name=None,
            overall_passed=True,
            case_results=[self._make_profile_result(True), self._make_profile_result(True)],
            summary={"faithfulness": {"mean": 0.9}},
            pass_rate=1.0,
            metadata={},
        )
        assert br.overall_passed is True
        assert br.pass_rate == 1.0
        assert len(br.case_results) == 2

    def test_some_cases_fail(self):
        br = BatchResult(
            profile_name="rag_default",
            policy_name=None,
            overall_passed=False,
            case_results=[self._make_profile_result(True), self._make_profile_result(False)],
            summary={"faithfulness": {"mean": 0.7}},
            pass_rate=0.5,
            metadata={},
        )
        assert br.overall_passed is False
        assert br.pass_rate == 0.5

    def test_json_serialization(self):
        br = BatchResult(
            profile_name="rag_default",
            policy_name=None,
            overall_passed=True,
            case_results=[self._make_profile_result(True)],
            summary={"faithfulness": {"mean": 0.9}},
            pass_rate=1.0,
            metadata={},
        )
        data = json.loads(br.model_dump_json())
        assert data["pass_rate"] == 1.0
        assert len(data["case_results"]) == 1
