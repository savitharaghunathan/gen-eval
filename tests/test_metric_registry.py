import pytest

from geneval.exceptions import UnknownMetricError
from geneval.metric_registry import METRIC_REGISTRY, get_available_metrics, resolve_metric, resolve_metric_candidates


class TestMetricRegistry:
    def test_registry_has_faithfulness(self):
        assert "faithfulness" in METRIC_REGISTRY
        entries = METRIC_REGISTRY["faithfulness"]
        assert len(entries) == 2
        assert entries[0]["adapter"] == "ragas"
        assert entries[0]["priority"] == 1

    def test_registry_has_all_expected_metrics(self):
        expected = {
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "context_precision_with_reference",
            "context_entity_recall",
            "noise_sensitivity",
            "context_relevance",
        }
        assert expected == set(METRIC_REGISTRY.keys())

    def test_entries_sorted_by_priority(self):
        for metric_name, entries in METRIC_REGISTRY.items():
            priorities = [e["priority"] for e in entries]
            assert priorities == sorted(priorities), f"{metric_name} entries not sorted by priority"


class TestResolveMetric:
    def test_resolves_to_highest_priority(self):
        adapter, metric_class = resolve_metric("faithfulness")
        assert adapter == "ragas"
        assert metric_class == "faithfulness"

    def test_resolves_deepeval_only_metric(self):
        adapter, metric_class = resolve_metric("context_relevance")
        assert adapter == "deepeval"
        assert metric_class == "context_relevance"

    def test_resolves_ragas_only_metric(self):
        adapter, metric_class = resolve_metric("noise_sensitivity")
        assert adapter == "ragas"
        assert metric_class == "noise_sensitivity"

    def test_context_precision_maps_to_without_reference(self):
        adapter, metric_class = resolve_metric("context_precision")
        assert adapter == "ragas"
        assert metric_class == "context_precision_without_reference"

    def test_unknown_metric_raises(self):
        with pytest.raises(UnknownMetricError, match="hallucination_score"):
            resolve_metric("hallucination_score")

    def test_unknown_metric_includes_available(self):
        with pytest.raises(UnknownMetricError) as exc_info:
            resolve_metric("hallucination_score")
        assert exc_info.value.available is not None
        assert "faithfulness" in exc_info.value.available


class TestResolveMetricCandidates:
    def test_returns_all_candidates_in_priority_order(self):
        candidates = resolve_metric_candidates("faithfulness")
        assert len(candidates) == 2
        assert candidates[0] == ("ragas", "faithfulness")
        assert candidates[1] == ("deepeval", "faithfulness")

    def test_single_candidate(self):
        candidates = resolve_metric_candidates("noise_sensitivity")
        assert len(candidates) == 1
        assert candidates[0] == ("ragas", "noise_sensitivity")

    def test_unknown_metric_raises(self):
        with pytest.raises(UnknownMetricError):
            resolve_metric_candidates("nonexistent")


class TestGetAvailableMetrics:
    def test_returns_all_metric_names(self):
        metrics = get_available_metrics()
        assert "faithfulness" in metrics
        assert "context_relevance" in metrics
        assert len(metrics) == len(METRIC_REGISTRY)
