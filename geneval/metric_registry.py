from geneval.exceptions import UnknownMetricError

METRIC_REGISTRY: dict[str, list[dict]] = {
    "faithfulness": [
        {"adapter": "ragas", "metric_class": "faithfulness", "priority": 1},
        {"adapter": "deepeval", "metric_class": "faithfulness", "priority": 2},
    ],
    "answer_relevancy": [
        {"adapter": "ragas", "metric_class": "answer_relevancy", "priority": 1},
        {"adapter": "deepeval", "metric_class": "answer_relevancy", "priority": 2},
    ],
    "context_recall": [
        {"adapter": "ragas", "metric_class": "context_recall", "priority": 1},
        {"adapter": "deepeval", "metric_class": "context_recall", "priority": 2},
    ],
    "context_precision": [
        {"adapter": "ragas", "metric_class": "context_precision_without_reference", "priority": 1},
        {"adapter": "deepeval", "metric_class": "context_precision", "priority": 2},
    ],
    "context_precision_with_reference": [
        {"adapter": "ragas", "metric_class": "context_precision_with_reference", "priority": 1},
    ],
    "context_entity_recall": [
        {"adapter": "ragas", "metric_class": "context_entity_recall", "priority": 1},
    ],
    "noise_sensitivity": [
        {"adapter": "ragas", "metric_class": "noise_sensitivity", "priority": 1},
    ],
    "context_relevance": [
        {"adapter": "deepeval", "metric_class": "context_relevance", "priority": 1},
    ],
}


def resolve_metric(metric_name: str) -> tuple[str, str]:
    entries = METRIC_REGISTRY.get(metric_name)
    if not entries:
        raise UnknownMetricError(metric_name, available=get_available_metrics())
    best = entries[0]
    return best["adapter"], best["metric_class"]


def get_available_metrics() -> list[str]:
    return list(METRIC_REGISTRY.keys())
