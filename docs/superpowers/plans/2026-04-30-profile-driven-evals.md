# Profile-Driven Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add profile-driven evaluation with reusable YAML profiles, automatic adapter routing, composite scoring with quality gates, evaluation policies, built-in profiles, batch evaluation, and a CLI entry point.

**Architecture:** A thin ProfileManager layer sits on top of the existing GenEvalFramework. It loads profiles/policies from YAML, resolves abstract metric names to concrete adapters via a static MetricRegistry, orchestrates evaluation through the existing framework, and produces scored ProfileResult/BatchResult objects. A Click-based CLI provides the shell entry point.

**Tech Stack:** Python 3.12+, Pydantic v2, PyYAML, Click, pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `geneval/exceptions.py` | Custom exception classes: ProfileValidationError, UnknownMetricError, ProfileNotFoundError |
| `geneval/metric_registry.py` | METRIC_REGISTRY dict mapping abstract metric names to adapter entries with priority; `resolve_metric()` function for auto-routing |
| `geneval/schemas.py` (modify) | Add MetricEvaluation, ProfileResult, BatchResult Pydantic models |
| `geneval/profiles/__init__.py` | Package marker for built-in profile resources |
| `geneval/profiles/default_profiles.yaml` | Built-in rag_default, strict, smoke_test profiles |
| `geneval/profile_manager.py` | ProfileManager class: YAML loading, validation, policy resolution, evaluation orchestration, scoring |
| `geneval/framework.py` (modify) | Add `evaluate_profile()` and `evaluate_profile_batch()` methods delegating to ProfileManager |
| `geneval/__init__.py` (modify) | Export new public symbols |
| `geneval/cli.py` | Click CLI with `evaluate`, `profiles list`, `profiles show`, `profiles validate` commands |
| `pyproject.toml` (modify) | Add `click` dependency, register `geneval` console script |
| `tests/test_exceptions.py` | Tests for custom exceptions |
| `tests/test_metric_registry.py` | Tests for registry resolution, priority, fallback |
| `tests/test_schemas_profile.py` | Tests for MetricEvaluation, ProfileResult, BatchResult |
| `tests/test_profile_manager.py` | Tests for ProfileManager: loading, validation, scoring, policies, batch |
| `tests/test_framework_profile.py` | Tests for evaluate_profile() and evaluate_profile_batch() on GenEvalFramework |
| `tests/test_cli.py` | CLI tests using Click's CliRunner |

---

### Task 1: Custom Exceptions

**Files:**
- Create: `geneval/exceptions.py`
- Test: `tests/test_exceptions.py`

- [ ] **Step 1: Write tests for custom exceptions**

Create `tests/test_exceptions.py`:

```python
import pytest

from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError


class TestProfileValidationError:
    def test_is_value_error(self):
        err = ProfileValidationError("weights don't sum to 1.0")
        assert isinstance(err, ValueError)

    def test_message(self):
        err = ProfileValidationError("weights for 'rigorous_rag' sum to 0.8, expected 1.0")
        assert "rigorous_rag" in str(err)
        assert "0.8" in str(err)


class TestUnknownMetricError:
    def test_is_value_error(self):
        err = UnknownMetricError("hallucination_score")
        assert isinstance(err, ValueError)

    def test_message(self):
        err = UnknownMetricError("hallucination_score", available=["faithfulness", "answer_relevancy"])
        assert "hallucination_score" in str(err)
        assert "faithfulness" in str(err)


class TestProfileNotFoundError:
    def test_is_key_error(self):
        err = ProfileNotFoundError("nonexistent_profile")
        assert isinstance(err, KeyError)

    def test_message(self):
        err = ProfileNotFoundError("nonexistent_profile")
        assert "nonexistent_profile" in str(err)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exceptions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneval.exceptions'`

- [ ] **Step 3: Implement exceptions**

Create `geneval/exceptions.py`:

```python
class ProfileValidationError(ValueError):
    pass


class UnknownMetricError(ValueError):
    def __init__(self, metric_name: str, available: list[str] | None = None):
        if available:
            msg = f"Unknown metric '{metric_name}'. Available metrics: {', '.join(sorted(available))}"
        else:
            msg = f"Unknown metric '{metric_name}'"
        super().__init__(msg)
        self.metric_name = metric_name
        self.available = available


class ProfileNotFoundError(KeyError):
    def __init__(self, profile_name: str):
        super().__init__(f"Profile not found: '{profile_name}'")
        self.profile_name = profile_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exceptions.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geneval/exceptions.py tests/test_exceptions.py
git commit -m "feat: add custom exceptions for profile evaluation"
```

---

### Task 2: Metric Registry

**Files:**
- Create: `geneval/metric_registry.py`
- Test: `tests/test_metric_registry.py`

- [ ] **Step 1: Write tests for metric registry**

Create `tests/test_metric_registry.py`:

```python
import pytest

from geneval.exceptions import UnknownMetricError
from geneval.metric_registry import METRIC_REGISTRY, get_available_metrics, resolve_metric


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


class TestGetAvailableMetrics:
    def test_returns_all_metric_names(self):
        metrics = get_available_metrics()
        assert "faithfulness" in metrics
        assert "context_relevance" in metrics
        assert len(metrics) == len(METRIC_REGISTRY)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metric_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneval.metric_registry'`

- [ ] **Step 3: Implement metric registry**

Create `geneval/metric_registry.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metric_registry.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geneval/metric_registry.py tests/test_metric_registry.py
git commit -m "feat: add metric registry with auto-routing resolution"
```

---

### Task 3: Output Schemas (MetricEvaluation, ProfileResult, BatchResult)

**Files:**
- Modify: `geneval/schemas.py`
- Test: `tests/test_schemas_profile.py`

- [ ] **Step 1: Write tests for new schemas**

Create `tests/test_schemas_profile.py`:

```python
import json

import pytest

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas_profile.py -v`
Expected: FAIL — `ImportError: cannot import name 'MetricEvaluation' from 'geneval.schemas'`

- [ ] **Step 3: Add new models to schemas.py**

Add the following to the end of `geneval/schemas.py` (after the existing `Output` class):

```python
from typing import Any, Optional

from pydantic import BaseModel, Field


# ... existing Input, MetricResult, Output classes stay unchanged ...


class MetricEvaluation(BaseModel):
    name: str = Field(..., description="Abstract metric name (e.g. 'faithfulness')")
    score: float = Field(..., description="Raw score from adapter (0.0 - 1.0)")
    threshold: float = Field(..., description="Criteria threshold from profile")
    passed: bool = Field(..., description="Whether score >= threshold")
    weight: float = Field(..., description="Weight from profile")
    weighted_score: float = Field(..., description="weight * score")
    adapter: str = Field(..., description="Which adapter produced this score (e.g. 'ragas')")
    details: Optional[str] = Field(default=None, description="Explanation from the adapter")


class ProfileResult(BaseModel):
    profile_name: str = Field(..., description="Name of the evaluation profile used")
    policy_name: Optional[str] = Field(default=None, description="Name of the policy, if used")
    overall_passed: bool = Field(..., description="True if composite AND all individual metrics passed")
    composite_score: float = Field(..., description="Weighted composite score")
    composite_threshold: float = Field(..., description="Threshold for composite score")
    composite_passed: bool = Field(..., description="Whether composite_score >= composite_threshold")
    metric_results: list[MetricEvaluation] = Field(..., description="Per-metric evaluation details")
    metadata: dict[str, Any] = Field(..., description="Timestamps, LLM info, framework version")


class BatchResult(BaseModel):
    profile_name: str = Field(..., description="Name of the evaluation profile used")
    policy_name: Optional[str] = Field(default=None, description="Name of the policy, if used")
    overall_passed: bool = Field(..., description="True only if ALL cases passed")
    case_results: list[ProfileResult] = Field(..., description="Per-case evaluation results")
    summary: dict[str, Any] = Field(..., description="Per-metric averages across cases")
    pass_rate: float = Field(..., description="Fraction of cases that passed (0.0 - 1.0)")
    metadata: dict[str, Any] = Field(..., description="Timestamps, LLM info, framework version")
```

Note: the existing `from typing import Any` import at the top of the file needs `Optional` added to it. Change line 1 of `geneval/schemas.py` from:
```python
from typing import Any
```
to:
```python
from typing import Any, Optional
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas_profile.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Run existing schema tests to verify no regression**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: All 18 existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add geneval/schemas.py tests/test_schemas_profile.py
git commit -m "feat: add MetricEvaluation, ProfileResult, BatchResult schemas"
```

---

### Task 4: Built-in Profiles YAML

**Files:**
- Create: `geneval/profiles/__init__.py`
- Create: `geneval/profiles/default_profiles.yaml`

- [ ] **Step 1: Create the profiles package**

Create `geneval/profiles/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Create the default profiles YAML**

Create `geneval/profiles/default_profiles.yaml`:

```yaml
profiles:
  rag_default:
    description: "Balanced RAG evaluation for general use"
    metrics:
      - faithfulness
      - answer_relevancy
      - context_recall
    weights:
      faithfulness: 0.4
      answer_relevancy: 0.3
      context_recall: 0.3
    criteria:
      faithfulness: 0.7
      answer_relevancy: 0.7
      context_recall: 0.7
    composite_threshold: 0.7

  strict:
    description: "High-bar evaluation for production deployments"
    metrics:
      - faithfulness
      - answer_relevancy
      - context_recall
      - context_precision
    weights:
      faithfulness: 0.35
      answer_relevancy: 0.25
      context_recall: 0.2
      context_precision: 0.2
    criteria:
      faithfulness: 0.9
      answer_relevancy: 0.85
      context_recall: 0.8
      context_precision: 0.8
    composite_threshold: 0.85

  smoke_test:
    description: "Fast sanity check with relaxed thresholds"
    metrics:
      - faithfulness
      - answer_relevancy
    weights:
      faithfulness: 0.5
      answer_relevancy: 0.5
    criteria:
      faithfulness: 0.5
      answer_relevancy: 0.5
    composite_threshold: 0.5
```

- [ ] **Step 3: Commit**

```bash
git add geneval/profiles/__init__.py geneval/profiles/default_profiles.yaml
git commit -m "feat: add built-in evaluation profiles (rag_default, strict, smoke_test)"
```

---

### Task 5: ProfileManager — Loading & Validation

**Files:**
- Create: `geneval/profile_manager.py`
- Test: `tests/test_profile_manager.py`

This task covers loading profiles from YAML, validating them, loading built-in profiles, and resolving policies. Scoring and evaluation orchestration come in Task 6.

- [ ] **Step 1: Write tests for profile loading and validation**

Create `tests/test_profile_manager.py`:

```python
import os
import tempfile

import pytest
import yaml

from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from geneval.profile_manager import ProfileManager


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


class TestProfileManagerValidation:
    def test_weights_must_sum_to_one(self, tmp_path):
        data = _valid_profile_data()
        data["profiles"]["test_profile"]["weights"] = {"faithfulness": 0.5, "answer_relevancy": 0.3}
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        with pytest.raises(ProfileValidationError, match="sum to 1.0"):
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
        # fix weights to sum to 1.0
        data["profiles"]["test_profile"]["weights"]["faithfulness"] = 0.7
        data["profiles"]["test_profile"]["weights"]["answer_relevancy"] = 0.3
        data["profiles"]["test_profile"]["weights"]["hallucination_score"] = 0.0
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
        # metrics should be unchanged from base profile
        assert profile["metrics"] == ["faithfulness", "answer_relevancy"]

    def test_policy_cannot_override_weights(self, tmp_path):
        data = _valid_profile_with_policy()
        data["policies"]["ci_gate"]["overrides"]["weights"] = {"faithfulness": 1.0}
        path = str(tmp_path / "profiles.yaml")
        _write_yaml(data, path)

        pm = ProfileManager(profiles_path=path)
        profile = pm.resolve_policy("ci_gate")
        # weights should be unchanged from base profile
        assert profile["weights"]["faithfulness"] == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_profile_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneval.profile_manager'`

- [ ] **Step 3: Implement ProfileManager (loading, validation, policies)**

Create `geneval/profile_manager.py`:

```python
import copy
import logging
from pathlib import Path

import yaml

from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from geneval.metric_registry import get_available_metrics, resolve_metric

_WEIGHT_TOLERANCE = 1e-6

logger = logging.getLogger(__name__)


class ProfileManager:
    def __init__(self, profiles_path: str | None = None):
        self._profiles: dict[str, dict] = {}
        self._policies: dict[str, dict] = {}

        self._load_builtins()

        if profiles_path is not None:
            path = Path(profiles_path)
            if not path.exists():
                raise FileNotFoundError(f"Profiles file not found: {profiles_path}")
            self._load_user_profiles(path)

        self._validate_all_profiles()

    def _load_builtins(self):
        builtin_path = Path(__file__).parent / "profiles" / "default_profiles.yaml"
        if builtin_path.exists():
            with open(builtin_path) as f:
                data = yaml.safe_load(f)
            if data and "profiles" in data:
                self._profiles.update(data["profiles"])

    def _load_user_profiles(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            return
        if "profiles" in data:
            self._profiles.update(data["profiles"])
        if "policies" in data:
            self._policies.update(data["policies"])

    def _validate_all_profiles(self):
        available_metrics = get_available_metrics()
        for name, profile in self._profiles.items():
            self._validate_profile(name, profile, available_metrics)

    def _validate_profile(self, name: str, profile: dict, available_metrics: list[str]):
        metrics = profile.get("metrics", [])
        weights = profile.get("weights", {})
        criteria = profile.get("criteria", {})

        for m in metrics:
            if m not in available_metrics:
                raise UnknownMetricError(m, available=available_metrics)
            if m not in weights:
                raise ProfileValidationError(f"Profile '{name}': metric '{m}' missing from weights")
            if m not in criteria:
                raise ProfileValidationError(f"Profile '{name}': metric '{m}' missing from criteria")

        weight_sum = sum(weights.get(m, 0) for m in metrics)
        if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
            raise ProfileValidationError(f"Profile '{name}': weights sum to {weight_sum}, expected 1.0")

        if "composite_threshold" not in profile:
            default_threshold = sum(weights.get(m, 0) * criteria.get(m, 0) for m in metrics)
            profile["composite_threshold"] = default_threshold

    def get_profile(self, name: str) -> dict:
        if name not in self._profiles:
            raise ProfileNotFoundError(name)
        return self._profiles[name]

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def list_policies(self) -> list[str]:
        return list(self._policies.keys())

    def resolve_policy(self, policy_name: str) -> dict:
        if policy_name not in self._policies:
            raise ProfileNotFoundError(policy_name)

        policy = self._policies[policy_name]
        base_profile_name = policy["profile"]

        if base_profile_name not in self._profiles:
            raise ProfileNotFoundError(base_profile_name)

        profile = copy.deepcopy(self._profiles[base_profile_name])

        overrides = policy.get("overrides", {})
        if "criteria" in overrides:
            profile["criteria"].update(overrides["criteria"])
        if "composite_threshold" in overrides:
            profile["composite_threshold"] = overrides["composite_threshold"]
        # metrics and weights are NOT overridable

        return profile
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_profile_manager.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add geneval/profile_manager.py tests/test_profile_manager.py
git commit -m "feat: add ProfileManager with YAML loading, validation, and policy resolution"
```

---

### Task 6: ProfileManager — Scoring & Evaluation Orchestration

**Files:**
- Modify: `geneval/profile_manager.py`
- Modify: `tests/test_profile_manager.py`

- [ ] **Step 1: Write tests for scoring and evaluation**

Append the following test classes to `tests/test_profile_manager.py`:

```python
from unittest.mock import Mock, patch

from geneval.schemas import MetricEvaluation, MetricResult, Output, ProfileResult, BatchResult


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
        assert result.composite_passed is True  # composite passes due to low threshold
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
            metric_str = kwargs["metrics"][0]  # e.g., "ragas.faithfulness"
            parts = metric_str.split(".", 1)
            metric_name = parts[1] if len(parts) > 1 else parts[0]
            mock_output = Mock(spec=Output)
            mock_output.metrics = [
                Mock(spec=MetricResult, name=metric_name, score=0.9, tool_name="ragas", details="good")
            ]
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
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_profile_manager.py::TestProfileManagerScoring -v`
Expected: FAIL — `AttributeError: 'ProfileManager' object has no attribute '_compute_profile_result'`

- [ ] **Step 3: Add scoring and evaluation methods to ProfileManager**

Append the following methods to the `ProfileManager` class in `geneval/profile_manager.py`:

```python
    # Add these imports at the top of the file:
    # import datetime
    # from geneval.schemas import BatchResult, MetricEvaluation, ProfileResult

    def _compute_profile_result(
        self,
        profile_name: str,
        policy_name: str | None,
        profile: dict,
        scores: dict[str, tuple],
    ) -> ProfileResult:
        metrics = profile["metrics"]
        weights = profile["weights"]
        criteria = profile["criteria"]
        composite_threshold = profile["composite_threshold"]

        metric_results = []
        for m in metrics:
            raw_score, adapter, details = scores[m]
            score = raw_score if raw_score is not None else 0.0
            threshold = criteria[m]
            weight = weights[m]
            passed = score >= threshold if raw_score is not None else False

            metric_results.append(
                MetricEvaluation(
                    name=m,
                    score=score,
                    threshold=threshold,
                    passed=passed,
                    weight=weight,
                    weighted_score=weight * score,
                    adapter=adapter,
                    details=details,
                )
            )

        composite_score = sum(mr.weighted_score for mr in metric_results)
        composite_passed = composite_score >= composite_threshold
        overall_passed = composite_passed and all(mr.passed for mr in metric_results)

        return ProfileResult(
            profile_name=profile_name,
            policy_name=policy_name,
            overall_passed=overall_passed,
            composite_score=composite_score,
            composite_threshold=composite_threshold,
            composite_passed=composite_passed,
            metric_results=metric_results,
            metadata={
                "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            },
        )

    def _run_metrics(self, framework, profile: dict, question: str, response: str, reference: str, retrieval_context: str) -> dict[str, tuple]:
        scores = {}
        for metric_name in profile["metrics"]:
            adapter_name, adapter_metric_class = resolve_metric(metric_name)
            try:
                raw_results = framework.evaluate(
                    question=question,
                    response=response,
                    reference=reference,
                    retrieval_context=retrieval_context,
                    metrics=[f"{adapter_name}.{adapter_metric_class}"],
                )
                key = f"{adapter_name}.{adapter_metric_class}"
                if key in raw_results:
                    _, output = raw_results[key]
                    if output.metrics:
                        result_metric = output.metrics[0]
                        scores[metric_name] = (result_metric.score, adapter_name, result_metric.details)
                    else:
                        scores[metric_name] = (None, adapter_name, None)
                else:
                    scores[metric_name] = (None, adapter_name, None)
            except Exception as e:
                logger.warning(f"Adapter '{adapter_name}' failed for metric '{metric_name}': {e}")
                scores[metric_name] = (None, adapter_name, None)
        return scores

    def evaluate(
        self,
        framework,
        profile_name: str | None = None,
        policy_name: str | None = None,
        question: str = "",
        response: str = "",
        reference: str = "",
        retrieval_context: str = "",
    ) -> ProfileResult:
        if policy_name:
            profile = self.resolve_policy(policy_name)
            effective_profile_name = self._policies[policy_name]["profile"]
        elif profile_name:
            profile = copy.deepcopy(self.get_profile(profile_name))
            effective_profile_name = profile_name
        else:
            raise ValueError("Either profile_name or policy_name must be provided")

        scores = self._run_metrics(framework, profile, question, response, reference, retrieval_context)
        return self._compute_profile_result(effective_profile_name, policy_name, profile, scores)

    def evaluate_batch(
        self,
        framework,
        data_path: str,
        profile_name: str | None = None,
        policy_name: str | None = None,
    ) -> BatchResult:
        with open(data_path) as f:
            data = yaml.safe_load(f)

        test_cases = data.get("test_cases", [])
        if not test_cases:
            raise ValueError(f"No test cases found in {data_path}")

        case_results = []
        for case in test_cases:
            result = self.evaluate(
                framework=framework,
                profile_name=profile_name,
                policy_name=policy_name,
                question=case.get("user_input", ""),
                response=case.get("response", ""),
                reference=case.get("reference", ""),
                retrieval_context=case.get("retrieved_contexts", ""),
            )
            case_results.append(result)

        all_passed = all(cr.overall_passed for cr in case_results)
        pass_rate = sum(1 for cr in case_results if cr.overall_passed) / len(case_results)

        summary = {}
        if case_results:
            all_metric_names = {mr.name for cr in case_results for mr in cr.metric_results}
            for mn in all_metric_names:
                metric_scores = [mr.score for cr in case_results for mr in cr.metric_results if mr.name == mn]
                summary[mn] = {"mean": sum(metric_scores) / len(metric_scores) if metric_scores else 0.0}

        if policy_name:
            effective_profile_name = self._policies[policy_name]["profile"]
        else:
            effective_profile_name = profile_name or ""

        return BatchResult(
            profile_name=effective_profile_name,
            policy_name=policy_name,
            overall_passed=all_passed,
            case_results=case_results,
            summary=summary,
            pass_rate=pass_rate,
            metadata={
                "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                "num_cases": len(case_results),
            },
        )
```

Also add these imports at the top of `geneval/profile_manager.py`:

```python
import datetime

from geneval.schemas import BatchResult, MetricEvaluation, ProfileResult
```

- [ ] **Step 4: Run all ProfileManager tests**

Run: `uv run pytest tests/test_profile_manager.py -v`
Expected: All tests PASS (loading + validation + scoring + evaluate)

- [ ] **Step 5: Commit**

```bash
git add geneval/profile_manager.py tests/test_profile_manager.py
git commit -m "feat: add scoring, evaluation, and batch processing to ProfileManager"
```

---

### Task 7: Framework Integration (evaluate_profile, evaluate_profile_batch)

**Files:**
- Modify: `geneval/framework.py`
- Test: `tests/test_framework_profile.py`

- [ ] **Step 1: Write tests for framework profile methods**

Create `tests/test_framework_profile.py`:

```python
from unittest.mock import Mock, patch

import pytest

from geneval.adapters.deepeval_adapter import DeepEvalAdapter
from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.framework import GenEvalFramework
from geneval.llm_manager import LLMManager
from geneval.schemas import BatchResult, ProfileResult


class TestEvaluateProfile:
    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_delegates_to_profile_manager(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
        tmp_path,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        mock_ragas_adapter.supported_metrics = ["faithfulness", "answer_relevancy"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]

        framework = GenEvalFramework("test_config.yaml")

        with patch("geneval.framework.ProfileManager") as mock_pm_class:
            mock_pm = Mock()
            mock_pm.evaluate.return_value = Mock(spec=ProfileResult)
            mock_pm_class.return_value = mock_pm

            result = framework.evaluate_profile(
                profile="rag_default",
                profiles_path=str(tmp_path / "profiles.yaml"),
                question="What is X?",
                response="X is Y.",
                reference="X is Y.",
                retrieval_context="Context.",
            )

            assert result is not None
            mock_pm.evaluate.assert_called_once()
            call_kwargs = mock_pm.evaluate.call_args[1]
            assert call_kwargs["profile_name"] == "rag_default"
            assert call_kwargs["question"] == "What is X?"

    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_with_policy(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
        tmp_path,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]

        framework = GenEvalFramework("test_config.yaml")

        with patch("geneval.framework.ProfileManager") as mock_pm_class:
            mock_pm = Mock()
            mock_pm.evaluate.return_value = Mock(spec=ProfileResult)
            mock_pm_class.return_value = mock_pm

            result = framework.evaluate_profile(
                policy="ci_gate",
                profiles_path=str(tmp_path / "profiles.yaml"),
                question="What is X?",
                response="X is Y.",
                reference="X is Y.",
                retrieval_context="Context.",
            )

            call_kwargs = mock_pm.evaluate.call_args[1]
            assert call_kwargs["policy_name"] == "ci_gate"
            assert call_kwargs["profile_name"] is None

    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_batch(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
        tmp_path,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]

        framework = GenEvalFramework("test_config.yaml")

        with patch("geneval.framework.ProfileManager") as mock_pm_class:
            mock_pm = Mock()
            mock_pm.evaluate_batch.return_value = Mock(spec=BatchResult)
            mock_pm_class.return_value = mock_pm

            result = framework.evaluate_profile_batch(
                profile="rag_default",
                data_path=str(tmp_path / "data.yaml"),
                profiles_path=str(tmp_path / "profiles.yaml"),
            )

            assert result is not None
            mock_pm.evaluate_batch.assert_called_once()

    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_no_profile_or_policy_raises(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter_class.return_value = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter_class.return_value = Mock(spec=DeepEvalAdapter)

        framework = GenEvalFramework("test_config.yaml")

        with pytest.raises(ValueError, match="Either profile or policy"):
            framework.evaluate_profile(
                question="What is X?",
                response="X is Y.",
                reference="X is Y.",
                retrieval_context="Context.",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_framework_profile.py -v`
Expected: FAIL — `AttributeError: 'GenEvalFramework' object has no attribute 'evaluate_profile'`

- [ ] **Step 3: Add evaluate_profile and evaluate_profile_batch to GenEvalFramework**

Add the following methods to `GenEvalFramework` in `geneval/framework.py`, and add the import `from geneval.profile_manager import ProfileManager` at the top:

```python
from geneval.profile_manager import ProfileManager

# ... existing code ...

    def evaluate_profile(
        self,
        profile: str | None = None,
        policy: str | None = None,
        profiles_path: str | None = None,
        question: str = "",
        response: str = "",
        reference: str = "",
        retrieval_context: str = "",
    ) -> "ProfileResult":
        if not profile and not policy:
            raise ValueError("Either profile or policy must be provided")

        pm = ProfileManager(profiles_path=profiles_path)
        return pm.evaluate(
            framework=self,
            profile_name=profile,
            policy_name=policy,
            question=question,
            response=response,
            reference=reference,
            retrieval_context=retrieval_context,
        )

    def evaluate_profile_batch(
        self,
        data_path: str,
        profile: str | None = None,
        policy: str | None = None,
        profiles_path: str | None = None,
    ) -> "BatchResult":
        if not profile and not policy:
            raise ValueError("Either profile or policy must be provided")

        pm = ProfileManager(profiles_path=profiles_path)
        return pm.evaluate_batch(
            framework=self,
            data_path=data_path,
            profile_name=profile,
            policy_name=policy,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_framework_profile.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run existing framework tests for regression**

Run: `uv run pytest tests/test_framework.py -v`
Expected: All 15 existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add geneval/framework.py tests/test_framework_profile.py
git commit -m "feat: add evaluate_profile and evaluate_profile_batch to GenEvalFramework"
```

---

### Task 8: Update __init__.py Exports

**Files:**
- Modify: `geneval/__init__.py`

- [ ] **Step 1: Update exports**

Modify `geneval/__init__.py` to export the new public symbols:

```python
"""
GenEval: A Unified Evaluation Framework for Generative AI Applications

This package provides a unified interface for evaluating generative AI models
across different frameworks like RAGAS and DeepEval.
"""

__version__ = "0.1.0"
__author__ = "Savitha Raghunathan"

from .exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from .framework import GenEvalFramework
from .llm_manager import LLMManager
from .profile_manager import ProfileManager
from .schemas import BatchResult, Input, MetricEvaluation, MetricResult, Output, ProfileResult

__all__ = [
    "GenEvalFramework",
    "LLMManager",
    "ProfileManager",
    "Input",
    "Output",
    "MetricResult",
    "MetricEvaluation",
    "ProfileResult",
    "BatchResult",
    "ProfileValidationError",
    "UnknownMetricError",
    "ProfileNotFoundError",
]
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from geneval import ProfileManager, ProfileResult, BatchResult, MetricEvaluation; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run all tests for regression**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add geneval/__init__.py
git commit -m "feat: export profile evaluation symbols from geneval package"
```

---

### Task 9: CLI — Evaluate Command

**Files:**
- Create: `geneval/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add click dependency to pyproject.toml**

In `pyproject.toml`, add `"click>=8.0.0"` to the `dependencies` list:

```toml
dependencies = [
    "click>=8.0.0",
    # ... existing deps ...
]
```

- [ ] **Step 2: Add console script entry point to pyproject.toml**

Add this section to `pyproject.toml`:

```toml
[project.scripts]
geneval = "geneval.cli:cli"
```

- [ ] **Step 3: Install the new dependency**

Run: `uv sync --dev --all-extras`

- [ ] **Step 4: Write CLI tests**

Create `tests/test_cli.py`:

```python
import json
import tempfile
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
                "--profile", "test_profile",
                "--data", data_file,
                "--config", config_file,
                "--profiles", profiles_file,
                "--format", "json",
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
                "--profile", "test_profile",
                "--data", data_file,
                "--config", config_file,
                "--profiles", profiles_file,
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
                "--profile", "test_profile",
                "--data", data_file,
                "--config", config_file,
                "--profiles", profiles_file,
                "--format", "table",
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
                "--profile", "test_profile",
                "--data", data_file,
                "--config", "/nonexistent/config.yaml",
                "--profiles", profiles_file,
            ],
        )

        assert result.exit_code == 2
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneval.cli'`

- [ ] **Step 6: Implement the CLI**

Create `geneval/cli.py`:

```python
import json
import sys

import click
import yaml

from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from geneval.framework import GenEvalFramework
from geneval.profile_manager import ProfileManager


@click.group()
def cli():
    pass


@cli.command()
@click.option("--profile", default=None, help="Profile name to evaluate with")
@click.option("--policy", default=None, help="Policy name to evaluate with")
@click.option("--data", "data_path", required=True, help="Path to test data YAML")
@click.option("--config", "config_path", default="./config/llm_config.yaml", help="Path to LLM config YAML")
@click.option("--profiles", "profiles_path", default=None, help="Path to profiles YAML")
@click.option("--output", "output_path", default=None, help="Write JSON results to file")
@click.option("--format", "output_format", type=click.Choice(["json", "table"]), default="json", help="Output format")
def evaluate(profile, policy, data_path, config_path, profiles_path, output_path, output_format):
    if not profile and not policy:
        raise click.UsageError("Either --profile or --policy is required")
    if profile and policy:
        raise click.UsageError("--profile and --policy are mutually exclusive")

    try:
        framework = GenEvalFramework(config_path=config_path)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    try:
        result = framework.evaluate_profile_batch(
            data_path=data_path,
            profile=profile,
            policy=policy,
            profiles_path=profiles_path,
        )
    except (ProfileValidationError, UnknownMetricError, ProfileNotFoundError, FileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if output_format == "json":
        json_str = result.model_dump_json(indent=2)
        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)
            click.echo(f"Results written to {output_path}")
        else:
            click.echo(json_str)
    elif output_format == "table":
        _print_table(result)

    sys.exit(0 if result.overall_passed else 1)


def _print_table(result):
    click.echo(f"\nProfile: {result.profile_name}")
    if result.policy_name:
        click.echo(f"Policy: {result.policy_name}")
    click.echo(f"Pass Rate: {result.pass_rate:.0%} ({sum(1 for c in result.case_results if c.overall_passed)}/{len(result.case_results)})")
    click.echo()

    click.echo(f"{'Metric':<30} {'Mean Score':<12} {'Threshold':<12}")
    click.echo("-" * 54)
    for metric_name, stats in result.summary.items():
        mean = stats.get("mean", 0.0)
        threshold = ""
        if result.case_results:
            for mr in result.case_results[0].metric_results:
                if mr.name == metric_name:
                    threshold = f"{mr.threshold:.2f}"
                    break
        click.echo(f"{metric_name:<30} {mean:<12.4f} {threshold:<12}")

    click.echo()
    status = "PASSED" if result.overall_passed else "FAILED"
    click.echo(f"Overall: {status}")


@cli.group()
def profiles():
    pass


@profiles.command("list")
@click.option("--profiles", "profiles_path", default=None, help="Path to profiles YAML")
def profiles_list(profiles_path):
    try:
        pm = ProfileManager(profiles_path=profiles_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    for name in sorted(pm.list_profiles()):
        profile = pm.get_profile(name)
        desc = profile.get("description", "")
        click.echo(f"  {name:<25} {desc}")


@profiles.command("show")
@click.argument("name")
@click.option("--profiles", "profiles_path", default=None, help="Path to profiles YAML")
def profiles_show(name, profiles_path):
    try:
        pm = ProfileManager(profiles_path=profiles_path)
        profile = pm.get_profile(name)
    except (FileNotFoundError, ProfileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    click.echo(f"\nProfile: {name}")
    click.echo(f"Description: {profile.get('description', 'N/A')}")
    click.echo(f"Metrics: {', '.join(profile['metrics'])}")
    click.echo(f"Composite Threshold: {profile.get('composite_threshold', 'N/A')}")
    click.echo()
    click.echo(f"{'Metric':<30} {'Weight':<10} {'Threshold':<10}")
    click.echo("-" * 50)
    for m in profile["metrics"]:
        w = profile["weights"].get(m, 0)
        c = profile["criteria"].get(m, 0)
        click.echo(f"{m:<30} {w:<10.2f} {c:<10.2f}")


@profiles.command("validate")
@click.argument("path")
def profiles_validate(path):
    try:
        ProfileManager(profiles_path=path)
        click.echo(f"Valid: {path}")
    except (ProfileValidationError, UnknownMetricError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
```

- [ ] **Step 7: Run CLI tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 8: Run all tests for regression**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add geneval/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: add Click CLI with evaluate, profiles list/show/validate commands"
```

---

### Task 10: Lint, Format, and Final Validation

**Files:**
- All modified files

- [ ] **Step 1: Run formatter**

Run: `uv run black .`
Run: `uv run isort .`

- [ ] **Step 2: Run linter**

Run: `uv run ruff check geneval/ tests/`

Fix any issues reported.

- [ ] **Step 3: Run full test suite with coverage**

Run: `uv run pytest tests/ -v --cov=geneval --cov-report=term-missing`

Expected: All tests PASS, good coverage on new modules.

- [ ] **Step 4: Test CLI manually**

Run: `uv run geneval profiles list`
Expected: Lists rag_default, strict, smoke_test

Run: `uv run geneval profiles show rag_default`
Expected: Shows rag_default profile details with metrics, weights, thresholds

- [ ] **Step 5: Commit formatting fixes (if any)**

```bash
git add -A
git commit -m "style: apply formatting to profile evaluation modules"
```

- [ ] **Step 6: Run pre-commit hooks**

Run: `uv run pre-commit run --all-files`

Fix any issues and commit.

---
