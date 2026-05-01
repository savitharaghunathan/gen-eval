# Profile-Driven Evaluation Framework

## Background

GenEval provides a unified evaluation framework for Generative AI applications, abstracting over multiple evaluation tools (RAGAS, DeepEval) with a consistent API. However, users currently must specify metrics explicitly in every `evaluate()` call, and there is no mechanism to define reusable evaluation configurations, enforce quality thresholds, or integrate with CI/CD pipelines as a pass/fail gate.

The profile-driven evaluation system addresses this gap by introducing declarative evaluation profiles that bundle metrics, weights, and criteria into reusable, shareable configurations.

## Current State

- **GenEvalFramework** orchestrates evaluation across RAGAS (7 metrics) and DeepEval (5 metrics) adapters
- **LLMManager** handles multi-provider LLM configuration via YAML
- **Pydantic schemas** define Input, Output, and MetricResult models
- **153 unit tests** with CI/CD pipelines for testing, linting, security scanning, and PyPI release
- **No CLI** beyond `demo_interactive.py` — designed as a library for import
- **No profile/config system** for evaluation workflows — metrics must be specified per call

## Proposal

Add a profile-driven evaluation layer that enables:

1. **Reusable profiles** — define metrics, weights, and criteria once in YAML, reuse across teams and workflows
2. **Automatic adapter routing** — users specify abstract metric names; the framework selects the best tool
3. **Composite scoring with quality gates** — weighted scores and per-metric thresholds produce pass/fail verdicts
4. **Evaluation policies** — CI/CD-friendly profile selectors with runtime overrides
5. **Built-in profiles** — common configurations shipped with the package
6. **CLI entry point** — `geneval evaluate --profile <name>` for pipeline integration

## Design

### Profile Schema & YAML Format

Profiles are defined in a dedicated YAML file (separate from `llm_config.yaml`). The file has two top-level keys: `profiles` and `policies`.

```yaml
# eval_profiles.yaml

profiles:
  rigorous_rag:
    description: "Strict RAG evaluation prioritizing faithfulness"
    metrics:
      - faithfulness
      - answer_relevancy
    weights:
      faithfulness: 0.7
      answer_relevancy: 0.3
    criteria:
      faithfulness: 0.8
      answer_relevancy: 0.9
    composite_threshold: 0.85

  smoke_test:
    description: "Quick sanity check across core RAG dimensions"
    metrics:
      - faithfulness
      - answer_relevancy
      - context_recall
    weights:
      faithfulness: 0.4
      answer_relevancy: 0.2
      context_recall: 0.4
    criteria:
      faithfulness: 0.8
      answer_relevancy: 0.9
      context_recall: 0.7

policies:
  ci_gate:
    profile: rigorous_rag
    overrides:
      criteria:
        faithfulness: 0.9
      composite_threshold: 0.9
  nightly:
    profile: smoke_test
```

**Policy overrides:** Policies can override `criteria` (per-metric thresholds) and `composite_threshold`. They cannot override `metrics` or `weights` — changing what is measured or how it's weighted constitutes a different profile.

**Validation rules:**

- Weights must sum to 1.0
- Every metric in `metrics` must have a corresponding weight and criteria entry
- Metric names must exist in the adapter capability registry
- `composite_threshold` defaults to the weighted average of individual criteria if omitted

### Adapter Capability Registry & Auto-Routing

A static mapping from abstract metric names to adapter implementations with priority ordering:

```python
METRIC_REGISTRY = {
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
```

**Auto-routing logic:**

1. Look up the metric in `METRIC_REGISTRY`
2. Pick the adapter with the lowest priority number (highest priority)
3. If that adapter fails to initialize, fall back to the next
4. If no adapter is available, raise `UnknownMetricError`

### ProfileManager

The core module that sits between the user and `GenEvalFramework`. It receives a `GenEvalFramework` instance via constructor injection.

**Responsibilities:**

1. **Load & validate profiles** — parse YAML, validate weights sum to 1.0, check metrics exist in registry
2. **Load & merge policies** — resolve a policy to its base profile + overrides
3. **Load built-in profiles** — ship default profiles bundled with the package, merge with user-defined ones (user profiles take precedence on name collision)
4. **Orchestrate evaluation** — for each metric in the profile, resolve the adapter via the registry, call `framework.evaluate()`, collect `MetricResult` objects
5. **Score & judge** — compute weighted composite score, check individual criteria, produce pass/fail verdict

**Scoring logic:**

```
For each metric m in profile:
    score[m] = evaluate(m)
    passed[m] = score[m] >= criteria[m]

composite_score = sum(weights[m] * score[m] for m in metrics)
composite_passed = composite_score >= composite_threshold

overall_passed = composite_passed AND all(passed[m] for m in metrics)
```

A result fails if either the composite score is below the composite threshold OR any individual metric fails its threshold. This prevents a catastrophic failure in one metric from being masked by others.

### Output Schema

```python
class MetricEvaluation(BaseModel):
    name: str                    # e.g., "faithfulness"
    score: float                 # raw score from adapter (0.0 - 1.0)
    threshold: float             # criteria threshold from profile
    passed: bool                 # score >= threshold
    weight: float                # weight from profile
    weighted_score: float        # weight * score
    adapter: str                 # which adapter was used (e.g., "ragas")
    details: Optional[str]       # explanation from the adapter, if available

class ProfileResult(BaseModel):
    profile_name: str            # e.g., "rigorous_rag"
    policy_name: Optional[str]   # e.g., "ci_gate" if invoked via policy
    overall_passed: bool         # composite_passed AND all individual passed
    composite_score: float       # weighted composite score
    composite_threshold: float   # threshold for composite
    composite_passed: bool       # composite_score >= composite_threshold
    metric_results: list[MetricEvaluation]
    metadata: dict               # timestamps, LLM info, framework version
```

`ProfileResult` uses Pydantic's `.model_dump_json()` for JSON serialization.

### CLI Interface

A Click-based CLI registered as a console script in `pyproject.toml`.

**Commands:**

```bash
# Evaluate using a profile
geneval evaluate --profile rigorous_rag \
                 --data test_data.yaml \
                 --config config/llm_config.yaml \
                 --profiles eval_profiles.yaml \
                 --output results.json

# Evaluate using a policy
geneval evaluate --policy ci_gate --data test_data.yaml

# List available profiles
geneval profiles list --profiles eval_profiles.yaml

# Show profile details
geneval profiles show rigorous_rag

# Validate a profiles file
geneval profiles validate eval_profiles.yaml
```

**Key flags:**

- `--profile` / `--policy`: mutually exclusive, one required for `evaluate`
- `--data`: path to test data YAML
- `--config`: LLM config path (defaults to `./config/llm_config.yaml`)
- `--profiles`: profiles YAML path (defaults to `./eval_profiles.yaml`)
- `--output`: write JSON results to file (default: stdout)
- `--format`: `json` (default) or `table` (human-readable summary)

**Exit codes:** 0 = all passed, 1 = evaluation failed criteria, 2 = configuration/runtime error.

### Built-in Profiles

Shipped as `geneval/profiles/default_profiles.yaml` inside the package:

```yaml
profiles:
  rag_default:
    description: "Balanced RAG evaluation for general use"
    metrics: [faithfulness, answer_relevancy, context_recall]
    weights: {faithfulness: 0.4, answer_relevancy: 0.3, context_recall: 0.3}
    criteria: {faithfulness: 0.7, answer_relevancy: 0.7, context_recall: 0.7}
    composite_threshold: 0.7

  strict:
    description: "High-bar evaluation for production deployments"
    metrics: [faithfulness, answer_relevancy, context_recall, context_precision]
    weights: {faithfulness: 0.35, answer_relevancy: 0.25, context_recall: 0.2, context_precision: 0.2}
    criteria: {faithfulness: 0.9, answer_relevancy: 0.85, context_recall: 0.8, context_precision: 0.8}
    composite_threshold: 0.85

  smoke_test:
    description: "Fast sanity check with relaxed thresholds"
    metrics: [faithfulness, answer_relevancy]
    weights: {faithfulness: 0.5, answer_relevancy: 0.5}
    criteria: {faithfulness: 0.5, answer_relevancy: 0.5}
    composite_threshold: 0.5
```

Built-in profiles are loaded first. User-defined profiles override built-ins on name collision.

### API Surface

New method on `GenEvalFramework`:

```python
framework = GenEvalFramework(config_path="config/llm_config.yaml")

result = framework.evaluate_profile(
    profile="rigorous_rag",
    profiles_path="eval_profiles.yaml",
    question="What is the capital of France?",
    response="Paris is the capital of France.",
    reference="Paris",
    retrieval_context="France is a country in Europe. Its capital city is Paris.",
)

# Or via policy
result = framework.evaluate_profile(
    policy="ci_gate",
    profiles_path="eval_profiles.yaml",
    question=...,
    response=...,
    reference=...,
    retrieval_context=...,
)

print(result.overall_passed)        # True/False
print(result.composite_score)       # 0.87
print(result.model_dump_json())     # full JSON output
```

The existing `evaluate()` method remains unchanged for backwards compatibility.

## New Files

| File | Purpose |
|------|---------|
| `geneval/profile_manager.py` | ProfileManager class — loading, validation, orchestration, scoring |
| `geneval/metric_registry.py` | METRIC_REGISTRY mapping and auto-routing logic |
| `geneval/profiles/default_profiles.yaml` | Built-in evaluation profiles |
| `geneval/profiles/__init__.py` | Package marker for profile resources |
| `geneval/cli.py` | Click-based CLI entry point |
| `geneval/exceptions.py` | Custom exceptions (ProfileValidationError, UnknownMetricError, ProfileNotFoundError) |
| `tests/test_profile_manager.py` | Unit tests for ProfileManager |
| `tests/test_metric_registry.py` | Unit tests for MetricRegistry |
| `tests/test_cli.py` | CLI tests using Click's CliRunner |

## Modified Files

| File | Change |
|------|--------|
| `geneval/framework.py` | Add `evaluate_profile()` method that delegates to ProfileManager |
| `geneval/schemas.py` | Add `MetricEvaluation` and `ProfileResult` Pydantic models |
| `geneval/__init__.py` | Export new public classes (ProfileManager, ProfileResult, MetricEvaluation) |
| `pyproject.toml` | Add `click` dependency, register `geneval` console script entry point |

## Error Handling

| Error | Exception | Behavior |
|-------|-----------|----------|
| Invalid profile YAML (weights don't sum to 1.0, missing fields) | `ProfileValidationError` | Raised at profile load time with specific message |
| Unknown metric name | `UnknownMetricError` | Raised at profile load time, lists available metrics |
| Policy references unknown profile | `ProfileNotFoundError` | Raised at policy resolution time |
| Adapter runtime failure | Warning logged | Falls back to next adapter in registry; if all fail, metric marked as errored (score=None, passed=False), evaluation continues |

## Testing Strategy

- **Unit tests for ProfileManager** — loading, validation, weight checks, policy merging, scoring logic (all mocked, no real adapter calls)
- **Unit tests for MetricRegistry** — resolution, priority ordering, fallback behavior
- **Unit tests for ProfileResult schema** — serialization, pass/fail logic edge cases
- **Integration tests** — end-to-end `evaluate_profile()` with mocked adapters returning known scores, verifying the full pipeline produces correct `ProfileResult`
- **CLI tests** — using Click's `CliRunner` to test command parsing, exit codes, output format

Follows existing test patterns: pytest, markers for unit/integration, coverage reporting.

## Security Implications

- Profile YAML files are loaded via `yaml.safe_load()` — no code execution risk
- No new credentials or secrets introduced — profiles reference metrics, not providers
- CLI reads files from user-specified paths — standard file system access, no elevation
- Built-in profiles are read-only resources bundled with the package
