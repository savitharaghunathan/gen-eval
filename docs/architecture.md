# GenEval Architecture

## Overview

GenEval is a multi-framework evaluation orchestrator for generative AI applications. It sits between the user and evaluation frameworks (RAGAS, DeepEval), providing a single interface for profile-driven evaluation with automatic adapter routing, weighted scoring, and pass/fail verdicts.

## Component Diagram

```
                          ┌─────────────────────────────┐
                          │        User / CI Pipeline    │
                          │  (CLI, Python API, scripts)  │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │      GenEvalFramework        │
                          │       (framework.py)         │
                          │                              │
                          │  evaluate()                  │
                          │  evaluate_profile()          │
                          │  evaluate_profile_batch()    │
                          └──┬──────────────┬───────────┘
                             │              │
              ┌──────────────▼───┐    ┌─────▼──────────────┐
              │  ProfileManager  │    │    LLMManager       │
              │(profile_manager) │    │  (llm_manager.py)   │
              │                  │    │                     │
              │ load profiles    │    │ provider config     │
              │ resolve policies │    │ API keys            │
              │ compute scores   │    │ model settings      │
              └──────┬───────────┘    └─────┬──────────────┘
                     │                      │
              ┌──────▼───────────┐          │
              │  MetricRegistry  │          │
              │(metric_registry) │          │
              │                  │          │
              │ resolve metric   │          │
              │ → adapter by     │          │
              │   priority       │          │
              └──────┬───────────┘          │
                     │                      │
         ┌───────────┴───────────┐          │
         │                       │          │
  ┌──────▼───────┐      ┌───────▼──────┐   │
  │ RAGASAdapter │      │DeepEvalAdapter│   │
  │(ragas_adapter│      │(deepeval_    │   │
  │         .py) │      │  adapter.py) │   │
  │              │      │              │   │
  │ 7 metrics    │      │ 5 metrics    │◄──┘
  │ uses ragas   │      │ uses deepeval│  LLM config
  │ collections  │      │ metrics API  │  passed to
  │ API          │      │              │  adapters
  └──────┬───────┘      └──────┬───────┘
         │                     │
         ▼                     ▼
   ┌──────────┐         ┌──────────┐
   │  RAGAS   │         │ DeepEval │
   │ library  │         │ library  │
   └──────────┘         └──────────┘
```

## Evaluation Flow

A profile evaluation follows this path:

```
1. User calls evaluate_profile(profile="strict", question=..., response=..., ...)
                │
2. ProfileManager loads profile definition
   ├── Built-in profiles from geneval/profiles/default_profiles.yaml
   └── User profiles merged on top (if profiles_path given)
                │
3. If policy requested, apply overrides (criteria, composite_threshold)
                │
4. For each metric in the profile:
   │
   ├── MetricRegistry resolves adapter candidates by priority
   │   e.g. "faithfulness" → [(ragas, faithfulness, pri=1), (deepeval, faithfulness, pri=2)]
   │
   ├── Try highest-priority adapter first
   │   └── Framework dispatches to adapter.evaluate(Input(...))
   │       └── Adapter calls LLM judge (configured via LLMManager)
   │           └── Returns score + explanation
   │
   └── If adapter fails, fall back to next candidate
                │
5. ProfileManager computes results:
   ├── Per-metric: score vs threshold → pass/fail
   ├── Composite: sum(weight_i * score_i) vs composite_threshold
   └── Overall: composite passed AND all individual metrics passed
                │
6. Returns ProfileResult (or BatchResult for multiple test cases)
```

## Adapter Architecture

Both adapters implement the same interface:

- **`__init__(llm_manager)`** -- initializes the evaluation framework with LLM config
- **`evaluate(input: Input) -> Output`** -- runs metrics and returns scores
- **`supported_metrics: list[str]`** -- metrics this adapter can evaluate
- **`close()`** -- cleanup (DeepEval only, for VLLMModel resources)

### RAGAS Adapter

Uses the RAGAS collections API (`ragas.metrics.collections`). Creates native SDK clients (OpenAI, Anthropic, etc.) and passes them to `ragas.llms.llm_factory()`. Each metric is scored individually via `metric.score(**kwargs)`.

Supported metrics: `faithfulness`, `answer_relevancy`, `context_precision_without_reference`, `context_precision_with_reference`, `context_recall`, `context_entity_recall`, `noise_sensitivity`

### DeepEval Adapter

Uses DeepEval's built-in model classes (`GPTModel`, `GeminiModel`, `AnthropicModel`, etc.) and metric classes. All metrics run in sync mode (`async_mode=False`). Each metric is scored via `metric.measure(test_case)`.

Supported metrics: `faithfulness`, `answer_relevancy`, `context_relevance`, `context_recall`, `context_precision`

## Metric Registry

The registry maps abstract metric names to adapter implementations with priority ordering:

| Metric | Primary (priority 1) | Fallback (priority 2) |
|--------|---------------------|----------------------|
| faithfulness | ragas | deepeval |
| answer_relevancy | ragas | deepeval |
| context_recall | ragas | deepeval |
| context_precision | ragas | deepeval |
| context_precision_with_reference | ragas | -- |
| context_entity_recall | ragas | -- |
| noise_sensitivity | ragas | -- |
| context_relevance | deepeval | -- |

When a profile requests "faithfulness", the registry returns both candidates. The framework tries RAGAS first; if it fails, it falls back to DeepEval.

## Profile System

### Profiles

A profile defines a reusable evaluation standard:

```yaml
strict:
  description: "High-bar evaluation for production"
  metrics: [faithfulness, answer_relevancy, context_recall, context_precision]
  weights:
    faithfulness: 0.35       # must sum to 1.0
    answer_relevancy: 0.25
    context_recall: 0.2
    context_precision: 0.2
  criteria:
    faithfulness: 0.9        # per-metric threshold
    answer_relevancy: 0.85
    context_recall: 0.8
    context_precision: 0.8
  composite_threshold: 0.85  # weighted score threshold
```

### Policies

A policy applies runtime overrides to a base profile. Only `criteria` and `composite_threshold` can be overridden -- metrics and weights stay the same.

```yaml
policies:
  ci_gate:
    profile: strict
    overrides:
      criteria:
        faithfulness: 0.95
      composite_threshold: 0.9
```

### Validation Rules

- Every metric in the list must have a corresponding weight and criterion
- Weights must sum to 1.0
- If `composite_threshold` is omitted, it defaults to the weighted sum of criteria

### Composite Scoring

```
composite_score = sum(weight[m] * score[m] for m in metrics)

overall_passed = (composite_score >= composite_threshold)
                 AND (all individual scores >= their criteria)
```

Both conditions must hold. A high composite score does not compensate for a single metric falling below its threshold.

## Key Source Files

| Component | File | Purpose |
|-----------|------|---------|
| Framework | `geneval/framework.py` | Main orchestrator, entry point |
| Profile Manager | `geneval/profile_manager.py` | Profile/policy loading, validation, scoring |
| Metric Registry | `geneval/metric_registry.py` | Metric-to-adapter resolution |
| Schemas | `geneval/schemas.py` | Pydantic models (Input, Output, ProfileResult, BatchResult) |
| CLI | `geneval/cli.py` | Click CLI commands |
| LLM Manager | `geneval/llm_manager.py` | Multi-provider LLM configuration |
| Exceptions | `geneval/exceptions.py` | ProfileValidationError, UnknownMetricError, ProfileNotFoundError |
| RAGAS Adapter | `geneval/adapters/ragas_adapter.py` | RAGAS framework integration |
| DeepEval Adapter | `geneval/adapters/deepeval_adapter.py` | DeepEval framework integration |
| Built-in Profiles | `geneval/profiles/default_profiles.yaml` | rag_default, strict, smoke_test |
| Example Profiles | `config/profiles_example.yaml` | Medical, legal, ci_quick, full_suite |
