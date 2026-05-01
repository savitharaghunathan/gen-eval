# GenEval

A unified evaluation framework for Gen AI applications. GenEval provides a standardized interface for evaluating generative AI models using multiple evaluation frameworks including RAGAS and DeepEval, with profile-driven evaluation for reproducible, reusable quality gates.

## Overview

GenEval solves the problem of fragmented evaluation approaches in the RAG ecosystem. Instead of learning different APIs and managing separate evaluation workflows, GenEval provides a single interface that works with multiple evaluation frameworks.

Key benefits:
- **Profile-driven evaluation** with reusable profiles, weighted scoring, and pass/fail verdicts
- **Automatic adapter routing** -- metrics resolve to the best available framework (RAGAS or DeepEval) without manual tool selection
- **CI/CD integration** -- exit codes, JSON output, and policy overrides for pipeline gates
- Unified API for RAGAS and DeepEval metrics
- Consistent output format across all frameworks
- Support for 8 unique evaluation metrics
- Config-driven LLM management supporting OpenAI, Anthropic, Google Gemini, Ollama, DeepSeek, Amazon Bedrock, Azure OpenAI, and vLLM

## Supported Metrics

GenEval supports 8 unique metrics through its metric registry. Each metric automatically resolves to the best available adapter:

| Metric | RAGAS | DeepEval |
|--------|:-----:|:--------:|
| faithfulness | 1st | 2nd |
| answer_relevancy | 1st | 2nd |
| context_recall | 1st | 2nd |
| context_precision | 1st | 2nd |
| context_precision_with_reference | yes | -- |
| context_entity_recall | yes | -- |
| noise_sensitivity | yes | -- |
| context_relevance | -- | yes |

When a metric is available in both frameworks, the registry tries the higher-priority adapter first and falls back to the other if it fails.

## Installation

### From Source

```bash
git clone https://github.com/savitharaghunathan/gen-eval.git
cd gen-eval
uv sync

# Set your API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# For vLLM servers
export VLLM_BASE_URL="https://your-vllm-server.com"
export VLLM_API_PATH="/v1"

# For AWS Bedrock (optional)
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
```

### Development Setup

```bash
uv sync --dev --all-extras
uv run pre-commit install
make lint
make test
```

## Configuration

GenEval requires an LLM configuration file. **API keys are read from environment variables -- never hardcoded.**

Create `llm_config.yaml`:

```yaml
providers:
  openai:
    enabled: true
    default: true
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o-mini"

  anthropic:
    enabled: true
    default: false
    api_key_env: "ANTHROPIC_API_KEY"
    model: "claude-3-5-haiku-20241022"

  gemini:
    enabled: true
    default: false
    api_key_env: "GOOGLE_API_KEY"
    model: "gemini-1.5-flash"

  ollama:
    enabled: true
    default: false
    base_url: "http://localhost:11434"
    model: "llama3.2"

  deepseek:
    enabled: true
    default: false
    api_key_env: "DEEPSEEK_API_KEY"
    model: "deepseek-chat"

  vllm:
    enabled: true
    default: false
    base_url_env: "VLLM_BASE_URL"
    api_path_env: "VLLM_API_PATH"
    api_key_env: "OPENAI_API_KEY"
    model: "your-model-name"
    ssl_verify: false

  amazon_bedrock:
    enabled: true
    default: false
    model: "anthropic.claude-3-sonnet-20240229-v1:0"
    region_name: "us-east-1"

  azure_openai:
    enabled: true
    default: false
    model: "gpt-4"
    deployment_name: "your-deployment-name"
    azure_openai_api_key: "your-azure-openai-api-key"
    openai_api_version: "2025-01-01-preview"
    azure_endpoint: "https://your-resource.openai.azure.com/"

settings:
  temperature: 0.1
  max_tokens: 1000
  timeout: 30
```

**Important:** Only one provider should have `default: true`.

A sample config is available at `config/llm_config.yaml`.

## Profile-Driven Evaluation

Profiles let you define evaluation standards once and reuse them across teams, workflows, and CI pipelines.

### What's in a Profile

A profile specifies:
- **Metrics** -- which metrics to evaluate
- **Weights** -- how much each metric contributes to the composite score (must sum to 1.0)
- **Criteria** -- minimum threshold for each individual metric
- **Composite threshold** -- minimum weighted composite score to pass

### Built-in Profiles

GenEval ships with three profiles:

| Profile | Metrics | Composite Threshold | Use Case |
|---------|---------|:-------------------:|----------|
| `rag_default` | faithfulness, answer_relevancy, context_recall | 0.70 | General RAG evaluation |
| `strict` | faithfulness, answer_relevancy, context_recall, context_precision | 0.85 | Production deployments |
| `smoke_test` | faithfulness, answer_relevancy | 0.50 | Quick sanity checks |

### Custom Profiles

Create a `profiles.yaml` file:

```yaml
profiles:
  medical_rag:
    description: "High-accuracy medical Q&A evaluation"
    metrics:
      - faithfulness
      - answer_relevancy
      - context_recall
    weights:
      faithfulness: 0.5
      answer_relevancy: 0.3
      context_recall: 0.2
    criteria:
      faithfulness: 0.95
      answer_relevancy: 0.85
      context_recall: 0.80
    composite_threshold: 0.90
```

### Policies

Policies let you create runtime variations of a profile without redefining metrics or weights. Only `criteria` and `composite_threshold` can be overridden:

```yaml
profiles:
  rag_default:
    metrics: [faithfulness, answer_relevancy, context_recall]
    weights: {faithfulness: 0.4, answer_relevancy: 0.3, context_recall: 0.3}
    criteria: {faithfulness: 0.7, answer_relevancy: 0.7, context_recall: 0.7}
    composite_threshold: 0.7

policies:
  ci_gate:
    profile: rag_default
    overrides:
      criteria:
        faithfulness: 0.9
      composite_threshold: 0.85

  nightly:
    profile: rag_default
```

### Pass/Fail Logic

A test case passes when **both** conditions are met:
1. Every metric score meets its individual criteria threshold
2. The weighted composite score meets the composite threshold

## CLI Usage

### Run Evaluation

```bash
# Evaluate with a profile
geneval evaluate --profile rag_default --data test_data.yaml --config llm_config.yaml

# Evaluate with a policy
geneval evaluate --policy ci_gate --data test_data.yaml --profiles profiles.yaml

# Output as table
geneval evaluate --profile strict --data test_data.yaml --format table

# Write results to file
geneval evaluate --profile rag_default --data test_data.yaml --output results.json
```

The CLI exits with code 0 if all test cases pass, code 1 if any fail.

### Manage Profiles

```bash
# List all available profiles
geneval profiles list

# List with custom profiles
geneval profiles list --profiles profiles.yaml

# Show profile details
geneval profiles show rag_default

# Validate a profiles file
geneval profiles validate profiles.yaml
```

### Test Data Format

```yaml
test_cases:
  - id: "test_001"
    user_input: "What is the capital of France?"
    retrieved_contexts: "France is a country in Europe. Its capital city is Paris."
    response: "Paris is the capital of France."
    reference: "Paris"

  - id: "test_002"
    user_input: "What is 2+2?"
    retrieved_contexts: "Basic arithmetic: 2+2 equals 4."
    response: "2+2 equals 4."
    reference: "4"
```

## Python API

### Profile-Based Evaluation

```python
from geneval import GenEvalFramework

framework = GenEvalFramework(config_path="llm_config.yaml")

# Single test case
result = framework.evaluate_profile(
    profile="rag_default",
    question="What is the capital of France?",
    response="Paris is the capital of France.",
    reference="Paris",
    retrieval_context="France is a country in Europe. Its capital city is Paris.",
)

print(f"Passed: {result.overall_passed}")
print(f"Composite score: {result.composite_score:.2f}")
for mr in result.metric_results:
    print(f"  {mr.name}: {mr.score:.2f} (threshold: {mr.threshold}, passed: {mr.passed})")

# Batch evaluation
batch = framework.evaluate_profile_batch(
    data_path="test_data.yaml",
    profile="strict",
)

print(f"Pass rate: {batch.pass_rate:.0%}")
print(f"Overall: {'PASSED' if batch.overall_passed else 'FAILED'}")
```

### Direct Metric Evaluation

```python
from geneval import GenEvalFramework

framework = GenEvalFramework(config_path="llm_config.yaml")

# Evaluate with explicit metrics (runs on all adapters that support them)
results = framework.evaluate(
    question="What is the capital of France?",
    response="Paris is the capital of France.",
    reference="Paris",
    retrieval_context="France is a country in Europe. Its capital city is Paris.",
    metrics=["faithfulness", "answer_relevancy", "context_precision"]
)
print(results)
```

## CI/CD Integration

GenEval is designed for pipeline integration. Use profiles to set quality gates:

```yaml
# .github/workflows/eval.yml
- name: Run evaluation
  run: |
    geneval evaluate \
      --profile strict \
      --data tests/eval_data.yaml \
      --config config/llm_config.yaml \
      --output eval_results.json
```

The exit code (0 = pass, 1 = fail) integrates directly with CI systems.

Use policies to vary thresholds per environment without changing the profile:

```bash
# Strict gate for production
geneval evaluate --policy prod_gate --data test_data.yaml

# Relaxed gate for development
geneval evaluate --policy dev_gate --data test_data.yaml
```

## Output Format

### Profile Evaluation (JSON)

```json
{
  "profile_name": "rag_default",
  "policy_name": null,
  "overall_passed": true,
  "composite_score": 0.87,
  "composite_threshold": 0.7,
  "composite_passed": true,
  "metric_results": [
    {
      "name": "faithfulness",
      "score": 0.95,
      "threshold": 0.7,
      "passed": true,
      "weight": 0.4,
      "weighted_score": 0.38,
      "adapter": "ragas",
      "details": "High faithfulness"
    }
  ],
  "metadata": {
    "timestamp": "2026-05-01T12:00:00+00:00"
  }
}
```

### Table Output

```
Profile: rag_default
Pass Rate: 100% (5/5)

Metric                         Mean Score   Threshold
------------------------------------------------------
faithfulness                   0.9200       0.70
answer_relevancy               0.8800       0.70
context_recall                 0.8500       0.70

Overall: PASSED
```

## Project Structure

```
gen-eval/
├── pyproject.toml
├── README.md
├── Makefile
├── config/
│   └── llm_config.yaml
├── geneval/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models (Input, Output, ProfileResult, BatchResult)
│   ├── framework.py            # Main evaluation framework
│   ├── llm_manager.py          # LLM provider management
│   ├── profile_manager.py      # Profile/policy loading, validation, scoring
│   ├── metric_registry.py      # Metric-to-adapter resolution
│   ├── exceptions.py           # ProfileValidationError, UnknownMetricError, ProfileNotFoundError
│   ├── cli.py                  # Click CLI (evaluate, profiles list/show/validate)
│   ├── normalization.py        # Score normalization utilities
│   ├── profiles/
│   │   └── default_profiles.yaml  # Built-in profiles (rag_default, strict, smoke_test)
│   └── adapters/
│       ├── ragas_adapter.py    # RAGAS integration
│       └── deepeval_adapter.py # DeepEval integration
├── tests/
│   ├── test_framework.py
│   ├── test_ragas_adapter.py
│   ├── test_deepeval_adapter.py
│   ├── test_llm_manager.py
│   ├── test_schemas.py
│   ├── test_schemas_profile.py
│   ├── test_profile_manager.py
│   ├── test_metric_registry.py
│   ├── test_cli.py
│   ├── test_data.yaml
│   └── test_data_clean.yaml
└── uv.lock
```

## Development

### Code Quality

- **Black** -- Code formatting
- **isort** -- Import sorting
- **Ruff** -- Fast linting
- **pre-commit** -- Git hooks for automated checks

### Available Commands

```bash
make install       # Install dependencies
make dev           # Install pre-commit hooks
make format        # Format code
make lint          # Run linting
make test          # Run tests with coverage
make test-ci       # Run tests without coverage (faster)
make pre-commit    # Run all hooks on all files
make pre-push      # Format + lint before pushing
make build         # Build the package
make security      # Run security checks
make clean         # Clean generated files
```

### Running Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=geneval --cov-report=term-missing
uv run pytest tests/test_profile_manager.py -v
uv run pytest tests/test_ragas_adapter.py -v
```

## Contributing

1. Fork and clone the repo
2. Set up dev environment: `make install && make dev`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make changes, add tests, verify: `make test && make lint`
5. Push and create a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/savitharaghunathan/gen-eval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/savitharaghunathan/gen-eval/discussions)
