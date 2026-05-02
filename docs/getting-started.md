# Getting Started with GenEval

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM API key (OpenAI, Anthropic, Google Gemini, or a local model via Ollama/vLLM)

## Installation

```bash
git clone https://github.com/savitharaghunathan/gen-eval.git
cd gen-eval
uv sync
```

Set your API key(s) as environment variables:

```bash
# Use whichever provider you want as the judge LLM
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
# or
export GEMINI_API_KEY="your-key"
```

## Configure Your LLM

GenEval needs a judge LLM to score your evaluation metrics. Edit `config/llm_config.yaml` and set exactly one provider as the default:

```yaml
providers:
  openai:
    enabled: true
    default: true              # this provider will be used as the judge
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o"

  gemini:
    enabled: false
    default: false
    api_key_env: "GEMINI_API_KEY"
    model: "gemini-2.5-flash"

settings:
  temperature: 0.1
  max_tokens: 4096
```

Only one provider should have `default: true`. See `config/llm_config.yaml` for all supported providers.

## Prepare Test Data

Create a YAML file with your test cases. Each case needs the question, the model's response, the retrieved context, and a reference answer:

```yaml
# my_test_data.yaml
test_cases:
  - id: "test_001"
    user_input: "What is the capital of France?"
    retrieved_contexts: "France is a country in Western Europe. Its capital is Paris."
    response: "The capital of France is Paris."
    reference: "Paris"

  - id: "test_002"
    user_input: "What is 2+2?"
    retrieved_contexts: "Basic arithmetic: 2+2 equals 4."
    response: "2+2 equals 4."
    reference: "4"
```

## Run Your First Evaluation

Use the CLI with a built-in profile. The `smoke_test` profile is a quick check with relaxed thresholds:

```bash
geneval evaluate --profile smoke_test --data my_test_data.yaml --config config/llm_config.yaml --format table
```

Output:

```
Profile: smoke_test
Pass Rate: 100% (2/2)

Metric                         Mean Score   Threshold
------------------------------------------------------
faithfulness                   0.9500       0.50
answer_relevancy               0.9200       0.50

Overall: PASSED
```

The CLI exits with code 0 if all test cases pass, code 1 if any fail -- making it ready for CI pipelines.

## Understanding the Output

Each test case is evaluated against the profile's metrics:

- **Per-metric score**: a value between 0.0 and 1.0 from the LLM judge
- **Threshold**: the minimum score required (from the profile's `criteria`)
- **Weight**: how much this metric contributes to the composite score
- **Composite score**: `sum(weight * score)` across all metrics
- **Composite threshold**: the minimum composite score required
- **Overall pass**: a test case passes only when the composite score meets the threshold AND every individual metric meets its threshold

## Built-in Profiles

GenEval ships with three profiles:

| Profile | Metrics | Composite Threshold | Use Case |
|---------|---------|:-------------------:|----------|
| `smoke_test` | faithfulness, answer_relevancy | 0.50 | Quick sanity checks |
| `rag_default` | faithfulness, answer_relevancy, context_recall | 0.70 | General RAG evaluation |
| `strict` | faithfulness, answer_relevancy, context_recall, context_precision | 0.85 | Production deployments |

View details of any profile:

```bash
geneval profiles show strict
```

## Create a Custom Profile

Create a `profiles.yaml` file:

```yaml
profiles:
  medical:
    description: "Medical domain -- high faithfulness required"
    metrics:
      - faithfulness
      - answer_relevancy
      - context_recall
    weights:
      faithfulness: 0.6
      answer_relevancy: 0.2
      context_recall: 0.2
    criteria:
      faithfulness: 0.95
      answer_relevancy: 0.8
      context_recall: 0.8
    composite_threshold: 0.9
```

Run with it:

```bash
geneval evaluate --profile medical --data my_test_data.yaml --profiles profiles.yaml --config config/llm_config.yaml
```

Validate your profiles file without running an evaluation:

```bash
geneval profiles validate profiles.yaml
```

See `config/profiles_example.yaml` and `config/profiles_example.json` for more examples.

## Use Policies for Environment Overrides

Policies let you reuse a profile with different thresholds per environment. Add them to the same profiles file:

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

```bash
# Strict gate for CI
geneval evaluate --policy ci_gate --data test_data.yaml --profiles profiles.yaml

# Same metrics/weights, default thresholds for nightly
geneval evaluate --policy nightly --data test_data.yaml --profiles profiles.yaml
```

## Python API

Use GenEval directly in your code:

```python
from geneval import GenEvalFramework

framework = GenEvalFramework(config_path="config/llm_config.yaml")

# Single test case with a profile
result = framework.evaluate_profile(
    profile="rag_default",
    question="What is the capital of France?",
    response="The capital of France is Paris.",
    reference="Paris",
    retrieval_context="France is a country in Western Europe. Its capital is Paris.",
)

print(f"Passed: {result.overall_passed}")
print(f"Composite score: {result.composite_score:.2f}")
for mr in result.metric_results:
    print(f"  {mr.name}: {mr.score:.2f} (threshold: {mr.threshold})")

# Batch evaluation from a data file
batch = framework.evaluate_profile_batch(
    data_path="my_test_data.yaml",
    profile="strict",
)
print(f"Pass rate: {batch.pass_rate:.0%}")
```

## Next Steps

- [Architecture](architecture.md) -- understand how the components fit together
- [README](../README.md) -- full reference for all metrics, providers, CLI options, and output formats
- `config/profiles_example.yaml` -- more profile examples (medical, legal, CI, full suite)
