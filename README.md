# GenEval

A unified evaluation framework for RAG (Retrieval-Augmented Generation) applications. GenEval provides a standardized interface for evaluating generative AI models using multiple evaluation frameworks including RAGAS and DeepEval.

## Overview

GenEval solves the problem of fragmented evaluation approaches in the RAG ecosystem. Instead of learning different APIs and managing separate evaluation workflows, GenEval provides a single interface that works with multiple evaluation frameworks.

Key benefits:
- Unified API for RAGAS and DeepEval metrics
- Consistent output format across all frameworks
- Support for 9 unique evaluation metrics
- Clean JSON output for easy integration

## Supported Metrics

GenEval supports 9 unique metrics across both frameworks:

**RAGAS Metrics:**
- context_precision_without_reference
- context_precision_with_reference
- context_recall
- context_entity_recall
- noise_sensitivity
- response_relevancy
- faithfulness

**DeepEval Metrics:**
- answer_relevance
- context_relevance
- context_precision
- context_recall
- faithfulness

Note: Some metrics like `faithfulness`, `context_precision`, and `context_recall` are available in both frameworks, giving you 12 total evaluations from 9 unique concepts.

## Installation

```bash
# Clone the repository
git clone https://github.com/savitharaghunathan/gen-eval.git
cd gen-eval

# Install dependencies using uv
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"
```

## Quick Start

```python
from geneval import GenEvalFramework, LLMInitializer

# Initialize with OpenAI
llm_init = LLMInitializer(provider="openai")
framework = GenEvalFramework(llm_initializer=llm_init)

# Evaluate with multiple metrics
results = framework.evaluate(
    question="What is the capital of France?",
    response="Paris is the capital of France.",
    reference="Paris",
    retrieval_context="France is a country in Europe. Its capital city is Paris.",
    metrics=["faithfulness", "answer_relevance", "context_precision"]
)
# Results contain evaluations from both RAGAS and DeepEval
print(results)

### Logging Configuration

To see detailed logging output during evaluation, configure logging before importing the package:

```python
import logging

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now import and use the package
from geneval import GenEvalFramework, LLMInitializer
# All INFO, WARNING, and ERROR logs will now be visible
```

## Interactive Demo

Run the interactive demo to test the framework:

```bash
python demo_interactive.py
```

The demo allows you to:
- Choose between OpenAI and Anthropic providers
- Select specific metrics or run all metrics
- Control the number of test cases
- Get clean JSON output

## Project Structure

```
gen-eval/
├── pyproject.toml
├── README.md
├── geneval/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models
│   ├── framework.py            # Main evaluation framework
│   ├── llm.py                 # LLM initialization
│   ├── normalization.py       # Score normalization utilities
│   └── adapters/
│       ├── ragas_adapter.py   # RAGAS integration
│       └── deepeval_adapter.py # DeepEval integration
├── tests/
│   ├── test_data.yaml         # Test dataset
│   ├── test_data_clean.yaml   # Clean test dataset
│   └── test_framework.py      # Test suite
├── demo_interactive.py        # Interactive demo
└── run_tests.py              # Batch testing script
```

## Configuration

GenEval uses GPT-4o-mini by default for cost-effective evaluations. You can configure different providers:

```python
# OpenAI (default: gpt-4o-mini)
llm_init = LLMInitializer(provider="openai")

# Anthropic (default: claude-3-5-haiku)
llm_init = LLMInitializer(provider="anthropic")

# Auto-detect (tries OpenAI first, then Anthropic)
llm_init = LLMInitializer(provider="auto")
```

## Testing

### Running Tests

```bash
# Run all tests (unit + integration)
python run_tests.py

# Run only unit tests (fast, no external dependencies)
python run_tests.py --unit

# Run only integration tests (requires API keys)
python run_tests.py --integration

# Run tests with coverage report
python run_tests.py --coverage

# Run tests with verbose output
python run_tests.py --verbose
```

### Test Coverage

- **Unit Tests (40 tests)**: Fast, isolated tests with mocked dependencies
  - Schema validation tests
  - LLM initialization tests
  - Adapter functionality tests
  - Framework integration tests
  - Error handling tests
  - **Coverage: 95%**

- **Integration Tests (8 tests)**: End-to-end tests with real external dependencies
  - Complete evaluation workflows
  - Real API calls (when keys available)
  - Performance and reliability tests
  - Real-world usage scenarios

### Test Structure

```
tests/
├── test_framework.py      # Unit tests (40 tests)
├── test_integration.py    # Integration tests (8 tests)
├── test_data.yaml         # Test dataset (not included in repo)
```

### Test Data Format

To create your own test data, use this YAML format:

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

### Running Integration Tests

Integration tests require API keys to run:

```bash
# Set your API key
export OPENAI_API_KEY="your-openai-api-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Run integration tests
python run_tests.py --integration
```



## Output Format

GenEval returns consistent JSON output:

```json
{
  "ragas.faithfulness": {
    "adapter": "ragas",
    "metrics": [
      {
        "name": "faithfulness",
        "score": 1.0,
        "details": "RAGAS faithfulness evaluation"
      }
    ],
    "metadata": {
      "framework": "ragas",
      "total_metrics": 1,
      "evaluation_successful": true
    }
  },
  "deepeval.faithfulness": {
    "adapter": "deepeval",
    "metrics": [
      {
        "name": "faithfulness",
        "score": 1.0,
        "details": "Great job! There are no contradictions, so the actual output is fully faithful to the retrieval context."
      }
    ],
    "metadata": {
      "framework": "deepeval",
      "total_metrics": 1,
      "evaluation_successful": true,
      "test_case_count": 1
    }
  }
}

```

