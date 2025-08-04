# GenEval

A unified evaluation framework for RAG (Retrieval-Augmented Generation) applications. GenEval provides a standardized interface for evaluating generative AI models using multiple evaluation frameworks including RAGAS and DeepEval.

## Overview

GenEval solves the problem of fragmented evaluation approaches in the RAG ecosystem. Instead of learning different APIs and managing separate evaluation workflows, GenEval provides a single interface that works with multiple evaluation frameworks.

Key benefits:
- Unified API for RAGAS and DeepEval metrics
- Consistent output format across all frameworks
- Support for 9 unique evaluation metrics
- Clean JSON output for easy integration
- config-driven LLM management supporting OpenAI, Anthropic, Google Gemini, and Ollama

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

# Set your API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## Configuration

GenEval uses a YAML configuration file for LLM provider management. Create `config/llm_config.yaml`:

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

settings:
  temperature: 0.1
  max_tokens: 1000
  timeout: 30
```

**Important:** Only one provider should have `default: true`. The framework will automatically select the default provider.

## Quick Start

```python
from geneval import GenEvalFramework, LLMManager

# Initialize LLM manager (uses config file)
llm_manager = LLMManager()
llm_manager.select_provider()  # Selects default provider

# Initialize framework
framework = GenEvalFramework(llm_manager=llm_manager)

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
```

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
from geneval import GenEvalFramework, LLMManager
# All INFO, WARNING, and ERROR logs will now be visible
```

## Interactive Demo

Run the interactive demo to test the framework:

```bash
# Make sure you have the config file set up first
uv run python demo_interactive.py
```

The demo allows you to:
- Use config-driven LLM provider selection
- Select specific metrics or run all metrics
- Control the number of test cases (1-10)
- Get clean JSON output with detailed statistics

## Project Structure

```
gen-eval/
├── pyproject.toml
├── README.md
├── config/
│   └── llm_config.yaml        # LLM provider configuration
├── geneval/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models
│   ├── framework.py            # Main evaluation framework
│   ├── llm_manager.py          # LLM provider management
│   ├── normalization.py        # Score normalization utilities
│   └── adapters/
│       ├── ragas_adapter.py    # RAGAS integration
│       └── deepeval_adapter.py # DeepEval integration
├── tests/
│   ├── test_data.yaml          # Test dataset
│   ├── test_data_clean.yaml    # Clean test dataset
│   ├── test_framework.py       # Unit tests (83 tests)
│   └── test_integration.py     # Integration tests (8 tests)
├── demo_interactive.py         # Interactive demo
└── run_tests.py               # Batch testing script
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

- **Unit Tests (83 tests)**: Fast, isolated tests with mocked dependencies
  - Schema validation tests
  - LLM manager tests 
  - Adapter functionality tests
  - Framework integration tests
  - Error handling tests

- **Integration Tests (8 tests)**: End-to-end tests with real external dependencies
  - Complete evaluation workflows
  - Real API calls (when keys available)
  - Performance and reliability tests
  - Real-world usage scenarios

### Test Structure

```
tests/
├── test_framework.py      # Unit tests (83 tests)
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

GenEval returns consistent JSON output with LLM provider information:

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
      "evaluation_successful": true,
      "llm_provider": "openai",
      "llm_model": "gpt-4o-mini"
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
      "test_case_count": 1,
      "llm_provider": "openai",
      "llm_model": "gpt-4o-mini"
    }
  }
}
```

