# GenEval

A unified evaluation framework for RAG (Retrieval-Augmented Generation) applications. GenEval provides a standardized interface for evaluating generative AI models using multiple evaluation frameworks including RAGAS and DeepEval.

## Overview

GenEval solves the problem of fragmented evaluation approaches in the RAG ecosystem. Instead of learning different APIs and managing separate evaluation workflows, GenEval provides a single interface that works with multiple evaluation frameworks.

Key benefits:
- Unified API for RAGAS and DeepEval metrics
- Consistent output format across all frameworks
- Support for 8 unique evaluation metrics
- Clean JSON output for easy integration

## Supported Metrics

GenEval supports 8 unique metrics across both frameworks:

**RAGAS Metrics:**
- context_precision
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

Note: Some metrics like `faithfulness`, `context_precision`, and `context_recall` are available in both frameworks, giving you 11 total evaluations from 8 unique concepts.

## Installation

```bash
# Clone the repository
git clone <repository-url>
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

todo

## Output Format

GenEval returns consistent JSON output:

```json
{
  "ragas.faithfulness": {
    "adapter": "ragas",
    "metrics": [
      {
        "name": "faithfulness",
        "score": 0.85,
        "details": "High faithfulness score"
      }
    ]
  },
  "deepeval.faithfulness": {
    "adapter": "deepeval", 
    "metrics": [
      {
        "name": "faithfulness",
        "score": 0.92,
        "details": "Answer is factually consistent"
      }
    ]
  }
}
```

