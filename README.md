# GenEval

A unified evaluation framework for Gen AI applications. GenEval provides a standardized interface for evaluating generative AI models using multiple evaluation frameworks including RAGAS and DeepEval.

## Overview

GenEval solves the problem of fragmented evaluation approaches in the RAG ecosystem. Instead of learning different APIs and managing separate evaluation workflows, GenEval provides a single interface that works with multiple evaluation frameworks.

Key benefits:
- Unified API for RAGAS and DeepEval metrics
- Consistent output format across all frameworks
- Support for 9 unique evaluation metrics
- Clean JSON output for easy integration
- config-driven LLM management supporting OpenAI, Anthropic, Google Gemini, Ollama, DeepSeek, Amazon Bedrock, Azure OpenAI, and vLLM

## Supported Metrics

GenEval supports 9 unique metrics across both frameworks:

**RAGAS Metrics:**
- context_precision_without_reference
- context_precision_with_reference
- context_recall
- context_entity_recall
- noise_sensitivity
- answer_relevancy
- faithfulness

**DeepEval Metrics:**
- answer_relevancy
- context_relevance
- context_precision
- context_recall
- faithfulness

**Note**: Some metrics like `faithfulness`, `answer_relevancy`, and `context_recall` are available in both frameworks, giving you up to 12 total evaluations from 9 unique concepts.

## Installation

### From Source

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
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# For vLLM servers
export VLLM_BASE_URL="https://your-vllm-server.com"
export VLLM_API_PATH="/v1"

# For AWS Bedrock (optional)
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"

# Create your LLM configuration file
# You'll need to specify the path to this file when using the framework
```

### Development Setup

```bash
# Install development dependencies
uv sync --dev --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Run all checks
make lint
make test
```

## Configuration

GenEval requires you to specify the path to your LLM configuration file. **API keys are never hardcoded - they are read from environment variables for security.**

Create `llm_config.yaml` in your project directory:

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
    api_key_env: "OPENAI_API_KEY"  # If authentication required
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

**Important:** Only one provider should have `default: true`. The framework will automatically select the default provider.

### Sample Configuration File

You can find a complete working example of the `llm_config.yaml` file in the project repository:

- **Local path**: `config/llm_config.yaml` (in the project root)
- **GitHub**: [https://github.com/savitharaghunathan/gen-eval/tree/main/config/llm_config.yaml](https://github.com/savitharaghunathan/gen-eval/tree/main/config/llm_config.yaml)

The sample config includes all supported providers with proper settings. You can copy this file and modify it for your own use.

## Quick Start

```python
from geneval import GenEvalFramework

# Initialize framework (handles LLM manager internally)
framework = GenEvalFramework(config_path="path/to/your/llm_config.yaml")

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
from geneval import GenEvalFramework
# All INFO, WARNING, and ERROR logs will now be visible
```

## Interactive Demo

Run the interactive demo to test the framework:

```bash
# Make sure you have the config file set up first
# You can use the sample config at config/llm_config.yaml
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
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── Makefile                    # Development commands
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── config/
│   └── llm_config.yaml        # LLM provider configuration
├── .github/
│   └── workflows/
│       ├── ci.yml             # Continuous Integration workflow
│       └── release.yml        # Release workflow
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
│   ├── test_framework.py       # Framework tests (15 tests)
│   ├── test_ragas_adapter.py   # RAGAS adapter tests (48 tests)
│   ├── test_deepeval_adapter.py # DeepEval adapter tests (35 tests)
│   ├── test_llm_manager.py     # LLM manager tests (40 tests)
│   ├── test_schemas.py         # Schema validation tests (15 tests)
│   ├── test_data.yaml          # Test dataset
│   └── test_data_clean.yaml    # Clean test dataset
├── demo_interactive.py         # Interactive demo
└── uv.lock                     # Dependency lock file
```

## Development

### Code Quality

This project uses several tools to maintain code quality:

- **Black** - Code formatting
- **isort** - Import sorting
- **Ruff** - Fast linting and import sorting
- **pre-commit** - Git hooks for automated checks

### Available Commands

```bash
# Install development dependencies
make install

# Install pre-commit hooks
make dev

# Run all checks
make pre-commit

# Run format and lint before pushing
make pre-push

# Format code
make format

# Run linting
make lint

# Run tests with coverage (local development)
make test

# Run tests without coverage (faster, for CI)
make test-ci

# Build the package
make build

# Run security checks
make security

# Clean up generated files
make clean
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit:

```bash
# Install hooks (one-time setup)
make dev

# Run hooks on all files
make pre-commit
```

## Testing

### Running Tests

```bash
# Run all tests using uv
uv run pytest tests/ -v

# Run tests with coverage report
uv run pytest tests/ --cov=geneval --cov-report=term-missing

# Run specific test files
uv run pytest tests/test_framework.py -v
uv run pytest tests/test_ragas_adapter.py -v
uv run pytest tests/test_deepeval_adapter.py -v
uv run pytest tests/test_llm_manager.py -v
uv run pytest tests/test_schemas.py -v

# Run tests with verbose output
uv run pytest tests/ -v -s
```

### Test Structure

```
tests/
├── test_framework.py           # Framework tests (14 tests)
├── test_ragas_adapter.py       # RAGAS adapter tests (47 tests)
├── test_deepeval_adapter.py    # DeepEval adapter tests (34 tests)
├── test_llm_manager.py         # LLM manager tests (35 tests)
├── test_schemas.py             # Schema validation tests (17 tests)
├── test_data.yaml              # Test dataset
└── test_data_clean.yaml        # Clean test dataset
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

## Contributing

We welcome contributions! Please follow these steps:

### 1. Fork and Clone
```bash
git clone https://github.com/your-username/gen-eval.git
cd gen-eval
```

### 2. Set up Development Environment
```bash
# Install dependencies
make install

# Install pre-commit hooks
make dev

# Run all checks
make pre-commit
```

### 3. Make Changes
- Create a feature branch: `git checkout -b feature/your-feature`
- Make your changes
- Add tests for new functionality
- Ensure all tests pass: `make test`
- Ensure code quality: `make lint`

### 4. Submit Pull Request
- Push your branch: `git push origin feature/your-feature`
- Create a pull request on GitHub
- The CI will automatically run tests and checks

### Development Guidelines

- **Code Style**: Follow Black formatting and isort import sorting
- **Type Hints**: Use type hints for all function parameters and return values
- **Testing**: Add tests for new features and bug fixes
- **Documentation**: Update README and docstrings as needed
- **Commits**: Use clear, descriptive commit messages

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/savitharaghunathan/gen-eval/issues)
- **Discussions**: Join the conversation on [GitHub Discussions](https://github.com/savitharaghunathan/gen-eval/discussions)
