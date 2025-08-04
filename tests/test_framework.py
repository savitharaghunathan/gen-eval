"""
Comprehensive test suite for the GenEval framework.

This module contains tests for all major components:
- Schemas (Input, Output, MetricResult)
- LLMInitializer
- RAGASAdapter
- DeepEvalAdapter
- GenEvalFramework
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit

from geneval.schemas import Input, Output, MetricResult
from geneval.llm import LLMInitializer
from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.adapters.deepeval_adapter import DeepEvalAdapter
from geneval.framework import GenEvalFramework


class TestSchemas:
    """Test cases for Pydantic schemas"""
    
    def test_input_schema_valid(self):
        """Test Input schema with valid data"""
        input_data = {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "retrieval_context": "Paris is the capital and largest city of France.",
            "reference": "Paris is the capital of France.",
            "metrics": ["faithfulness", "answer_relevance"]
        }
        
        input_obj = Input(**input_data)
        assert input_obj.question == "What is the capital of France?"
        assert input_obj.response == "The capital of France is Paris."
        assert input_obj.retrieval_context == "Paris is the capital and largest city of France."
        assert input_obj.reference == "Paris is the capital of France."
        assert input_obj.metrics == ["faithfulness", "answer_relevance"]
    
    def test_input_schema_missing_fields(self):
        """Test Input schema validation with missing required fields"""
        with pytest.raises(ValueError):
            Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                # Missing required fields
            )
    
    def test_metric_result_schema(self):
        """Test MetricResult schema"""
        metric_result = MetricResult(
            name="faithfulness",
            score=0.85,
            tool_name="ragas",
            details="RAGAS faithfulness evaluation"
        )
        
        assert metric_result.name == "faithfulness"
        assert metric_result.score == 0.85
        assert metric_result.tool_name == "ragas"
        assert metric_result.details == "RAGAS faithfulness evaluation"
    
    def test_output_schema(self):
        """Test Output schema"""
        metric_results = [
            MetricResult(
                name="faithfulness",
                score=0.85,
                tool_name="ragas",
                details="RAGAS faithfulness evaluation"
            ),
            MetricResult(
                name="answer_relevance",
                score=0.92,
                tool_name="deepeval",
                details="DeepEval answer relevance evaluation"
            )
        ]
        
        metadata = {
            "framework": "ragas",
            "total_metrics": 2,
            "evaluation_successful": True
        }
        
        output = Output(metrics=metric_results, metadata=metadata)
        assert len(output.metrics) == 2
        assert output.metadata["framework"] == "ragas"
        assert output.metadata["total_metrics"] == 2


class TestLLMInitializer:
    """Test cases for LLMInitializer"""
    
    def test_initialization_without_provider(self):
        """Test LLMInitializer initialization without provider"""
        initializer = LLMInitializer()
        assert initializer.selected_provider is None
        assert initializer.openai_client is None
        assert initializer.anthropic_client is None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('geneval.llm.OpenAI')
    def test_initialize_openai(self, mock_openai):
        """Test OpenAI initialization"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("openai")
        
        assert initializer.selected_provider == "openai"
        assert initializer.model_name == "gpt-4o-mini"
        assert initializer.openai_client == mock_client
        mock_openai.assert_called_once_with(api_key="test_key")
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch('geneval.llm.Anthropic')
    def test_initialize_anthropic(self, mock_anthropic):
        """Test Anthropic initialization"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("anthropic")
        
        assert initializer.selected_provider == "anthropic"
        assert initializer.model_name == "claude-3-5-haiku-20241022"
        assert initializer.anthropic_client == mock_client
        mock_anthropic.assert_called_once_with(api_key="test_key")
    
    def test_initialize_unsupported_provider(self):
        """Test initialization with unsupported provider"""
        initializer = LLMInitializer()
        with pytest.raises(ValueError, match="Unsupported provider"):
            initializer.initialize_provider("unsupported")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('geneval.llm.OpenAI')
    def test_initialize_auto_openai_available(self, mock_openai):
        """Test auto initialization with OpenAI available"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("auto")
        
        assert initializer.selected_provider == "openai"
        assert initializer.openai_client == mock_client
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_auto_no_providers(self):
        """Test auto initialization with no providers available"""
        initializer = LLMInitializer()
        with pytest.raises(ValueError, match="No LLM provider available"):
            initializer.initialize_provider("auto")
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch('geneval.llm.OpenAI')
    @patch('geneval.llm.Anthropic')
    def test_initialize_auto_openai_fails_anthropic_succeeds(self, mock_anthropic, mock_openai):
        """Test auto initialization where OpenAI fails but Anthropic succeeds"""
        # Mock OpenAI to fail
        mock_openai.side_effect = Exception("OpenAI failed")
        
        # Mock Anthropic to succeed
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("auto")
        
        assert initializer.selected_provider == "anthropic"
        assert initializer.anthropic_client == mock_anthropic_client
        assert initializer.model_name == "claude-3-5-haiku-20241022"
    
    def test_client_availability_checks(self):
        """Test client availability check methods"""
        initializer = LLMInitializer()
        
        # Initially no clients available
        assert not initializer.is_openai_available()
        assert not initializer.is_anthropic_available()
        
        # Mock clients
        initializer.openai_client = Mock()
        initializer.anthropic_client = Mock()
        
        assert initializer.is_openai_available()
        assert initializer.is_anthropic_available()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('geneval.llm.OpenAI')
    def test_configure_ragas_llm_openai(self, mock_openai):
        """Test RAGAS LLM configuration with OpenAI"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("openai")
        
        config = initializer.configure_ragas_llm()
        
        assert config["llm"] == mock_client
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o-mini"
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch('geneval.llm.Anthropic')
    def test_configure_deepeval_llm_anthropic(self, mock_anthropic):
        """Test DeepEval LLM configuration with Anthropic"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("anthropic")
        
        config = initializer.configure_deepeval_llm()
        
        assert config["llm"] == mock_client
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-3-5-haiku-20241022"
    
    def test_configure_ragas_llm_no_provider(self):
        """Test RAGAS LLM configuration with no provider selected"""
        initializer = LLMInitializer()
        
        config = initializer.configure_ragas_llm()
        
        assert config == {}
    
    def test_configure_deepeval_llm_no_provider(self):
        """Test DeepEval LLM configuration with no provider selected"""
        initializer = LLMInitializer()
        
        config = initializer.configure_deepeval_llm()
        
        assert config == {}
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('geneval.llm.OpenAI')
    def test_configure_ragas_llm_provider_not_available(self, mock_openai):
        """Test RAGAS LLM configuration when provider is set but client is not available"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("openai")
        
        # Manually set client to None to simulate unavailability
        initializer.openai_client = None
        
        config = initializer.configure_ragas_llm()
        
        assert config == {}
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch('geneval.llm.Anthropic')
    def test_configure_deepeval_llm_provider_not_available(self, mock_anthropic):
        """Test DeepEval LLM configuration when provider is set but client is not available"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        initializer = LLMInitializer()
        initializer.initialize_provider("anthropic")
        
        # Manually set client to None to simulate unavailability
        initializer.anthropic_client = None
        
        config = initializer.configure_deepeval_llm()
        
        assert config == {}


class TestRAGASAdapter:
    """Test cases for RAGASAdapter"""
    
    def test_initialization_without_llm(self):
        """Test RAGASAdapter initialization without LLM"""
        adapter = RAGASAdapter()
        assert adapter.llm_initializer is None
        assert adapter.supported_metrics == []
    
    @patch('geneval.adapters.ragas_adapter.LLMContextPrecisionWithoutReference')
    @patch('geneval.adapters.ragas_adapter.LLMContextPrecisionWithReference')
    @patch('geneval.adapters.ragas_adapter.LLMContextRecall')
    @patch('geneval.adapters.ragas_adapter.ContextEntityRecall')
    @patch('geneval.adapters.ragas_adapter.NoiseSensitivity')
    @patch('geneval.adapters.ragas_adapter.ResponseRelevancy')
    @patch('geneval.adapters.ragas_adapter.Faithfulness')
    def test_initialization_with_llm(self, mock_faithfulness, mock_response_relevancy, 
                                   mock_noise_sensitivity, mock_context_entity_recall,
                                   mock_context_recall, mock_context_precision_with_ref,
                                   mock_context_precision_without_ref):
        """Test RAGASAdapter initialization with LLM"""
        mock_llm = Mock()
        mock_llm.selected_provider = "openai"
        mock_llm.configure_ragas_llm.return_value = {"provider": "openai"}
        mock_llm.get_selected_provider.return_value = "openai"
        
        adapter = RAGASAdapter(mock_llm)
        
        expected_metrics = [
            "context_precision_without_reference",
            "context_precision_with_reference", 
            "context_recall",
            "context_entity_recall",
            "noise_sensitivity",
            "response_relevancy",
            "faithfulness"
        ]
        
        assert set(adapter.supported_metrics) == set(expected_metrics)
    
    @patch('geneval.adapters.ragas_adapter.LLMContextPrecisionWithoutReference')
    def test_initialization_with_llm_exception(self, mock_metric):
        """Test RAGASAdapter initialization with LLM when metric initialization fails"""
        # Mock metric to raise exception
        mock_metric.side_effect = Exception("Metric initialization failed")
        
        mock_llm = Mock()
        mock_llm.selected_provider = "openai"
        mock_llm.configure_ragas_llm.return_value = {"provider": "openai"}
        mock_llm.get_selected_provider.return_value = "openai"
        
        adapter = RAGASAdapter(mock_llm)
        
        # Should handle exception gracefully and have empty metrics
        assert adapter.available_metrics == {}
        assert adapter.supported_metrics == []
    
    def test_prepare_dataset(self):
        """Test dataset preparation"""
        adapter = RAGASAdapter()
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        dataset = adapter._prepare_dataset(input_data)
        
        assert dataset["question"][0] == "What is the capital of France?"
        assert dataset["answer"][0] == "The capital of France is Paris."
        assert dataset["contexts"][0] == ["Paris is the capital and largest city of France."]
        assert dataset["ground_truths"][0] == ["Paris is the capital of France."]
    
    def test_get_metrics_unsupported(self):
        """Test getting unsupported metrics"""
        adapter = RAGASAdapter()
        adapter.available_metrics = {}
        adapter.supported_metrics = []
        
        with pytest.raises(ValueError, match="Unsupported metric"):
            adapter._get_metrics(["unsupported_metric"])
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    @patch('geneval.adapters.ragas_adapter.Dataset')
    def test_evaluate_success(self, mock_dataset, mock_evaluate):
        """Test successful evaluation"""
        # Mock dataset
        mock_dataset.from_dict.return_value = Mock()
        
        # Mock evaluation results
        mock_results = Mock()
        mock_results.scores = [{"faithfulness": 0.85}]
        mock_evaluate.return_value = mock_results
        
        # Mock metrics
        mock_metric = Mock()
        
        adapter = RAGASAdapter()
        adapter.available_metrics = {"faithfulness": mock_metric}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is True
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "faithfulness"
        assert result.metrics[0].score == 0.85
        assert result.metrics[0].tool_name == "ragas"
    
    @patch('geneval.adapters.ragas_adapter.evaluate')
    def test_evaluate_exception(self, mock_evaluate):
        """Test evaluation with exception"""
        mock_evaluate.side_effect = Exception("Test error")
        
        adapter = RAGASAdapter()
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "ragas"
        assert result.metadata["evaluation_successful"] is False
        assert "Test error" in result.metadata["error"]
        assert len(result.metrics) == 0


class TestDeepEvalAdapter:
    """Test cases for DeepEvalAdapter"""
    
    def test_initialization_without_llm(self):
        """Test DeepEvalAdapter initialization without LLM"""
        adapter = DeepEvalAdapter()
        assert adapter.llm_initializer is None
        assert adapter.supported_metrics == []
    
    @patch('geneval.adapters.deepeval_adapter.AnswerRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRelevancyMetric')
    @patch('geneval.adapters.deepeval_adapter.FaithfulnessMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualRecallMetric')
    @patch('geneval.adapters.deepeval_adapter.ContextualPrecisionMetric')
    def test_initialization_with_llm(self, mock_context_precision, mock_context_recall,
                                   mock_faithfulness, mock_context_relevance, mock_answer_relevance):
        """Test DeepEvalAdapter initialization with LLM"""
        mock_llm = Mock()
        mock_llm.selected_provider = "openai"
        mock_llm.configure_deepeval_llm.return_value = {"provider": "openai"}
        mock_llm.get_selected_provider.return_value = "openai"
        
        adapter = DeepEvalAdapter(mock_llm)
        
        expected_metrics = [
            "answer_relevance",
            "context_relevance",
            "faithfulness",
            "context_recall",
            "context_precision"
        ]
        
        assert set(adapter.supported_metrics) == set(expected_metrics)
    
    def test_create_test_case(self):
        """Test test case creation"""
        adapter = DeepEvalAdapter()
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        test_case = adapter._create_test_case(input_data)
        
        assert test_case.input == "What is the capital of France?"
        assert test_case.actual_output == "The capital of France is Paris."
        assert test_case.expected_output == "Paris is the capital of France."
        assert test_case.retrieval_context == ["Paris is the capital and largest city of France."]
    
    def test_get_metrics_unsupported(self):
        """Test getting unsupported metrics"""
        adapter = DeepEvalAdapter()
        adapter.available_metrics = {}
        adapter.supported_metrics = []
        
        with pytest.raises(ValueError, match="Unsupported metric"):
            adapter._get_metrics(["unsupported_metric"])
    
    def test_evaluate_success(self):
        """Test successful evaluation"""
        # Mock metric
        mock_metric = Mock()
        mock_metric.measure.return_value = 0.92
        mock_metric.reason = "DeepEval faithfulness evaluation"
        
        adapter = DeepEvalAdapter()
        adapter.available_metrics = {"faithfulness": mock_metric}
        adapter.supported_metrics = ["faithfulness"]
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "deepeval"
        assert result.metadata["evaluation_successful"] is True
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "faithfulness"
        assert result.metrics[0].score == 0.92
        assert result.metrics[0].tool_name == "deepeval"
        assert "faithfulness evaluation" in result.metrics[0].details
    
    def test_evaluate_exception(self):
        """Test evaluation with exception"""
        adapter = DeepEvalAdapter()
        adapter.available_metrics = {"faithfulness": Mock()}
        adapter.supported_metrics = ["faithfulness"]
        
        # Mock metric to raise exception
        mock_metric = Mock()
        mock_metric.measure.side_effect = Exception("Test error")
        adapter.available_metrics["faithfulness"] = mock_metric
        
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        result = adapter.evaluate(input_data)
        
        assert result.metadata["framework"] == "deepeval"
        assert result.metadata["evaluation_successful"] is False
        assert "Test error" in result.metadata["error"]
        assert len(result.metrics) == 0


class TestGenEvalFramework:
    """Test cases for GenEvalFramework"""
    
    def test_initialization(self):
        """Test framework initialization"""
        framework = GenEvalFramework()
        
        assert "ragas" in framework.adapters
        assert "deepeval" in framework.adapters
        assert isinstance(framework.adapters["ragas"], RAGASAdapter)
        assert isinstance(framework.adapters["deepeval"], DeepEvalAdapter)
    
    def test_initialization_with_llm(self):
        """Test framework initialization with LLM"""
        mock_llm = Mock()
        framework = GenEvalFramework(mock_llm)
        
        assert framework.adapters["ragas"].llm_initializer == mock_llm
        assert framework.adapters["deepeval"].llm_initializer == mock_llm
    
    @patch.object(RAGASAdapter, 'evaluate')
    def test_evaluate_single_metric_ragas(self, mock_ragas_evaluate):
        """Test evaluation with single RAGAS metric"""
        # Mock RAGAS evaluation result
        mock_result = Output(
            metrics=[
                MetricResult(
                    name="faithfulness",
                    score=0.85,
                    tool_name="ragas",
                    details="RAGAS faithfulness evaluation"
                )
            ],
            metadata={"framework": "ragas", "evaluation_successful": True}
        )
        mock_ragas_evaluate.return_value = mock_result
        
        framework = GenEvalFramework()
        
        # Mock supported metrics
        framework.adapters["ragas"].supported_metrics = ["faithfulness"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["ragas.faithfulness"]
        )
        
        assert "ragas.faithfulness" in result
        assert result["ragas.faithfulness"][0] == "ragas"
        assert result["ragas.faithfulness"][1] == mock_result
    
    @patch.object(DeepEvalAdapter, 'evaluate')
    def test_evaluate_single_metric_deepeval(self, mock_deepeval_evaluate):
        """Test evaluation with single DeepEval metric"""
        # Mock DeepEval evaluation result
        mock_result = Output(
            metrics=[
                MetricResult(
                    name="faithfulness",
                    score=0.92,
                    tool_name="deepeval",
                    details="DeepEval faithfulness evaluation"
                )
            ],
            metadata={"framework": "deepeval", "evaluation_successful": True}
        )
        mock_deepeval_evaluate.return_value = mock_result
        
        framework = GenEvalFramework()
        
        # Mock supported metrics
        framework.adapters["deepeval"].supported_metrics = ["faithfulness"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["deepeval.faithfulness"]
        )
        
        assert "deepeval.faithfulness" in result
        assert result["deepeval.faithfulness"][0] == "deepeval"
        assert result["deepeval.faithfulness"][1] == mock_result
    
    def test_evaluate_unknown_adapter(self):
        """Test evaluation with unknown adapter"""
        framework = GenEvalFramework()
        
        with pytest.raises(ValueError, match="Unknown adapter"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["unknown.faithfulness"]
            )
    
    def test_evaluate_unsupported_metric(self):
        """Test evaluation with unsupported metric"""
        framework = GenEvalFramework()
        
        # Mock adapters to have no supported metrics
        framework.adapters["ragas"].supported_metrics = []
        framework.adapters["deepeval"].supported_metrics = []
        
        with pytest.raises(ValueError, match="does not support metric"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["ragas.unsupported_metric"]
            )
    
    def test_evaluate_no_adapter_supports_metric(self):
        """Test evaluation when no adapter supports the metric"""
        framework = GenEvalFramework()
        
        # Mock adapters to have no supported metrics
        framework.adapters["ragas"].supported_metrics = []
        framework.adapters["deepeval"].supported_metrics = []
        
        with pytest.raises(ValueError, match="No adapter supports metric"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"]
            )
    
    @patch.object(RAGASAdapter, 'evaluate')
    @patch.object(DeepEvalAdapter, 'evaluate')
    def test_evaluate_multiple_metrics(self, mock_deepeval_evaluate, mock_ragas_evaluate):
        """Test evaluation with multiple metrics from different adapters"""
        # Mock RAGAS result
        mock_ragas_result = Output(
            metrics=[MetricResult(name="faithfulness", score=0.85, tool_name="ragas", details="RAGAS")],
            metadata={"framework": "ragas", "evaluation_successful": True}
        )
        mock_ragas_evaluate.return_value = mock_ragas_result
        
        # Mock DeepEval result
        mock_deepeval_result = Output(
            metrics=[MetricResult(name="answer_relevance", score=0.92, tool_name="deepeval", details="DeepEval")],
            metadata={"framework": "deepeval", "evaluation_successful": True}
        )
        mock_deepeval_evaluate.return_value = mock_deepeval_result
        
        framework = GenEvalFramework()
        
        # Mock supported metrics
        framework.adapters["ragas"].supported_metrics = ["faithfulness"]
        framework.adapters["deepeval"].supported_metrics = ["answer_relevance"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["ragas.faithfulness", "deepeval.answer_relevance"]
        )
        
        assert len(result) == 2
        assert "ragas.faithfulness" in result
        assert "deepeval.answer_relevance" in result
        assert result["ragas.faithfulness"][0] == "ragas"
        assert result["deepeval.answer_relevance"][0] == "deepeval"
    
    @patch.object(RAGASAdapter, 'evaluate')
    @patch.object(DeepEvalAdapter, 'evaluate')
    def test_evaluate_metric_without_adapter_prefix(self, mock_deepeval_evaluate, mock_ragas_evaluate):
        """Test evaluation with metric name without adapter prefix (uses all supporting adapters)"""
        # Mock RAGAS result
        mock_ragas_result = Output(
            metrics=[MetricResult(name="faithfulness", score=0.85, tool_name="ragas", details="RAGAS")],
            metadata={"framework": "ragas", "evaluation_successful": True}
        )
        mock_ragas_evaluate.return_value = mock_ragas_result
        
        # Mock DeepEval result
        mock_deepeval_result = Output(
            metrics=[MetricResult(name="faithfulness", score=0.92, tool_name="deepeval", details="DeepEval")],
            metadata={"framework": "deepeval", "evaluation_successful": True}
        )
        mock_deepeval_evaluate.return_value = mock_deepeval_result
        
        framework = GenEvalFramework()
        
        # Mock both adapters to support the same metric
        framework.adapters["ragas"].supported_metrics = ["faithfulness"]
        framework.adapters["deepeval"].supported_metrics = ["faithfulness"]
        
        result = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]  # No adapter prefix
        )
        
        assert len(result) == 2
        assert "ragas.faithfulness" in result
        assert "deepeval.faithfulness" in result
        assert result["ragas.faithfulness"][0] == "ragas"
        assert result["deepeval.faithfulness"][0] == "deepeval"
        
        # Verify both adapters were called
        mock_ragas_evaluate.assert_called_once()
        mock_deepeval_evaluate.assert_called_once()





# Fixtures for common test data
@pytest.fixture
def sample_input_data():
    """Sample input data for tests"""
    return {
        "question": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "retrieval_context": "Paris is the capital and largest city of France.",
        "reference": "Paris is the capital of France.",
        "metrics": ["faithfulness", "answer_relevance"]
    }


@pytest.fixture
def mock_llm_initializer():
    """Mock LLM initializer for tests"""
    mock_llm = Mock()
    mock_llm.selected_provider = "openai"
    mock_llm.configure_ragas_llm.return_value = {"provider": "openai"}
    mock_llm.configure_deepeval_llm.return_value = {"provider": "openai"}
    mock_llm.get_selected_provider.return_value = "openai"
    return mock_llm


@pytest.fixture
def sample_metric_result():
    """Sample metric result for tests"""
    return MetricResult(
        name="faithfulness",
        score=0.85,
        tool_name="ragas",
        details="RAGAS faithfulness evaluation"
    )
