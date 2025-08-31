import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from geneval.framework import GenEvalFramework
from geneval.llm_manager import LLMManager
from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.adapters.deepeval_adapter import DeepEvalAdapter
from geneval.schemas import Input, Output


class TestGenEvalFramework:
    """Test cases for GenEvalFramework"""
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_initialization_success(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test successful GenEvalFramework initialization"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness", "answer_relevancy"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness", "context_relevance"]
        
        framework = GenEvalFramework("test_config.yaml")
        
        assert framework.llm_manager == mock_llm_manager
        assert framework.llm_info["provider"] == "openai"
        assert framework.llm_info["model"] == "gpt-4o-mini"
        assert "ragas" in framework.adapters
        assert "deepeval" in framework.adapters
        assert framework.adapters["ragas"] == mock_ragas_adapter
        assert framework.adapters["deepeval"] == mock_deepeval_adapter
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_initialization_adapter_failure(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test GenEvalFramework initialization when adapter creation fails"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock RAGAS adapter to fail
        mock_ragas_adapter_class.side_effect = Exception("RAGAS adapter failed")
        
        with pytest.raises(RuntimeError, match="Failed to initialize adapters"):
            GenEvalFramework("test_config.yaml")
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_initialization_deepeval_adapter_failure(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test GenEvalFramework initialization when DeepEval adapter creation fails"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock RAGAS adapter to succeed
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        
        # Mock DeepEval adapter to fail
        mock_deepeval_adapter_class.side_effect = Exception("DeepEval adapter failed")
        
        with pytest.raises(RuntimeError, match="Failed to initialize adapters"):
            GenEvalFramework("test_config.yaml")
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_single_metric_single_adapter(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with a single metric that only one adapter supports"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics - only RAGAS supports faithfulness
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["context_relevance"]
        
        # Mock evaluation results
        mock_output = Mock(spec=Output)
        mock_ragas_adapter.evaluate.return_value = mock_output
        
        framework = GenEvalFramework("test_config.yaml")
        
        results = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        assert len(results) == 1
        assert "ragas.faithfulness" in results
        adapter_name, output = results["ragas.faithfulness"]
        assert adapter_name == "ragas"
        assert output == mock_output
        
        # Verify adapter was called correctly
        mock_ragas_adapter.evaluate.assert_called_once()
        call_args = mock_ragas_adapter.evaluate.call_args[0][0]
        assert call_args.question == "What is the capital of France?"
        assert call_args.response == "The capital of France is Paris."
        assert call_args.metrics == ["faithfulness"]
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_single_metric_multiple_adapters(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with a single metric that multiple adapters support"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics - both support faithfulness
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]
        
        # Mock evaluation results
        mock_ragas_output = Mock(spec=Output)
        mock_deepeval_output = Mock(spec=Output)
        mock_ragas_adapter.evaluate.return_value = mock_ragas_output
        mock_deepeval_adapter.evaluate.return_value = mock_deepeval_output
        
        framework = GenEvalFramework("test_config.yaml")
        
        results = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        assert len(results) == 2
        assert "ragas.faithfulness" in results
        assert "deepeval.faithfulness" in results
        
        # Verify RAGAS result
        ragas_adapter_name, ragas_output = results["ragas.faithfulness"]
        assert ragas_adapter_name == "ragas"
        assert ragas_output == mock_ragas_output
        
        # Verify DeepEval result
        deepeval_adapter_name, deepeval_output = results["deepeval.faithfulness"]
        assert deepeval_adapter_name == "deepeval"
        assert deepeval_output == mock_deepeval_output
        
        # Verify both adapters were called
        assert mock_ragas_adapter.evaluate.call_count == 1
        assert mock_deepeval_adapter.evaluate.call_count == 1
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_specific_adapter_metric(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with specific adapter.metric format"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness", "answer_relevancy"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness", "context_relevance"]
        
        # Mock evaluation results
        mock_output = Mock(spec=Output)
        mock_ragas_adapter.evaluate.return_value = mock_output
        
        framework = GenEvalFramework("test_config.yaml")
        
        results = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["ragas.faithfulness"]
        )
        
        assert len(results) == 1
        assert "ragas.faithfulness" in results
        adapter_name, output = results["ragas.faithfulness"]
        assert adapter_name == "ragas"
        assert output == mock_output
        
        # Verify only RAGAS adapter was called
        mock_ragas_adapter.evaluate.assert_called_once()
        mock_deepeval_adapter.evaluate.assert_not_called()
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_multiple_metrics_mixed_format(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with multiple metrics in mixed format"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness", "answer_relevancy"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness", "context_relevance"]
        
        # Mock evaluation results
        mock_ragas_faithfulness = Mock(spec=Output)
        mock_ragas_answer_relevancy = Mock(spec=Output)
        mock_deepeval_context_relevance = Mock(spec=Output)
        
        mock_ragas_adapter.evaluate.side_effect = [mock_ragas_faithfulness, mock_ragas_answer_relevancy]
        mock_deepeval_adapter.evaluate.return_value = mock_deepeval_context_relevance
        
        framework = GenEvalFramework("test_config.yaml")
        
        results = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["ragas.faithfulness", "ragas.answer_relevancy", "deepeval.context_relevance"]
        )
        
        assert len(results) == 3
        assert "ragas.faithfulness" in results
        assert "ragas.answer_relevancy" in results
        assert "deepeval.context_relevance" in results
        
        # Verify results
        assert results["ragas.faithfulness"][0] == "ragas"
        assert results["ragas.faithfulness"][1] == mock_ragas_faithfulness
        assert results["ragas.answer_relevancy"][0] == "ragas"
        assert results["ragas.answer_relevancy"][1] == mock_ragas_answer_relevancy
        assert results["deepeval.context_relevance"][0] == "deepeval"
        assert results["deepeval.context_relevance"][1] == mock_deepeval_context_relevance
        
        # Verify call counts
        assert mock_ragas_adapter.evaluate.call_count == 2
        assert mock_deepeval_adapter.evaluate.call_count == 1
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_unknown_adapter(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with unknown adapter"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]
        
        framework = GenEvalFramework("test_config.yaml")
        
        with pytest.raises(ValueError, match="Unknown adapter: unknown"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["unknown.faithfulness"]
            )
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_unsupported_metric_for_adapter(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with metric that adapter doesn't support"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics - RAGAS doesn't support context_relevance
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness", "context_relevance"]
        
        framework = GenEvalFramework("test_config.yaml")
        
        with pytest.raises(ValueError, match="Adapter 'ragas' does not support metric 'context_relevance'"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["ragas.context_relevance"]
            )
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_no_adapter_supports_metric(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with metric that no adapter supports"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics - neither supports unknown_metric
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]
        
        framework = GenEvalFramework("test_config.yaml")
        
        with pytest.raises(ValueError, match="No adapter supports metric 'unknown_metric'"):
            framework.evaluate(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["unknown_metric"]
            )
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_empty_metrics_list(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with empty metrics list"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]
        
        framework = GenEvalFramework("test_config.yaml")
        
        results = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=[]
        )
        
        assert len(results) == 0
        assert results == {}
        
        # Verify no adapters were called
        mock_ragas_adapter.evaluate.assert_not_called()
        mock_deepeval_adapter.evaluate.assert_not_called()
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_with_complex_input_data(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with complex input data"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness", "answer_relevancy"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness", "context_relevance"]
        
        # Mock evaluation results
        mock_ragas_faithfulness = Mock(spec=Output)
        mock_ragas_answer_relevancy = Mock(spec=Output)
        mock_deepeval_context_relevance = Mock(spec=Output)
        
        mock_ragas_adapter.evaluate.side_effect = [mock_ragas_faithfulness, mock_ragas_answer_relevancy]
        mock_deepeval_adapter.evaluate.return_value = mock_deepeval_context_relevance
        
        framework = GenEvalFramework("test_config.yaml")
        
        # Test with long text and special characters
        long_question = "What is the capital of France and what are its main attractions? " + "This is a very long question. " * 10
        long_response = "The capital of France is Paris. " + "Paris is a beautiful city with many attractions. " * 15
        long_context = "Paris is the capital and largest city of France. " + "It is known for its art, fashion, gastronomy and culture. " * 20
        long_reference = "Paris is the capital of France. " + "It is famous for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. " * 12
        
        results = framework.evaluate(
            question=long_question,
            response=long_response,
            retrieval_context=long_context,
            reference=long_reference,
            metrics=["ragas.faithfulness", "ragas.answer_relevancy", "deepeval.context_relevance"]
        )
        
        assert len(results) == 3
        
        # Verify the input data was passed correctly to adapters
        ragas_calls = mock_ragas_adapter.evaluate.call_args_list
        assert len(ragas_calls) == 2
        
        # Check first call (faithfulness)
        first_call_input = ragas_calls[0][0][0]
        assert first_call_input.question == long_question
        assert first_call_input.response == long_response
        assert first_call_input.retrieval_context == long_context
        assert first_call_input.reference == long_reference
        assert first_call_input.metrics == ["faithfulness"]
        
        # Check second call (answer_relevancy)
        second_call_input = ragas_calls[1][0][0]
        assert second_call_input.question == long_question
        assert second_call_input.response == long_response
        assert second_call_input.retrieval_context == long_context
        assert second_call_input.reference == long_reference
        assert second_call_input.metrics == ["answer_relevancy"]
    

    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_with_empty_strings(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with empty strings in input data"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]
        
        # Mock evaluation results
        mock_output = Mock(spec=Output)
        mock_ragas_adapter.evaluate.return_value = mock_output
        mock_deepeval_adapter.evaluate.return_value = mock_output
        
        framework = GenEvalFramework("test_config.yaml")
        
        # Test with empty strings
        results = framework.evaluate(
            question="",
            response="",
            retrieval_context="",
            reference="",
            metrics=["faithfulness"]
        )
        
        assert len(results) == 2
        assert "ragas.faithfulness" in results
        assert "deepeval.faithfulness" in results
        
        # Verify empty strings were passed correctly
        ragas_call = mock_ragas_adapter.evaluate.call_args[0][0]
        assert ragas_call.question == ""
        assert ragas_call.response == ""
        assert ragas_call.retrieval_context == ""
        assert ragas_call.reference == ""
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_with_vllm_provider(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with vLLM provider"""
        # Mock LLM manager with vLLM provider
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "vllm"
        mock_llm_manager.get_provider_config.return_value = {"model": "gemini-2.0-flash"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness", "answer_relevancy"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness", "context_relevance"]
        
        # Mock evaluation results
        mock_ragas_output = Mock(spec=Output)
        mock_deepeval_output = Mock(spec=Output)
        mock_ragas_adapter.evaluate.return_value = mock_ragas_output
        mock_deepeval_adapter.evaluate.return_value = mock_deepeval_output
        
        framework = GenEvalFramework("test_config.yaml")
        
        # Verify framework initialization worked with vLLM
        assert framework.llm_info["provider"] == "vllm"
        assert framework.llm_info["model"] == "gemini-2.0-flash"
        
        results = framework.evaluate(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness"]
        )
        
        # Should work with both adapters
        assert len(results) == 2
        assert "ragas.faithfulness" in results
        assert "deepeval.faithfulness" in results
        
        # Verify both adapters were called
        mock_ragas_adapter.evaluate.assert_called_once()
        mock_deepeval_adapter.evaluate.assert_called_once()
    
    @patch('geneval.framework.RAGASAdapter')
    @patch('geneval.framework.DeepEvalAdapter')
    @patch('geneval.framework.LLMManager')
    def test_evaluate_with_none_values(self, mock_llm_manager_class, mock_deepeval_adapter_class, mock_ragas_adapter_class):
        """Test evaluation with None values in input data"""
        # Mock LLM manager
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Mock adapters
        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        
        # Mock supported metrics
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]
        
        # Mock evaluation results
        mock_output = Mock(spec=Output)
        mock_ragas_adapter.evaluate.return_value = mock_output
        mock_deepeval_adapter.evaluate.return_value = mock_output
        
        framework = GenEvalFramework("test_config.yaml")
        
        # Test with None values - but the Input schema requires strings, so this should fail
        with pytest.raises(Exception):  # Should fail due to validation
            framework.evaluate(
                question=None,
                response=None,
                retrieval_context=None,
                reference=None,
                metrics=["faithfulness"]
            )
