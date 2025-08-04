"""
Integration tests for GenEval framework.

These tests verify end-to-end functionality with real external dependencies.
They require API keys and may be slower than unit tests.
"""

import pytest
import os
import yaml
from pathlib import Path

from geneval import GenEvalFramework, LLMManager
from geneval.schemas import Input


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def test_data():
    """Load test data from YAML file"""
    test_data_path = Path(__file__).parent / "test_data_clean.yaml"
    
    if not test_data_path.exists():
        pytest.skip("Test data file not found")
        
    with open(test_data_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def llm_manager():
    """Initialize LLM manager with config file"""
    try:
        llm_manager = LLMManager()
        if llm_manager.select_provider():
            return llm_manager
        else:
            pytest.skip("No default LLM provider configured in config file")
    except Exception as e:
        pytest.skip(f"LLM manager initialization failed: {e}")


class TestEndToEndEvaluation:
    """Test complete evaluation workflows"""
    
    def test_single_metric_evaluation(self, llm_manager, test_data):
        """Test evaluation with a single metric"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        test_case = test_data['test_cases'][0]
        
        results = framework.evaluate(
            question=test_case['user_input'],
            response=test_case['response'],
            reference=test_case['reference'],
            retrieval_context=test_case['retrieved_contexts'],
            metrics=['faithfulness']
        )
        
        # Verify we got results
        assert len(results) > 0
        
        # Check that we have faithfulness results
        faithfulness_results = [k for k in results.keys() if 'faithfulness' in k]
        assert len(faithfulness_results) > 0
        
        # Verify result structure
        for metric_key, (adapter_name, output) in results.items():
            assert adapter_name in ['ragas', 'deepeval']
            assert hasattr(output, 'metrics')
            assert hasattr(output, 'metadata')
            assert output.metadata['framework'] == adapter_name
    
    def test_multiple_metrics_evaluation(self, llm_manager, test_data):
        """Test evaluation with multiple metrics"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        test_case = test_data['test_cases'][0]
        
        results = framework.evaluate(
            question=test_case['user_input'],
            response=test_case['response'],
            reference=test_case['reference'],
            retrieval_context=test_case['retrieved_contexts'],
            metrics=['faithfulness', 'answer_relevance', 'context_relevance']
        )
        
        # Should have results for multiple metrics
        assert len(results) >= 2
        
        # Check that we have results from different adapters
        adapters_used = set()
        for metric_key, (adapter_name, output) in results.items():
            adapters_used.add(adapter_name)
        
        assert len(adapters_used) >= 1
    
    def test_ragas_specific_evaluation(self, llm_manager, test_data):
        """Test RAGAS-specific metrics"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        test_case = test_data['test_cases'][0]
        
        results = framework.evaluate(
            question=test_case['user_input'],
            response=test_case['response'],
            reference=test_case['reference'],
            retrieval_context=test_case['retrieved_contexts'],
            metrics=['ragas.faithfulness', 'ragas.response_relevancy']
        )
        
        # Should have RAGAS results
        ragas_results = [k for k in results.keys() if k.startswith('ragas.')]
        assert len(ragas_results) > 0
        
        for metric_key, (adapter_name, output) in results.items():
            if metric_key.startswith('ragas.'):
                assert adapter_name == 'ragas'
                assert output.metadata['framework'] == 'ragas'
    
    def test_deepeval_specific_evaluation(self, llm_manager, test_data):
        """Test DeepEval-specific metrics"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        test_case = test_data['test_cases'][0]
        
        results = framework.evaluate(
            question=test_case['user_input'],
            response=test_case['response'],
            reference=test_case['reference'],
            retrieval_context=test_case['retrieved_contexts'],
            metrics=['deepeval.faithfulness', 'deepeval.answer_relevance']
        )
        
        # Should have DeepEval results
        deepeval_results = [k for k in results.keys() if k.startswith('deepeval.')]
        assert len(deepeval_results) > 0
        
        for metric_key, (adapter_name, output) in results.items():
            if metric_key.startswith('deepeval.'):
                assert adapter_name == 'deepeval'
                assert output.metadata['framework'] == 'deepeval'


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""
    
    def test_multiple_test_cases(self, llm_manager, test_data):
        """Test evaluation across multiple test cases"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        # Test first 3 cases
        for i in range(min(3, len(test_data['test_cases']))):
            test_case = test_data['test_cases'][i]
            
            results = framework.evaluate(
                question=test_case['user_input'],
                response=test_case['response'],
                reference=test_case['reference'],
                retrieval_context=test_case['retrieved_contexts'],
                metrics=['faithfulness']
            )
            
            # Each case should produce results
            assert len(results) > 0
            
            # Check that scores are reasonable (between 0 and 1)
            for metric_key, (adapter_name, output) in results.items():
                for metric_result in output.metrics:
                    assert 0 <= metric_result.score <= 1
    
    def test_error_handling_real_apis(self, llm_manager):
        """Test error handling with real APIs"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        # Test with invalid data that should cause errors
        try:
            results = framework.evaluate(
                question="",  # Empty question
                response="",  # Empty response
                reference="",  # Empty reference
                retrieval_context="",  # Empty context
                metrics=['faithfulness']
            )
            
            # Should handle gracefully
            assert isinstance(results, dict)
            
        except Exception as e:
            # Should not crash, but may raise reasonable exceptions
            assert "validation" in str(e).lower() or "empty" in str(e).lower()


class TestPerformanceAndReliability:
    """Test performance and reliability aspects"""
    
    def test_evaluation_consistency(self, llm_manager, test_data):
        """Test that evaluations are consistent"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        test_case = test_data['test_cases'][0]
        
        # Run same evaluation twice
        results1 = framework.evaluate(
            question=test_case['user_input'],
            response=test_case['response'],
            reference=test_case['reference'],
            retrieval_context=test_case['retrieved_contexts'],
            metrics=['faithfulness']
        )
        
        results2 = framework.evaluate(
            question=test_case['user_input'],
            response=test_case['response'],
            reference=test_case['reference'],
            retrieval_context=test_case['retrieved_contexts'],
            metrics=['faithfulness']
        )
        
        # Results should be consistent (same structure)
        assert len(results1) == len(results2)
        
        # Same metrics should be present
        assert set(results1.keys()) == set(results2.keys())
    
    def test_large_context_handling(self, llm_manager):
        """Test handling of large context"""
        framework = GenEvalFramework(llm_manager=llm_manager)
        
        # Create large context
        large_context = "This is a test context. " * 1000  # ~25KB
        
        try:
            results = framework.evaluate(
                question="What is the main topic?",
                response="The main topic is testing.",
                reference="Testing",
                retrieval_context=large_context,
                metrics=['faithfulness']
            )
            
            # Should handle large context without crashing
            assert isinstance(results, dict)
            
        except Exception as e:
            # Should handle gracefully or provide meaningful error
            assert "context" in str(e).lower() or "size" in str(e).lower() or "limit" in str(e).lower() 