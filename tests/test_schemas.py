import pytest

from geneval.schemas import Input, MetricResult, Output


class TestSchemas:
    """Test cases for Pydantic schemas"""

    def test_input_schema_valid(self):
        """Test Input schema with valid data"""
        input_data = {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "retrieval_context": "Paris is the capital and largest city of France.",
            "reference": "Paris is the capital of France.",
            "metrics": ["faithfulness", "answer_relevance"],
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

    def test_input_schema_empty_strings(self):
        """Test Input schema with empty strings"""
        input_data = {
            "question": "",
            "response": "",
            "retrieval_context": "",
            "reference": "",
            "metrics": ["faithfulness"],
        }

        input_obj = Input(**input_data)
        assert input_obj.question == ""
        assert input_obj.response == ""
        assert input_obj.retrieval_context == ""
        assert input_obj.reference == ""
        assert input_obj.metrics == ["faithfulness"]

    def test_input_schema_long_text(self):
        """Test Input schema with long text"""
        long_text = "This is a very long text that exceeds the typical length of a normal question or response. " * 10

        input_data = {
            "question": long_text,
            "response": long_text,
            "retrieval_context": long_text,
            "reference": long_text,
            "metrics": ["faithfulness"],
        }

        input_obj = Input(**input_data)
        assert input_obj.question == long_text
        assert input_obj.response == long_text
        assert input_obj.retrieval_context == long_text
        assert input_obj.reference == long_text
        assert input_obj.metrics == ["faithfulness"]

    def test_metric_result_schema(self):
        """Test MetricResult schema"""
        metric_result = MetricResult(
            name="faithfulness",
            score=0.85,
            tool_name="ragas",
            details="RAGAS faithfulness evaluation",
        )

        assert metric_result.name == "faithfulness"
        assert metric_result.score == 0.85
        assert metric_result.tool_name == "ragas"
        assert metric_result.details == "RAGAS faithfulness evaluation"

    def test_metric_result_schema_all_fields(self):
        """Test MetricResult schema with all fields"""
        metric_result = MetricResult(
            name="faithfulness",
            score=0.85,
            tool_name="ragas",
            details="RAGAS faithfulness evaluation",
        )

        assert metric_result.name == "faithfulness"
        assert metric_result.score == 0.85
        assert metric_result.tool_name == "ragas"
        assert metric_result.details == "RAGAS faithfulness evaluation"

    def test_metric_result_schema_score_validation(self):
        """Test MetricResult schema score validation"""
        # Test various score values
        metric_result = MetricResult(
            name="faithfulness",
            score=0.5,
            tool_name="ragas",
            details="RAGAS faithfulness evaluation",
        )

        assert metric_result.score == 0.5

        # Test boundary values
        metric_result = MetricResult(
            name="faithfulness",
            score=0.0,
            tool_name="ragas",
            details="RAGAS faithfulness evaluation",
        )
        assert metric_result.score == 0.0

        metric_result = MetricResult(
            name="faithfulness",
            score=1.0,
            tool_name="ragas",
            details="RAGAS faithfulness evaluation",
        )
        assert metric_result.score == 1.0

    def test_metric_result_schema_extreme_scores(self):
        """Test MetricResult schema with extreme score values"""
        # Test that extreme scores are accepted (no validation constraints)
        metric_result = MetricResult(
            name="faithfulness",
            score=1.5,  # Score > 1
            tool_name="ragas",
            details="RAGAS faithfulness evaluation",
        )
        assert metric_result.score == 1.5

        metric_result = MetricResult(
            name="faithfulness",
            score=-0.1,  # Score < 0
            tool_name="ragas",
            details="RAGAS faithfulness evaluation",
        )
        assert metric_result.score == -0.1

    def test_output_schema(self):
        """Test Output schema"""
        metric_results = [
            MetricResult(
                name="faithfulness",
                score=0.85,
                tool_name="ragas",
                details="RAGAS faithfulness evaluation",
            ),
            MetricResult(
                name="answer_relevance",
                score=0.92,
                tool_name="ragas",
                details="RAGAS answer relevance evaluation",
            ),
        ]

        metadata = {
            "framework": "ragas",
            "total_metrics": 2,
            "evaluation_successful": True,
        }

        output = Output(metrics=metric_results, metadata=metadata)
        assert len(output.metrics) == 2
        assert output.metadata["framework"] == "ragas"
        assert output.metadata["total_metrics"] == 2
        assert output.metadata["evaluation_successful"] is True

    def test_output_schema_empty_metrics(self):
        """Test Output schema with empty metrics"""
        metadata = {
            "framework": "ragas",
            "total_metrics": 0,
            "evaluation_successful": False,
        }

        output = Output(metrics=[], metadata=metadata)
        assert len(output.metrics) == 0
        assert output.metadata["framework"] == "ragas"
        assert output.metadata["total_metrics"] == 0
        assert output.metadata["evaluation_successful"] is False

    def test_output_schema_minimal_metadata(self):
        """Test Output schema with minimal metadata"""
        metric_results = [
            MetricResult(
                name="faithfulness",
                score=0.85,
                tool_name="ragas",
                details="RAGAS faithfulness evaluation",
            )
        ]

        output = Output(metrics=metric_results, metadata={})
        assert len(output.metrics) == 1
        assert output.metadata == {}

    def test_output_schema_complex_metadata(self):
        """Test Output schema with complex metadata"""
        metric_results = [
            MetricResult(
                name="faithfulness",
                score=0.85,
                tool_name="ragas",
                details="RAGAS faithfulness evaluation",
            )
        ]

        metadata = {
            "framework": "ragas",
            "total_metrics": 1,
            "evaluation_successful": True,
            "timestamp": "2024-01-01T00:00:00Z",
            "model_info": {"provider": "openai", "model": "gpt-4o-mini"},
            "processing_time_ms": 1250,
        }

        output = Output(metrics=metric_results, metadata=metadata)
        assert len(output.metrics) == 1
        assert output.metadata["framework"] == "ragas"
        assert output.metadata["model_info"]["provider"] == "openai"
        assert output.metadata["processing_time_ms"] == 1250

    def test_schema_mutability(self):
        """Test that schemas are mutable after creation"""
        input_data = {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "retrieval_context": "Paris is the capital and largest city of France.",
            "reference": "Paris is the capital of France.",
            "metrics": ["faithfulness"],
        }

        input_obj = Input(**input_data)

        # Test that we can modify the object after creation
        input_obj.question = "Modified question"
        assert input_obj.question == "Modified question"

    def test_schema_serialization(self):
        """Test schema serialization to dict"""
        input_data = {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "retrieval_context": "Paris is the capital and largest city of France.",
            "reference": "Paris is the capital of France.",
            "metrics": ["faithfulness"],
        }

        input_obj = Input(**input_data)
        input_dict = input_obj.model_dump()

        assert input_dict["question"] == "What is the capital of France?"
        assert input_dict["response"] == "The capital of France is Paris."
        assert input_dict["retrieval_context"] == "Paris is the capital and largest city of France."
        assert input_dict["reference"] == "Paris is the capital of France."
        assert input_dict["metrics"] == ["faithfulness"]

    def test_schema_json_serialization(self):
        """Test schema JSON serialization"""
        input_data = {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "retrieval_context": "Paris is the capital and largest city of France.",
            "reference": "Paris is the capital of France.",
            "metrics": ["faithfulness"],
        }

        input_obj = Input(**input_data)
        input_json = input_obj.model_dump_json()

        # Should be valid JSON
        import json

        parsed = json.loads(input_json)
        assert parsed["question"] == "What is the capital of France?"
        assert parsed["response"] == "The capital of France is Paris."
        assert parsed["retrieval_context"] == "Paris is the capital and largest city of France."
        assert parsed["reference"] == "Paris is the capital of France."
        assert parsed["metrics"] == ["faithfulness"]

    def test_schema_validation_error_messages(self):
        """Test schema validation error messages"""
        # Test missing required field
        with pytest.raises(ValueError) as exc_info:
            Input(
                question="What is the capital of France?",
                # Missing response, retrieval_context, reference, metrics
            )

        error_message = str(exc_info.value)
        assert "response" in error_message or "retrieval_context" in error_message or "reference" in error_message or "metrics" in error_message

    def test_schema_field_types(self):
        """Test schema field type validation"""
        # Test invalid types
        with pytest.raises(ValueError):
            Input(
                question=123,  # Should be string
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness"],
            )

        with pytest.raises(ValueError):
            Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics="faithfulness",  # Should be list
            )

    def test_schema_metrics_validation(self):
        """Test schema metrics field validation"""
        # Test empty metrics list (should be allowed)
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=[],  # Empty list should be allowed
        )
        assert input_data.metrics == []

        # Test metrics with string items (enforced by schema)
        input_data = Input(
            question="What is the capital of France?",
            response="The capital of France is Paris.",
            retrieval_context="Paris is the capital and largest city of France.",
            reference="Paris is the capital of France.",
            metrics=["faithfulness", "answer_relevance"],  # String items enforced
        )
        assert input_data.metrics == ["faithfulness", "answer_relevance"]

        # Test that non-string items are rejected
        with pytest.raises(ValueError):
            Input(
                question="What is the capital of France?",
                response="The capital of France is Paris.",
                retrieval_context="Paris is the capital and largest city of France.",
                reference="Paris is the capital of France.",
                metrics=["faithfulness", 123],  # Non-string item should be rejected
            )
