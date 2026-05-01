from unittest.mock import Mock, patch

import pytest

from geneval.adapters.deepeval_adapter import DeepEvalAdapter
from geneval.adapters.ragas_adapter import RAGASAdapter
from geneval.framework import GenEvalFramework
from geneval.llm_manager import LLMManager
from geneval.schemas import BatchResult, ProfileResult


class TestEvaluateProfile:
    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_delegates_to_profile_manager(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
        tmp_path,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        mock_ragas_adapter.supported_metrics = ["faithfulness", "answer_relevancy"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]

        framework = GenEvalFramework("test_config.yaml")

        with patch("geneval.framework.ProfileManager") as mock_pm_class:
            mock_pm = Mock()
            mock_pm.evaluate.return_value = Mock(spec=ProfileResult)
            mock_pm_class.return_value = mock_pm

            result = framework.evaluate_profile(
                profile="rag_default",
                profiles_path=str(tmp_path / "profiles.yaml"),
                question="What is X?",
                response="X is Y.",
                reference="X is Y.",
                retrieval_context="Context.",
            )

            assert result is not None
            mock_pm.evaluate.assert_called_once()
            call_kwargs = mock_pm.evaluate.call_args[1]
            assert call_kwargs["profile_name"] == "rag_default"
            assert call_kwargs["question"] == "What is X?"

    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_with_policy(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
        tmp_path,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]

        framework = GenEvalFramework("test_config.yaml")

        with patch("geneval.framework.ProfileManager") as mock_pm_class:
            mock_pm = Mock()
            mock_pm.evaluate.return_value = Mock(spec=ProfileResult)
            mock_pm_class.return_value = mock_pm

            framework.evaluate_profile(
                policy="ci_gate",
                profiles_path=str(tmp_path / "profiles.yaml"),
                question="What is X?",
                response="X is Y.",
                reference="X is Y.",
                retrieval_context="Context.",
            )

            call_kwargs = mock_pm.evaluate.call_args[1]
            assert call_kwargs["policy_name"] == "ci_gate"
            assert call_kwargs["profile_name"] is None

    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_batch(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
        tmp_path,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter = Mock(spec=DeepEvalAdapter)
        mock_ragas_adapter_class.return_value = mock_ragas_adapter
        mock_deepeval_adapter_class.return_value = mock_deepeval_adapter
        mock_ragas_adapter.supported_metrics = ["faithfulness"]
        mock_deepeval_adapter.supported_metrics = ["faithfulness"]

        framework = GenEvalFramework("test_config.yaml")

        with patch("geneval.framework.ProfileManager") as mock_pm_class:
            mock_pm = Mock()
            mock_pm.evaluate_batch.return_value = Mock(spec=BatchResult)
            mock_pm_class.return_value = mock_pm

            result = framework.evaluate_profile_batch(
                profile="rag_default",
                data_path=str(tmp_path / "data.yaml"),
                profiles_path=str(tmp_path / "profiles.yaml"),
            )

            assert result is not None
            mock_pm.evaluate_batch.assert_called_once()

    @patch("geneval.framework.RAGASAdapter")
    @patch("geneval.framework.DeepEvalAdapter")
    @patch("geneval.framework.LLMManager")
    def test_evaluate_profile_no_profile_or_policy_raises(
        self,
        mock_llm_manager_class,
        mock_deepeval_adapter_class,
        mock_ragas_adapter_class,
    ):
        mock_llm_manager = Mock(spec=LLMManager)
        mock_llm_manager.get_default_provider.return_value = "openai"
        mock_llm_manager.get_provider_config.return_value = {"model": "gpt-4o-mini"}
        mock_llm_manager_class.return_value = mock_llm_manager

        mock_ragas_adapter_class.return_value = Mock(spec=RAGASAdapter)
        mock_deepeval_adapter_class.return_value = Mock(spec=DeepEvalAdapter)

        framework = GenEvalFramework("test_config.yaml")

        with pytest.raises(ValueError, match="Either profile or policy"):
            framework.evaluate_profile(
                question="What is X?",
                response="X is Y.",
                reference="X is Y.",
                retrieval_context="Context.",
            )
