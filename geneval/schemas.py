from typing import Any

from pydantic import BaseModel, Field


class Input(BaseModel):
    """
    Input for the GenEval framework.
    """

    question: str = Field(..., description="The user query or question to be answered")
    response: str = Field(..., description="The response from the model")
    retrieval_context: str = Field(..., description="Contents of documents or snippets used for RAG retrieval")
    reference: str = Field(..., description="The reference answer or ground truth")
    metrics: list[str] = Field(..., description="List of metrics to evaluate the response")


class MetricResult(BaseModel):
    """
    Result of a single metric evaluation.
    """

    name: str = Field(..., description="Name of the metric")
    score: float = Field(..., description="Score of the metric")
    tool_name: str = Field(..., description="Name of the tool used to evaluate the metric")
    details: str = Field(..., description="Details of the metric evaluation")


class Output(BaseModel):
    """
    Output of the GenEval framework.
    """

    metrics: list[MetricResult] = Field(..., description="List of metric results")
    metadata: dict[str, Any] = Field(..., description="Metadata about the evaluation")


class MetricEvaluation(BaseModel):
    name: str = Field(..., description="Abstract metric name (e.g. 'faithfulness')")
    score: float = Field(..., description="Raw score from adapter (0.0 - 1.0)")
    threshold: float = Field(..., description="Criteria threshold from profile")
    passed: bool = Field(..., description="Whether score >= threshold")
    weight: float = Field(..., description="Weight from profile")
    weighted_score: float = Field(..., description="weight * score")
    adapter: str = Field(..., description="Which adapter produced this score (e.g. 'ragas')")
    details: str | None = Field(default=None, description="Explanation from the adapter")


class ProfileResult(BaseModel):
    profile_name: str = Field(..., description="Name of the evaluation profile used")
    policy_name: str | None = Field(default=None, description="Name of the policy, if used")
    overall_passed: bool = Field(..., description="True if composite AND all individual metrics passed")
    composite_score: float = Field(..., description="Weighted composite score")
    composite_threshold: float = Field(..., description="Threshold for composite score")
    composite_passed: bool = Field(..., description="Whether composite_score >= composite_threshold")
    metric_results: list[MetricEvaluation] = Field(..., description="Per-metric evaluation details")
    metadata: dict[str, Any] = Field(..., description="Timestamps, LLM info, framework version")


class BatchResult(BaseModel):
    profile_name: str = Field(..., description="Name of the evaluation profile used")
    policy_name: str | None = Field(default=None, description="Name of the policy, if used")
    overall_passed: bool = Field(..., description="True only if ALL cases passed")
    case_results: list[ProfileResult] = Field(..., description="Per-case evaluation results")
    summary: dict[str, Any] = Field(..., description="Per-metric averages across cases")
    pass_rate: float = Field(..., description="Fraction of cases that passed (0.0 - 1.0)")
    metadata: dict[str, Any] = Field(..., description="Timestamps, LLM info, framework version")
