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
