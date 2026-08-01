"""Models for function definitions using Pydantic."""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    """Specification for an individual function parameter."""

    type: str
    description: Optional[str] = None


class ReturnSpec(BaseModel):
    """Specification for function return type."""

    type: str


class FunctionDefinition(BaseModel):
    """Definition of a function tool available to the LLM."""

    name: str
    description: str
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ReturnSpec] = None


class TestCase(BaseModel):
    """Input prompt model."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Validated schema for generated output JSON."""

    prompt: str
    name: str
    parameters: Dict[str, Any]