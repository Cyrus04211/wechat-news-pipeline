"""Agent provider abstraction for LLM-enhanced news collection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentDecision:
    """Structured decision returned by an agent."""
    decision: str  # Primary decision/label
    confidence: float  # 0.0 - 1.0
    reasoning: str = ""  # Short explanation
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentProvider(ABC):
    """Abstract LLM provider for pipeline agents."""

    @abstractmethod
    def analyze(self, prompt: str, system: str = "", context: Optional[dict] = None) -> AgentDecision:
        """Send a prompt to the LLM and return a structured decision."""
        ...

    @abstractmethod
    def analyze_batch(
        self,
        prompts: list[str],
        system: str = "",
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        """Analyze multiple prompts (may batch for efficiency)."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    def is_available(self) -> bool:
        return True
