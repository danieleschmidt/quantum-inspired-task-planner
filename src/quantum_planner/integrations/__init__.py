"""Framework integrations for popular agent orchestration platforms."""

from .base_integration import BaseIntegration, IntegrationError
from .crewai_integration import CrewAIScheduler
from .autogen_integration import AutoGenScheduler

try:
    from .langchain_integration import LangChainScheduler
    _has_langchain = True
except ImportError:
    _has_langchain = False
    LangChainScheduler = None

__all__ = [
    "BaseIntegration",
    "IntegrationError",
    "CrewAIScheduler",
    "AutoGenScheduler"
]

if _has_langchain:
    __all__.append("LangChainScheduler")