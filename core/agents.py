from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config import ModelTier, ToolType


@dataclass(frozen=True)
class AgentSpec:
    """Declarative definition for an agent in the workforce."""

    name: str
    label: str
    description: str
    tier: ModelTier
    tools: List[ToolType]
    system_preamble: str


_AGENT_SPECS: Dict[str, AgentSpec] = {
    "workforce": AgentSpec(
        name="workforce",
        label="Workforce",
        description="General-purpose agent that completes multi-step workplace tasks.",
        tier=ModelTier.POWER,
        tools=[ToolType.FILE_READER, ToolType.DATABASE, ToolType.SPREADSHEET, ToolType.CALCULATOR, ToolType.WEB_SEARCH],
        system_preamble=(
            "You are an agentic workforce that completes tasks for the user.\n"
            "You must be proactive, structured, and outcome-focused.\n\n"
            "Rules:\n"
            "- If the request is ambiguous, ask at most 3 clarifying questions, otherwise proceed.\n"
            "- If tools are available, use them when they materially improve correctness.\n"
            "- Provide a clear final deliverable (checklist, draft, table, or steps) and keep it concise.\n"
            "- Do not reveal internal tool errors; assume errors are logged server-side.\n"
        ),
    ),
    "research": AgentSpec(
        name="research",
        label="Research",
        description="Web research agent for current info with citations.",
        tier=ModelTier.POWER,
        tools=[ToolType.WEB_SEARCH],
        system_preamble=(
            "You are a research agent.\n"
            "- Prefer web search for current facts and include Markdown links for sources.\n"
            "- If sources are not available, say so explicitly.\n"
        ),
    ),
    "analyst": AgentSpec(
        name="analyst",
        label="Analyst",
        description="Document/spreadsheet analyst agent grounded in provided files.",
        tier=ModelTier.POWER,
        tools=[ToolType.FILE_READER, ToolType.DATABASE, ToolType.SPREADSHEET, ToolType.CALCULATOR],
        system_preamble=(
            "You are an analyst agent focused on uploaded documents and structured data.\n"
            "- Base answers on document content and computed results.\n"
            "- If the data is missing, state what is missing and what you can do next.\n"
        ),
    ),
}


def list_agents() -> List[Dict[str, str]]:
    return [
        {
            "name": a.name,
            "label": a.label,
            "description": a.description,
        }
        for a in _AGENT_SPECS.values()
    ]


def get_agent(name: str) -> AgentSpec:
    key = (name or "").strip().lower()
    if not key:
        key = "workforce"
    if key not in _AGENT_SPECS:
        raise KeyError(f"Unknown agent: {name}")
    return _AGENT_SPECS[key]
