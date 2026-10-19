"""
Shard Agents — Multi-agent system (Fraudster, Planner, Researcher, Investigator, Critic, Rule-Writer).
"""

from shard.agents.critic import CriticAgent
from shard.agents.fraudster import FraudsterAgent, FraudsterRoundOutput
from shard.agents.investigator import (
    InvestigationPipeline,
    InvestigatorAgent,
    PipelineResult,
)
from shard.agents.planner import PlannerAgent
from shard.agents.researcher import ResearchAgent
from shard.agents.rule_writer import RuleWriterAgent, RuleWriterOutput

__all__ = [
    "CriticAgent",
    "FraudsterAgent",
    "FraudsterRoundOutput",
    "InvestigationPipeline",
    "InvestigatorAgent",
    "PipelineResult",
    "PlannerAgent",
    "ResearchAgent",
    "RuleWriterAgent",
    "RuleWriterOutput",
]


