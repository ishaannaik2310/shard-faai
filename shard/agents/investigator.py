"""
Shard — Investigator Agent and Pipeline.

Executes verification checks mandated by an InvestigationPlan using evidence
gathered in an EvidenceBundle, scoring findings via the centralised Rulebook.

CRITICAL INVARIANTS:
1. Ground-Truth Isolation (TI-01): The Investigator MUST NEVER import GroundTruthVault,
   request hidden labels, or use evaluation-only fraud metadata.
2. Centralised Scoring: Bounded risk scores and triage dispositions are determined strictly
   by the Rulebook implementation; no independent thresholds are duplicated here.
3. Genuine Collaboration: The Investigator strictly requires and consumes the actual outputs
   of the Planner (InvestigationPlan) and Researcher (EvidenceBundle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shard.core.models import (
    Claim,
    ClaimDecision,
    Disposition,
    EvidenceBundle,
    Finding,
    InvestigationPlan,
)
from shard.rules.rulebook import Rulebook, load_default_rulebook
from shard.rules.templates import TEMPLATE_REGISTRY, ToolCheckResult


class InvestigatorAgent:
    """
    Deterministic investigation agent that validates plans, executes verification tools
    against retrieved evidence, and compiles an auditable ClaimDecision.
    """

    def investigate(
        self,
        claim: Claim,
        plan: InvestigationPlan,
        evidence: EvidenceBundle,
        rulebook: Optional[Rulebook] = None,
        round_number: int = 1,
    ) -> ClaimDecision:
        """
        Execute planned checks using retrieved evidence and produce a ClaimDecision.
        
        Validates:
        - All inputs are non-None and typed.
        - Identifiers match: claim.claim_id == plan.claim_id == evidence.claim_id.
        - Plan tasks are executed in plan-specified order.
        """
        if not isinstance(claim, Claim):
            raise TypeError(f"Expected Claim instance, got {type(claim).__name__}")
        if not isinstance(plan, InvestigationPlan):
            raise TypeError(f"Expected InvestigationPlan instance, got {type(plan).__name__}")
        if not isinstance(evidence, EvidenceBundle):
            raise TypeError(f"Expected EvidenceBundle instance, got {type(evidence).__name__}")

        # Identifier consistency check across collaboration stages
        if not (claim.claim_id == plan.claim_id == evidence.claim_id):
            raise ValueError(
                f"Identifier mismatch across investigation stages: "
                f"Claim '{claim.claim_id}' vs Plan '{plan.claim_id}' vs Evidence '{evidence.claim_id}'"
            )

        rb = rulebook or load_default_rulebook()
        if not isinstance(rb, Rulebook):
            raise TypeError(f"Expected Rulebook instance, got {type(rb).__name__}")

        findings: List[Finding] = []
        tool_traces: List[Dict[str, Any]] = []

        # Execute checks strictly in the order determined by the plan
        for step_idx, rule_id in enumerate(plan.tasks, start=1):
            try:
                rule_def = rb.get_rule(rule_id)
            except KeyError:
                # Plan requested an unknown rule ID
                raise ValueError(
                    f"Plan requested rule '{rule_id}' which does not exist in Rulebook v{rb.version}"
                )

            if not rule_def.enabled:
                finding = Finding(
                    check_name=rule_id,
                    result="DISABLED",
                    risk_contribution=0.0,
                    detail=f"Rule '{rule_id}' was planned but is disabled in Rulebook v{rb.version}.",
                )
                findings.append(finding)
                tool_traces.append({
                    "step": step_idx,
                    "rule_id": rule_id,
                    "enabled": False,
                    "triggered": False,
                    "points": 0.0,
                    "plan_reason": plan.task_reasons.get(rule_id, ""),
                })
                continue

            template_fn = TEMPLATE_REGISTRY[rule_def.template_id]

            # Execute the deterministic verification tool with rule parameters
            result: ToolCheckResult = template_fn(
                claim=claim,
                score_contribution=rule_def.score_contribution,
                severity=rule_def.severity,
                rule_id=rule_id,
                **rule_def.parameters,
            )

            finding = result.to_finding()
            findings.append(finding)

            tool_traces.append({
                "step": step_idx,
                "rule_id": rule_id,
                "template_id": rule_def.template_id,
                "enabled": True,
                "triggered": result.triggered,
                "points": result.score_contribution if result.triggered else 0.0,
                "severity": result.severity,
                "observed": result.observed_value,
                "detail": result.detail,
                "plan_reason": plan.task_reasons.get(rule_id, ""),
            })

        # Score and triage using the centralized Rulebook implementation
        return rb.build_decision(
            claim_id=claim.claim_id,
            findings=findings,
            tool_traces=tool_traces,
            round_number=round_number,
        )


@dataclass(frozen=True)
class PipelineResult:
    """
    End-to-end outcome of processing a claim through the three-agent pipeline.
    """
    claim: Claim
    plan: InvestigationPlan
    evidence: EvidenceBundle
    decision: ClaimDecision
    execution_order: Tuple[str, ...] = ("Planner", "Researcher", "Investigator")


class InvestigationPipeline:
    """
    Coordinates the genuine collaboration pipeline across the three agents:
    Claim -> Planner -> Plan -> Researcher -> Evidence -> Investigator -> ClaimDecision
    """

    def __init__(
        self,
        planner: Optional[Any] = None,
        researcher: Optional[Any] = None,
        investigator: Optional[InvestigatorAgent] = None,
        rulebook: Optional[Rulebook] = None,
    ) -> None:
        from shard.agents.planner import PlannerAgent
        from shard.agents.researcher import ResearchAgent

        self.planner = planner or PlannerAgent()
        self.researcher = researcher or ResearchAgent()
        self.investigator = investigator or InvestigatorAgent()
        self.rulebook = rulebook or load_default_rulebook()

    def process(self, claim: Claim, round_number: int = 1) -> PipelineResult:
        """Process a single observable claim through the three collaboration stages."""
        # Stage 1: Planning
        plan = self.planner.plan(claim)

        # Stage 2: Research
        evidence = self.researcher.gather_evidence(claim, plan)

        # Stage 3: Investigation
        decision = self.investigator.investigate(
            claim=claim,
            plan=plan,
            evidence=evidence,
            rulebook=self.rulebook,
            round_number=round_number,
        )

        return PipelineResult(
            claim=claim,
            plan=plan,
            evidence=evidence,
            decision=decision,
        )
