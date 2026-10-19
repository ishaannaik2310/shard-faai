"""
Shard — Planner Agent.

Inspects an observable Claim and produces a contextual, ordered InvestigationPlan.

CRITICAL INVARIANTS:
1. Ground-Truth Isolation (TI-01): The Planner MUST NEVER import GroundTruthVault,
   request hidden labels, or use evaluation-only fraud metadata.
2. Determinism (TI-07): Identical observable claims produce identical plans.
3. Traceability: Every task includes explicit contextual relevance and ordering rationale.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

from shard.core.models import Claim, InvestigationPlan

# Standard complete catalogue of verification rule IDs
ALL_RULE_TASKS: Tuple[str, ...] = (
    "RULE_VEHICLE_REGISTRATION",
    "RULE_WRITE_OFF_STATUS",
    "RULE_REPAIR_BENCHMARK",
    "RULE_LINE_ITEM_ARITHMETIC",
    "RULE_PRIOR_DAMAGE",
    "RULE_STAGED_COLLISION_INDICATORS",
)


class PlannerAgent:
    """
    Deterministic planning agent that analyses observable claim attributes to
    produce an ordered investigation strategy.
    """

    def plan(self, claim: Claim) -> InvestigationPlan:
        """
        Produce a typed InvestigationPlan for an observable claim.

        Planning Strategy:
        - For broad MVP coverage, all applicable checks are scheduled.
        - Checks are contextually reordered to front-load the most consequential
          potential anomalies based on observable claim indicators.
        """
        if not isinstance(claim, Claim):
            raise TypeError(f"Expected Claim instance, got {type(claim).__name__}")

        # Assess observable indicators
        has_cost_anomaly = (
            claim.benchmark_repair_cost > 0.0
            and (claim.claimed_amount / claim.benchmark_repair_cost) > 1.25
        )
        has_vehicle_anomaly = (
            claim.vehicle_registration_status.upper() != "VALID"
            or claim.vehicle_write_off_status.upper() not in ("CLEAN", "NONE")
        )
        has_prior_damage = bool(claim.prior_damage_reported)

        # Policy tenure calculation
        tenure_days = -1
        try:
            incept = datetime.date.fromisoformat(claim.policy_inception_date)
            incident = datetime.date.fromisoformat(claim.incident_date)
            tenure_days = (incident - incept).days
        except (ValueError, TypeError):
            pass

        # Specific staged collision triggers (aligned with rulebook min_policy_age_days: 10)
        rapid_inception = (0 <= tenure_days <= 10)
        multi_party = len(claim.linked_parties) >= 2
        flagged_provider = claim.repair_provider_id.upper().startswith("PRV-FLAG")
        has_staged_indicators = rapid_inception or multi_party or flagged_provider

        # Determine prioritized ordering and task rationales
        prioritized: List[str] = []
        task_reasons: Dict[str, str] = {}
        missing_reqs: List[str] = []

        # Check for missing basic information
        if not claim.vehicle_registration.strip():
            missing_reqs.append("vehicle_registration_plate_missing")
        if not claim.repair_line_items and claim.claimed_amount > 0:
            missing_reqs.append("repair_line_items_missing")

        # 1. Vehicle Existence / Identity Priority
        if has_vehicle_anomaly:
            prioritized.extend(["RULE_VEHICLE_REGISTRATION", "RULE_WRITE_OFF_STATUS"])
            task_reasons["RULE_VEHICLE_REGISTRATION"] = (
                f"High priority: Observable registration status is '{claim.vehicle_registration_status}'."
            )
            task_reasons["RULE_WRITE_OFF_STATUS"] = (
                f"High priority: Observable write-off marker is '{claim.vehicle_write_off_status}'."
            )
        else:
            task_reasons["RULE_VEHICLE_REGISTRATION"] = "Standard verification of official registration status."
            task_reasons["RULE_WRITE_OFF_STATUS"] = "Standard verification of vehicle total-loss history."

        # 2. Staged Collision / Network Priority
        if has_staged_indicators:
            if "RULE_STAGED_COLLISION_INDICATORS" not in prioritized:
                prioritized.append("RULE_STAGED_COLLISION_INDICATORS")
            staged_triggers: List[str] = []
            if rapid_inception:
                staged_triggers.append(f"rapid inception tenure ({tenure_days}d <= 10d)")
            if multi_party:
                staged_triggers.append(f"{len(claim.linked_parties)} linked parties")
            if flagged_provider:
                staged_triggers.append(f"flagged provider '{claim.repair_provider_id}'")
            task_reasons["RULE_STAGED_COLLISION_INDICATORS"] = (
                f"High priority: Observable staged collision indicators detected ({', '.join(staged_triggers)})."
            )
        else:
            task_reasons["RULE_STAGED_COLLISION_INDICATORS"] = "Baseline review of collision timing and linked parties."

        # 3. Cost Benchmark & Line-Items Priority
        if has_cost_anomaly:
            for task in ("RULE_REPAIR_BENCHMARK", "RULE_LINE_ITEM_ARITHMETIC"):
                if task not in prioritized:
                    prioritized.append(task)
            markup_pct = ((claim.claimed_amount / claim.benchmark_repair_cost) - 1.0) * 100.0
            task_reasons["RULE_REPAIR_BENCHMARK"] = (
                f"High priority: Claimed repair cost exceeds benchmark by {markup_pct:.1f}%."
            )
            task_reasons["RULE_LINE_ITEM_ARITHMETIC"] = (
                f"High priority: Detailed line-item audit required due to £{claim.claimed_amount:.2f} quote total."
            )
        else:
            task_reasons["RULE_REPAIR_BENCHMARK"] = "Standard comparison against independent repair estimatics."
            task_reasons["RULE_LINE_ITEM_ARITHMETIC"] = "Standard arithmetic and component verification of repair quote."

        # 4. Prior Damage Priority
        if has_prior_damage:
            if "RULE_PRIOR_DAMAGE" not in prioritized:
                prioritized.append("RULE_PRIOR_DAMAGE")
            task_reasons["RULE_PRIOR_DAMAGE"] = (
                "High priority: Claimant self-reported prior damage; verify potential damage overlap."
            )
        else:
            task_reasons["RULE_PRIOR_DAMAGE"] = "Baseline verification of prior loss records."

        # Append remaining baseline checks to ensure 100% audit coverage
        for task in ALL_RULE_TASKS:
            if task not in prioritized:
                prioritized.append(task)

        # Build strategy rationale
        if has_vehicle_anomaly:
            strategy = "IDENTITY_FIRST — Front-load vehicle existence and write-off status checks."
        elif has_staged_indicators:
            strategy = "NETWORK_FIRST — Front-load inception timing, linked parties, and provider network checks."
        elif has_cost_anomaly:
            strategy = "COST_FIRST — Front-load repair cost benchmarking and line-item integrity audits."
        elif has_prior_damage:
            strategy = "DAMAGE_OVERLAP_FIRST — Front-load prior loss history and damage verification."
        else:
            strategy = "STANDARD_BASELINE — Full operational verification across all six fraud control points."

        return InvestigationPlan(
            claim_id=claim.claim_id,
            tasks=tuple(prioritized),
            rationale=strategy,
            task_reasons=task_reasons,
            missing_evidence_requirements=tuple(missing_reqs),
        )
