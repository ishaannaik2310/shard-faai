"""
Shard — Deterministic Verification Tools.

This module provides six standalone, independently callable Python tools implementing
deterministic checks on observable motor insurance claims.

ARCHITECTURAL TRUST BOUNDARY (TI-01):
- Tools accept ONLY observable `Claim` instances (or explicitly supplied research bundles).
- Tools MUST NEVER accept `GenerationBatch`, `GroundTruthVault`, or hidden labels.
- Deterministic: Identical inputs produce bit-identical ToolCheckResult outcomes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from shard.core.models import Claim
from shard.rules.templates import (
    TEMPLATE_REGISTRY,
    ToolCheckResult,
    tpl_cost_benchmark,
    tpl_line_item_arithmetic,
    tpl_prior_damage,
    tpl_staged_collision,
    tpl_vehicle_registration,
    tpl_vehicle_write_off,
)


def check_repair_benchmark(
    claim: Claim,
    max_markup_ratio: float = 1.35,
    score_contribution: float = 30.0,
    severity: str = "HIGH",
    rule_id: str = "RULE_REPAIR_BENCHMARK",
) -> ToolCheckResult:
    """Tool 1: Evaluate claimed repair total against independent benchmark repair cost."""
    return tpl_cost_benchmark(
        claim=claim,
        max_markup_ratio=max_markup_ratio,
        score_contribution=score_contribution,
        severity=severity,
        rule_id=rule_id,
    )


def check_repair_line_items(
    claim: Claim,
    max_discrepancy_gbp: float = 1.0,
    flag_unrelated_items: bool = True,
    score_contribution: float = 20.0,
    severity: str = "MEDIUM",
    rule_id: str = "RULE_LINE_ITEM_ARITHMETIC",
) -> ToolCheckResult:
    """Tool 2: Verify repair line-item arithmetic consistency and inspect for unrelated operations."""
    return tpl_line_item_arithmetic(
        claim=claim,
        max_discrepancy_gbp=max_discrepancy_gbp,
        flag_unrelated_items=flag_unrelated_items,
        score_contribution=score_contribution,
        severity=severity,
        rule_id=rule_id,
    )


def check_vehicle_registration(
    claim: Claim,
    allowed_statuses: Sequence[str] = ("VALID",),
    score_contribution: float = 35.0,
    severity: str = "HIGH",
    rule_id: str = "RULE_VEHICLE_REGISTRATION",
) -> ToolCheckResult:
    """Tool 3: Verify vehicle registration status with vehicle licensing database."""
    return tpl_vehicle_registration(
        claim=claim,
        allowed_statuses=allowed_statuses,
        score_contribution=score_contribution,
        severity=severity,
        rule_id=rule_id,
    )


def check_vehicle_write_off(
    claim: Claim,
    disallowed_categories: Sequence[str] = ("CAT_A", "CAT_B"),
    score_contribution: float = 40.0,
    severity: str = "CRITICAL",
    rule_id: str = "RULE_WRITE_OFF_STATUS",
) -> ToolCheckResult:
    """Tool 4: Verify vehicle write-off history for scrap / break-only structural bans."""
    return tpl_vehicle_write_off(
        claim=claim,
        disallowed_categories=disallowed_categories,
        score_contribution=score_contribution,
        severity=severity,
        rule_id=rule_id,
    )


def check_prior_damage(
    claim: Claim,
    score_contribution: float = 25.0,
    severity: str = "MEDIUM",
    rule_id: str = "RULE_PRIOR_DAMAGE",
) -> ToolCheckResult:
    """Tool 5: Check self-reported or known prior damage history for overlap."""
    return tpl_prior_damage(
        claim=claim,
        score_contribution=score_contribution,
        severity=severity,
        rule_id=rule_id,
    )


def check_staged_collision_indicators(
    claim: Claim,
    min_policy_age_days: int = 10,
    min_linked_parties: int = 2,
    score_contribution: float = 35.0,
    severity: str = "HIGH",
    rule_id: str = "RULE_STAGED_COLLISION_INDICATORS",
) -> ToolCheckResult:
    """Tool 6: Evaluate policy inception velocity, co-claimant linkages, and garage network flags."""
    return tpl_staged_collision(
        claim=claim,
        min_policy_age_days=min_policy_age_days,
        min_linked_parties=min_linked_parties,
        score_contribution=score_contribution,
        severity=severity,
        rule_id=rule_id,
    )


ALL_VERIFICATION_TOOLS = (
    check_repair_benchmark,
    check_repair_line_items,
    check_vehicle_registration,
    check_vehicle_write_off,
    check_prior_damage,
    check_staged_collision_indicators,
)


def execute_all_default_tools(claim: Claim) -> List[ToolCheckResult]:
    """Execute all six baseline tools against an observable claim with default thresholds."""
    return [tool(claim) for tool in ALL_VERIFICATION_TOOLS]
