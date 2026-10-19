"""
Shard — Parameterised Rule Templates.

This module defines deterministic, bounded Python evaluation templates for motor
insurance fraud verification.

CRITICAL INVARIANTS:
1. Pure deterministic logic: NO eval(), NO exec(), NO dynamic code execution.
2. Ground-truth isolation: NO imports or calls to GroundTruthVault.
3. Parameterised: All thresholds and weights are passed as typed parameters.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from shard.core.models import (
    Claim,
    Finding,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)


@dataclass(frozen=True)
class ToolCheckResult:
    """
    Structured outcome of an individual deterministic verification check.
    
    Attributes:
        rule_id: Stable identifier of the rule (e.g., 'RULE_REPAIR_BENCHMARK').
        check_name: Human-readable category/tool name.
        triggered: True if evidence breached the configured threshold.
        score_contribution: Points added to the claim risk score (0.0 to 100.0).
        severity: Qualitative severity level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'NONE').
        observed_value: Actual metric or value extracted from the claim.
        threshold_value: Target or boundary value used in the comparison.
        detail: Human-readable explanation of the comparison.
    """
    rule_id: str
    check_name: str
    triggered: bool
    score_contribution: float
    severity: str
    observed_value: Any
    threshold_value: Any
    detail: str

    def to_finding(self) -> Finding:
        """Convert result into an audit Finding contract for ClaimDecision."""
        status = f"FLAGGED ({self.severity})" if self.triggered else "PASS"
        points = self.score_contribution if self.triggered else 0.0
        return Finding(
            check_name=self.rule_id if self.rule_id else self.check_name,
            result=status,
            risk_contribution=points,
            detail=self.detail,
        )


# ---------------------------------------------------------------------------
# Template 1: Repair Cost Benchmark Comparison
# ---------------------------------------------------------------------------

def tpl_cost_benchmark(
    claim: Claim,
    max_markup_ratio: float = 1.35,
    score_contribution: float = 30.0,
    severity: str = "HIGH",
    rule_id: str = "RULE_REPAIR_BENCHMARK",
) -> ToolCheckResult:
    """Compare claimed repair total against independent benchmark repair cost."""
    check_name = "BENCHMARK_REPAIR_COST"

    # Edge cases: missing or non-positive benchmark
    if claim.benchmark_repair_cost <= 0.0:
        if claim.claimed_amount <= 0.0:
            return ToolCheckResult(
                rule_id=rule_id,
                check_name=check_name,
                triggered=False,
                score_contribution=score_contribution,
                severity="NONE",
                observed_value=0.0,
                threshold_value=max_markup_ratio,
                detail="Zero claimed amount and zero benchmark (zero-cost claim).",
            )
        return ToolCheckResult(
            rule_id=rule_id,
            check_name=check_name,
            triggered=True,
            score_contribution=score_contribution,
            severity=severity,
            observed_value={"claimed": claim.claimed_amount, "benchmark": claim.benchmark_repair_cost},
            threshold_value=max_markup_ratio,
            detail=f"Invalid or zero benchmark cost (£{claim.benchmark_repair_cost:.2f}) for non-zero claim (£{claim.claimed_amount:.2f}).",
        )

    if claim.claimed_amount < 0.0:
        return ToolCheckResult(
            rule_id=rule_id,
            check_name=check_name,
            triggered=True,
            score_contribution=score_contribution,
            severity=severity,
            observed_value=claim.claimed_amount,
            threshold_value=0.0,
            detail=f"Negative claimed repair amount (£{claim.claimed_amount:.2f}) is invalid.",
        )

    ratio = round(claim.claimed_amount / claim.benchmark_repair_cost, 4)
    triggered = ratio > max_markup_ratio

    if triggered:
        markup_pct = (ratio - 1.0) * 100.0
        allowed_pct = (max_markup_ratio - 1.0) * 100.0
        detail = (
            f"Claimed repair cost (£{claim.claimed_amount:.2f}) exceeds benchmark "
            f"(£{claim.benchmark_repair_cost:.2f}) by {markup_pct:.1f}% "
            f"(configured threshold: max {allowed_pct:.1f}% markup)."
        )
    else:
        detail = (
            f"Claimed repair cost (£{claim.claimed_amount:.2f}) is within acceptable "
            f"benchmark variance ({ratio:.2f}x vs threshold {max_markup_ratio:.2f}x)."
        )

    return ToolCheckResult(
        rule_id=rule_id,
        check_name=check_name,
        triggered=triggered,
        score_contribution=score_contribution,
        severity=severity if triggered else "NONE",
        observed_value=ratio,
        threshold_value=max_markup_ratio,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Template 2: Repair Line-Item Arithmetic and Integrity
# ---------------------------------------------------------------------------

def tpl_line_item_arithmetic(
    claim: Claim,
    max_discrepancy_gbp: float = 1.0,
    flag_unrelated_items: bool = True,
    score_contribution: float = 20.0,
    severity: str = "MEDIUM",
    rule_id: str = "RULE_LINE_ITEM_ARITHMETIC",
) -> ToolCheckResult:
    """Verify sum of itemised repair operations matches claimed total and check for anomalous entries."""
    check_name = "LINE_ITEM_INTEGRITY"

    if not claim.repair_line_items:
        # If non-zero claim has no itemised quote
        if claim.claimed_amount > 0.0:
            return ToolCheckResult(
                rule_id=rule_id,
                check_name=check_name,
                triggered=True,
                score_contribution=score_contribution,
                severity=severity,
                observed_value=0,
                threshold_value="non-empty line items",
                detail=f"Claim for £{claim.claimed_amount:.2f} has no itemised repair operations.",
            )
        return ToolCheckResult(
            rule_id=rule_id,
            check_name=check_name,
            triggered=False,
            score_contribution=score_contribution,
            severity="NONE",
            observed_value=0,
            threshold_value=0.0,
            detail="Zero-cost claim with empty line items verified.",
        )

    item_sum = round(sum(item.cost for item in claim.repair_line_items), 2)
    discrepancy = round(abs(item_sum - claim.claimed_amount), 2)

    unrelated_keywords = ("unrelated", "pre-existing", "preexisting", "scratch respray", "refurbishment (pre-existing)")
    anomalous_items: List[str] = []

    if flag_unrelated_items:
        for item in claim.repair_line_items:
            op_lower = item.part_or_operation.lower()
            if any(kw in op_lower for kw in unrelated_keywords):
                anomalous_items.append(item.part_or_operation)

    arithmetic_failed = discrepancy > max_discrepancy_gbp
    anomalies_found = len(anomalous_items) > 0
    triggered = arithmetic_failed or anomalies_found

    if arithmetic_failed and anomalies_found:
        detail = (
            f"Arithmetic discrepancy of £{discrepancy:.2f} (item sum £{item_sum:.2f} vs claimed £{claim.claimed_amount:.2f}) "
            f"and unrelated line items identified: {anomalous_items}."
        )
    elif arithmetic_failed:
        detail = (
            f"Line item total £{item_sum:.2f} does not match claimed amount £{claim.claimed_amount:.2f} "
            f"(discrepancy £{discrepancy:.2f} exceeds threshold £{max_discrepancy_gbp:.2f})."
        )
    elif anomalies_found:
        detail = (
            f"Line item arithmetic verified (£{item_sum:.2f}), but unrelated or pre-existing operations "
            f"detected in quote: {anomalous_items}."
        )
    else:
        detail = f"Line item arithmetic verified (£{item_sum:.2f} matches claimed £{claim.claimed_amount:.2f}) with no anomalous entries."

    return ToolCheckResult(
        rule_id=rule_id,
        check_name=check_name,
        triggered=triggered,
        score_contribution=score_contribution,
        severity=severity if triggered else "NONE",
        observed_value={"item_sum": item_sum, "anomalous_items": anomalous_items},
        threshold_value={"max_discrepancy_gbp": max_discrepancy_gbp},
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Template 3: Vehicle Registration and Existence Status
# ---------------------------------------------------------------------------

def tpl_vehicle_registration(
    claim: Claim,
    allowed_statuses: Sequence[str] = (VehicleRegistrationStatus.VALID.value,),
    score_contribution: float = 35.0,
    severity: str = "HIGH",
    rule_id: str = "RULE_VEHICLE_REGISTRATION",
) -> ToolCheckResult:
    """Verify that vehicle registration is officially recognised and active with licensing database."""
    check_name = "VERIFY_VEHICLE_REGISTRATION"

    reg_status = claim.vehicle_registration_status.strip().upper()
    plate = claim.vehicle_registration.strip()

    plate_missing = len(plate) == 0
    status_disallowed = reg_status not in [s.upper() for s in allowed_statuses]
    triggered = plate_missing or status_disallowed

    if plate_missing:
        detail = "Vehicle registration number is missing or empty."
    elif status_disallowed:
        detail = f"Vehicle registration status is '{reg_status}' (allowed statuses: {list(allowed_statuses)})."
    else:
        detail = f"Vehicle registration '{plate}' verified active (status: {reg_status})."

    return ToolCheckResult(
        rule_id=rule_id,
        check_name=check_name,
        triggered=triggered,
        score_contribution=score_contribution,
        severity=severity if triggered else "NONE",
        observed_value={"plate": plate, "status": reg_status},
        threshold_value=list(allowed_statuses),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Template 4: Vehicle Write-Off History Check
# ---------------------------------------------------------------------------

def tpl_vehicle_write_off(
    claim: Claim,
    disallowed_categories: Sequence[str] = (
        VehicleWriteOffStatus.CAT_A.value,
        VehicleWriteOffStatus.CAT_B.value,
    ),
    score_contribution: float = 40.0,
    severity: str = "CRITICAL",
    rule_id: str = "RULE_WRITE_OFF_STATUS",
) -> ToolCheckResult:
    """Inspect vehicle history for prior non-repairable write-off markers (CAT A or CAT B scrap/crush)."""
    check_name = "VERIFY_WRITE_OFF_STATUS"

    status = claim.vehicle_write_off_status.strip().upper()
    disallowed_upper = [c.upper() for c in disallowed_categories]
    triggered = status in disallowed_upper

    if triggered:
        detail = (
            f"Vehicle is recorded under statutory write-off category '{status}' (non-repairable / scrap). "
            f"Vehicle cannot lawfully be repaired or driven on public roads."
        )
    else:
        detail = f"Vehicle write-off status verified as '{status}' (clean history or permitted repair category)."

    return ToolCheckResult(
        rule_id=rule_id,
        check_name=check_name,
        triggered=triggered,
        score_contribution=score_contribution,
        severity=severity if triggered else "NONE",
        observed_value=status,
        threshold_value=disallowed_upper,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Template 5: Prior Damage Verification Check
# ---------------------------------------------------------------------------

def tpl_prior_damage(
    claim: Claim,
    score_contribution: float = 25.0,
    severity: str = "MEDIUM",
    rule_id: str = "RULE_PRIOR_DAMAGE",
) -> ToolCheckResult:
    """Check for self-reported or recorded prior damage that may overlap with claimed damage."""
    check_name = "CHECK_PRIOR_DAMAGE"

    has_prior_flag = bool(claim.prior_damage_reported)
    desc_mention = "prior damage" in claim.incident_description.lower()
    triggered = has_prior_flag or desc_mention

    if triggered:
        detail = (
            f"Prior vehicle damage identified (prior_damage_reported={has_prior_flag}, "
            f"description_mention={desc_mention}). Potential bundling of pre-existing damage."
        )
    else:
        detail = "No self-reported or recorded prior damage identified for this vehicle."

    return ToolCheckResult(
        rule_id=rule_id,
        check_name=check_name,
        triggered=triggered,
        score_contribution=score_contribution,
        severity=severity if triggered else "NONE",
        observed_value={"prior_damage_reported": has_prior_flag, "description_mention": desc_mention},
        threshold_value=False,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Template 6: Staged Collision and Party Network Indicators
# ---------------------------------------------------------------------------

def tpl_staged_collision(
    claim: Claim,
    min_policy_age_days: int = 10,
    min_linked_parties: int = 2,
    score_contribution: float = 35.0,
    severity: str = "HIGH",
    rule_id: str = "RULE_STAGED_COLLISION_INDICATORS",
) -> ToolCheckResult:
    """Detect anomalous policy inception timing and linked co-claimant / provider networks."""
    check_name = "CHECK_STAGED_COLLISION"

    try:
        incept = datetime.date.fromisoformat(claim.policy_inception_date)
        incident = datetime.date.fromisoformat(claim.incident_date)
        tenure_days = (incident - incept).days
    except (ValueError, TypeError):
        tenure_days = -1

    n_linked = len(claim.linked_parties)
    is_flagged_provider = claim.repair_provider_id.upper().startswith("PRV-FLAG")

    short_tenure = 0 <= tenure_days <= min_policy_age_days
    multiple_parties = n_linked >= min_linked_parties

    # Staged collision pattern: rapid claim post inception with multiple linked parties or flagged garage
    triggered = (short_tenure and multiple_parties) or (short_tenure and is_flagged_provider)

    if triggered:
        reasons = []
        if short_tenure:
            reasons.append(f"short policy tenure ({tenure_days} days post-inception <= {min_policy_age_days}d threshold)")
        if multiple_parties:
            reasons.append(f"{n_linked} linked parties (>= {min_linked_parties} threshold)")
        if is_flagged_provider:
            reasons.append(f"flagged repair provider '{claim.repair_provider_id}'")
        detail = f"High-risk staged collision indicator triggered: {', '.join(reasons)}."
    else:
        detail = (
            f"Policy tenure ({tenure_days} days), party count ({n_linked}), and provider "
            f"('{claim.repair_provider_id}') conform to standard operational parameters."
        )

    return ToolCheckResult(
        rule_id=rule_id,
        check_name=check_name,
        triggered=triggered,
        score_contribution=score_contribution,
        severity=severity if triggered else "NONE",
        observed_value={
            "tenure_days": tenure_days,
            "linked_parties": n_linked,
            "provider_id": claim.repair_provider_id,
        },
        threshold_value={
            "min_policy_age_days": min_policy_age_days,
            "min_linked_parties": min_linked_parties,
        },
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Central Template Registry
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: Dict[str, Callable[..., ToolCheckResult]] = {
    "TPL_COST_BENCHMARK": tpl_cost_benchmark,
    "TPL_LINE_ITEM_ARITHMETIC": tpl_line_item_arithmetic,
    "TPL_VEHICLE_REGISTRATION": tpl_vehicle_registration,
    "TPL_WRITE_OFF_STATUS": tpl_vehicle_write_off,
    "TPL_PRIOR_DAMAGE": tpl_prior_damage,
    "TPL_STAGED_COLLISION": tpl_staged_collision,
}
