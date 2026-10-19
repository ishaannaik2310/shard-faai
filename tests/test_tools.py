"""
Tests for Shard Deterministic Verification Tools.

Covers:
1. Expected result for clean claims (all 6 tools pass).
2. Expected result for each deliberately constructed anomaly across all 6 tools.
3. Threshold boundaries and borderline values.
4. Missing, zero, negative or invalid evidence handling.
5. Determinism: identical inputs produce identical ToolCheckResult.
6. Typed contracts and Finding conversion.
7. Safety: Ground-truth isolation and absence of eval/exec/network calls.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Tuple

import pytest

from shard.core.models import (
    Claim,
    Finding,
    IncidentType,
    RepairLineItem,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)
from shard.rules.templates import ToolCheckResult
from shard.rules.tools import (
    ALL_VERIFICATION_TOOLS,
    check_prior_damage,
    check_repair_benchmark,
    check_repair_line_items,
    check_staged_collision_indicators,
    check_vehicle_registration,
    check_vehicle_write_off,
    execute_all_default_tools,
)


@pytest.fixture
def clean_claim() -> Claim:
    """Fixture providing an authentic, non-fraudulent motor claim."""
    return Claim(
        claim_id="CLM-TEST-0001",
        policy_id="POL-123456",
        claimant_id="CLMT-9876",
        policy_inception_date="2025-06-01",
        incident_date="2026-03-15",
        filing_date="2026-03-18",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Standard minor front bumper collision at roundabout.",
        vehicle_registration="AB19 XYZ",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
        prior_damage_reported=False,
        claimed_amount=1200.0,
        benchmark_repair_cost=1150.0,
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(
            RepairLineItem(part_or_operation="Front bumper replacement", cost=650.0, category="PARTS"),
            RepairLineItem(part_or_operation="Bumper spray and paint finish", cost=350.0, category="PAINT"),
            RepairLineItem(part_or_operation="Fitting labour 2.0 hrs", cost=200.0, category="LABOUR"),
        ),
        linked_parties=(),
    )


# ---------------------------------------------------------------------------
# 1. Clean Claim Behaviour Across All Six Tools
# ---------------------------------------------------------------------------

def test_clean_claim_passes_all_six_tools(clean_claim: Claim) -> None:
    """A clean, authentic claim must pass all six verification tools with no triggered findings."""
    results = execute_all_default_tools(clean_claim)

    assert len(results) == 6
    for res in results:
        assert isinstance(res, ToolCheckResult)
        assert res.triggered is False, f"Tool '{res.rule_id}' unexpectedly triggered on clean claim"
        assert res.severity == "NONE"

        finding = res.to_finding()
        assert isinstance(finding, Finding)
        assert finding.result == "PASS"
        assert finding.risk_contribution == 0.0
        assert len(finding.detail) > 0


# ---------------------------------------------------------------------------
# 2. Deliberately Constructed Anomalies (One Per Tool)
# ---------------------------------------------------------------------------

def test_tool1_repair_benchmark_anomaly(clean_claim: Claim) -> None:
    """Tool 1 triggers when claimed amount significantly exceeds benchmark cost."""
    # Exaggerated quote: £3,500 claimed vs £1,000 benchmark (3.5x markup)
    inflated_claim = Claim(
        **{**clean_claim.__dict__, "claimed_amount": 3500.0, "benchmark_repair_cost": 1000.0}
    )
    result = check_repair_benchmark(inflated_claim, max_markup_ratio=1.35)

    assert result.triggered is True
    assert result.rule_id == "RULE_REPAIR_BENCHMARK"
    assert result.severity == "HIGH"
    assert result.score_contribution == 30.0
    assert "exceeds benchmark" in result.detail


def test_tool2_line_item_arithmetic_discrepancy(clean_claim: Claim) -> None:
    """Tool 2 triggers when line items sum does not equal the claimed amount."""
    # Items sum to £1,200, but claimed amount is set to £1,800
    mismatched_claim = Claim(
        **{**clean_claim.__dict__, "claimed_amount": 1800.0}
    )
    result = check_repair_line_items(mismatched_claim, max_discrepancy_gbp=1.0)

    assert result.triggered is True
    assert result.rule_id == "RULE_LINE_ITEM_ARITHMETIC"
    assert result.severity == "MEDIUM"
    assert "Arithmetic discrepancy" in result.detail or "does not match" in result.detail


def test_tool2_line_item_unrelated_item(clean_claim: Claim) -> None:
    """Tool 2 triggers when unrelated or pre-existing operations appear in line items."""
    padded_items = clean_claim.repair_line_items + (
        RepairLineItem(part_or_operation="Alloy wheel refurbishment (pre-existing)", cost=150.0, category="MISC"),
    )
    padded_claim = Claim(
        **{**clean_claim.__dict__, "claimed_amount": 1350.0, "repair_line_items": padded_items}
    )
    result = check_repair_line_items(padded_claim)

    assert result.triggered is True
    assert "unrelated or pre-existing" in result.detail


def test_tool3_vehicle_registration_anomaly(clean_claim: Claim) -> None:
    """Tool 3 triggers when vehicle registration status is not VALID."""
    unregistered_claim = Claim(
        **{**clean_claim.__dict__, "vehicle_registration_status": VehicleRegistrationStatus.UNREGISTERED.value}
    )
    result = check_vehicle_registration(unregistered_claim)

    assert result.triggered is True
    assert result.rule_id == "RULE_VEHICLE_REGISTRATION"
    assert result.severity == "HIGH"
    assert "UNREGISTERED" in result.detail


def test_tool4_vehicle_write_off_anomaly(clean_claim: Claim) -> None:
    """Tool 4 triggers when vehicle is a non-repairable statutory write-off (CAT A or B)."""
    scrapped_claim = Claim(
        **{**clean_claim.__dict__, "vehicle_write_off_status": VehicleWriteOffStatus.CAT_A.value}
    )
    result = check_vehicle_write_off(scrapped_claim)

    assert result.triggered is True
    assert result.rule_id == "RULE_WRITE_OFF_STATUS"
    assert result.severity == "CRITICAL"
    assert result.score_contribution == 40.0
    assert "CAT_A" in result.detail


def test_tool5_prior_damage_reported(clean_claim: Claim) -> None:
    """Tool 5 triggers when prior damage is reported or noted."""
    prior_dmg_claim = Claim(
        **{**clean_claim.__dict__, "prior_damage_reported": True}
    )
    result = check_prior_damage(prior_dmg_claim)

    assert result.triggered is True
    assert result.rule_id == "RULE_PRIOR_DAMAGE"
    assert result.severity == "MEDIUM"
    assert result.score_contribution == 25.0
    assert "Prior vehicle damage" in result.detail


def test_tool6_staged_collision_indicators(clean_claim: Claim) -> None:
    """Tool 6 triggers when collision occurs shortly after policy inception with linked parties."""
    # Policy inception 3 days before incident, with 2 linked parties and flagged provider
    staged_claim = Claim(
        **{
            **clean_claim.__dict__,
            "policy_inception_date": "2026-03-12",
            "incident_date": "2026-03-15",
            "linked_parties": ("CLMT-5501", "CLMT-5502"),
            "repair_provider_id": "PRV-FLAG-901",
        }
    )
    result = check_staged_collision_indicators(staged_claim, min_policy_age_days=10, min_linked_parties=2)

    assert result.triggered is True
    assert result.rule_id == "RULE_STAGED_COLLISION_INDICATORS"
    assert result.severity == "HIGH"
    assert "staged collision" in result.detail.lower()


# ---------------------------------------------------------------------------
# 3. Threshold Boundaries and Edge Cases
# ---------------------------------------------------------------------------

def test_tool1_boundary_ratio(clean_claim: Claim) -> None:
    """Tool 1 boundary: ratio exactly at 1.35 does not trigger; 1.3501 triggers."""
    benchmark = 1000.0
    exact_claim = Claim(**{**clean_claim.__dict__, "claimed_amount": 1350.0, "benchmark_repair_cost": benchmark})
    res_exact = check_repair_benchmark(exact_claim, max_markup_ratio=1.35)
    assert res_exact.triggered is False

    breach_claim = Claim(**{**clean_claim.__dict__, "claimed_amount": 1351.0, "benchmark_repair_cost": benchmark})
    res_breach = check_repair_benchmark(breach_claim, max_markup_ratio=1.35)
    assert res_breach.triggered is True


def test_tool2_arithmetic_tolerance_boundary(clean_claim: Claim) -> None:
    """Tool 2 boundary: discrepancy <= max_discrepancy_gbp does not trigger."""
    # Sum is £1,200; claimed £1,200.50 (diff £0.50 <= £1.0 tolerance)
    sub_tol_claim = Claim(**{**clean_claim.__dict__, "claimed_amount": 1200.50})
    res = check_repair_line_items(sub_tol_claim, max_discrepancy_gbp=1.0)
    assert res.triggered is False

    # Claimed £1,201.50 (diff £1.50 > £1.0 tolerance)
    over_tol_claim = Claim(**{**clean_claim.__dict__, "claimed_amount": 1201.50})
    res_over = check_repair_line_items(over_tol_claim, max_discrepancy_gbp=1.0)
    assert res_over.triggered is True


def test_missing_and_invalid_evidence_handling(clean_claim: Claim) -> None:
    """Tools handle zero or negative amounts and empty lists gracefully without crashing."""
    # Tool 1: Zero benchmark with positive claim
    zero_bench_claim = Claim(**{**clean_claim.__dict__, "benchmark_repair_cost": 0.0})
    res_zero_bench = check_repair_benchmark(zero_bench_claim)
    assert res_zero_bench.triggered is True
    assert "Invalid or zero benchmark" in res_zero_bench.detail

    # Tool 2: Empty line items on positive claim
    empty_items_claim = Claim(**{**clean_claim.__dict__, "repair_line_items": ()})
    res_empty_items = check_repair_line_items(empty_items_claim)
    assert res_empty_items.triggered is True
    assert "no itemised repair operations" in res_empty_items.detail

    # Tool 3: Missing plate string
    missing_plate_claim = Claim(**{**clean_claim.__dict__, "vehicle_registration": "   "})
    res_plate = check_vehicle_registration(missing_plate_claim)
    assert res_plate.triggered is True
    assert "missing or empty" in res_plate.detail


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

def test_tool_determinism(clean_claim: Claim) -> None:
    """Calling verification tools multiple times with identical claims yields bit-identical results."""
    res_run1 = [tool(clean_claim) for tool in ALL_VERIFICATION_TOOLS]
    res_run2 = [tool(clean_claim) for tool in ALL_VERIFICATION_TOOLS]

    assert res_run1 == res_run2
    for r1, r2 in zip(res_run1, res_run2):
        assert r1.to_finding() == r2.to_finding()


# ---------------------------------------------------------------------------
# 5. Safety Invariants: No GroundTruthVault Dependency & No eval/exec
# ---------------------------------------------------------------------------

def test_tools_do_not_import_ground_truth_vault() -> None:
    """
    CRITICAL INVARIANT (TI-01):
    Tools module must never import GroundTruthVault or access hidden labels.
    """
    rule_dir = Path(__file__).resolve().parent.parent / "shard" / "rules"
    for py_file in (rule_dir / "tools.py", rule_dir / "templates.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "vault" not in alias.name.lower(), f"Forbidden import '{alias.name}' in {py_file.name}"
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert "vault" not in mod_name.lower(), f"Forbidden import from '{mod_name}' in {py_file.name}"
                for alias in node.names:
                    assert "groundtruth" not in alias.name.lower(), f"Forbidden symbol '{alias.name}' in {py_file.name}"


def test_no_eval_or_exec_in_rule_modules() -> None:
    """
    CRITICAL INVARIANT (TI-04):
    Rule execution must never invoke eval() or exec().
    """
    rule_dir = Path(__file__).resolve().parent.parent / "shard" / "rules"
    for py_file in rule_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec"), (
                    f"Forbidden dynamic call '{node.func.id}' found in {py_file.name}:{node.lineno}"
                )
