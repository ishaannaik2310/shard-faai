"""
Tests for Shard Versioned Rulebook and Risk Scoring Engine.

Covers:
1. Correct loading and parsing of data/rulebook_v1.json.
2. Uniqueness of rule identifiers.
3. Validation and rejection of invalid configurations (negative weights, bad ranges, malformed JSON).
4. Enabled vs disabled rule behavior.
5. Risk scoring bounds [0.0, 100.0] and non-inflation.
6. Correct disposition triage mapping at exact boundaries (<25: APPROVE, 25-59: REFER, >=60: SUSPECTED_FRAUD).
7. Non-empty, traceable audit findings for every evaluated claim (TI-02).
8. Reproducibility of decisions.
9. Safety: Ground-truth isolation and absence of eval/exec/network calls.
"""

from __future__ import annotations

import ast
import inspect
import json
import tempfile
from pathlib import Path

import pytest

from shard.core.models import (
    Claim,
    ClaimDecision,
    Disposition,
    Finding,
    IncidentType,
    RepairLineItem,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)
from shard.rules.rulebook import (
    DispositionThresholds,
    RuleDefinition,
    Rulebook,
    load_default_rulebook,
)


@pytest.fixture
def clean_claim() -> Claim:
    """Fixture providing an authentic motor insurance claim."""
    return Claim(
        claim_id="CLM-RULE-0001",
        policy_id="POL-112233",
        claimant_id="CLMT-4455",
        policy_inception_date="2025-05-01",
        incident_date="2026-04-10",
        filing_date="2026-04-13",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Low speed parking scrape against low wall.",
        vehicle_registration="WM20 BCD",
        vehicle_make="Volkswagen",
        vehicle_model="Golf",
        vehicle_year=2020,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
        prior_damage_reported=False,
        claimed_amount=950.0,
        benchmark_repair_cost=920.0,
        repair_provider_id="PRV-NORM-102",
        repair_line_items=(
            RepairLineItem(part_or_operation="Side door respray", cost=600.0, category="PAINT"),
            RepairLineItem(part_or_operation="Door handle blend", cost=350.0, category="LABOUR"),
        ),
        linked_parties=(),
    )


# ---------------------------------------------------------------------------
# 1. Loading and Initial Configuration
# ---------------------------------------------------------------------------

def test_load_default_rulebook_success() -> None:
    """The canonical baseline rulebook (data/rulebook_v1.json) must load without error."""
    rulebook = load_default_rulebook()

    assert isinstance(rulebook, Rulebook)
    assert rulebook.version == "1.0.0"
    assert len(rulebook.rules) == 6
    assert rulebook.disposition_thresholds.approve_below == 25.0
    assert rulebook.disposition_thresholds.refer_below == 60.0
    assert rulebook.disposition_thresholds.suspected_fraud_at_or_above == 60.0
    assert len(rulebook.history) >= 1


def test_unique_rule_identifiers() -> None:
    """Rulebook must enforce uniqueness of all rule_id strings."""
    rulebook = load_default_rulebook()
    rule_ids = [r.rule_id for r in rulebook.rules]

    assert len(rule_ids) == len(set(rule_ids)), f"Duplicate rule IDs detected: {rule_ids}"

    # Attempting to construct a rulebook with duplicate rule_id must raise ValueError
    dup_rules = rulebook.rules + [rulebook.rules[0]]
    with pytest.raises(ValueError, match="Duplicate rule_id"):
        Rulebook(
            version="1.0.1",
            name="Dup",
            description="",
            disposition_thresholds=rulebook.disposition_thresholds,
            rules=dup_rules,
        )


# ---------------------------------------------------------------------------
# 2. Rejection of Invalid Configurations
# ---------------------------------------------------------------------------

def test_reject_invalid_disposition_thresholds() -> None:
    """Thresholds with approve_below > refer_below or out of [0, 100] must be rejected."""
    with pytest.raises(ValueError, match="Invalid disposition thresholds"):
        DispositionThresholds(approve_below=70.0, refer_below=30.0)

    with pytest.raises(ValueError, match="Invalid disposition thresholds"):
        DispositionThresholds(approve_below=-5.0, refer_below=50.0)

    with pytest.raises(ValueError, match="Invalid disposition thresholds"):
        DispositionThresholds(approve_below=20.0, refer_below=105.0)

    with pytest.raises(ValueError, match="must equal refer_below"):
        DispositionThresholds(approve_below=20.0, refer_below=60.0, suspected_fraud_at_or_above=70.0)


def test_reject_invalid_rule_definition() -> None:
    """Invalid rule definitions (unknown template, negative weight, empty ID) must raise ValueError."""
    # Unknown template
    with pytest.raises(ValueError, match="Unknown template_id"):
        RuleDefinition(
            rule_id="RULE_TEST",
            template_id="TPL_NONEXISTENT",
            description="Test",
        )

    # Negative score contribution
    with pytest.raises(ValueError, match="score_contribution must be between 0.0 and 100.0"):
        RuleDefinition(
            rule_id="RULE_TEST",
            template_id="TPL_COST_BENCHMARK",
            description="Test",
            score_contribution=-10.0,
        )

    # Score contribution > 100
    with pytest.raises(ValueError, match="score_contribution must be between 0.0 and 100.0"):
        RuleDefinition(
            rule_id="RULE_TEST",
            template_id="TPL_COST_BENCHMARK",
            description="Test",
            score_contribution=150.0,
        )

    # Empty rule_id
    with pytest.raises(ValueError, match="rule_id must be a non-empty string"):
        RuleDefinition(
            rule_id="",
            template_id="TPL_COST_BENCHMARK",
            description="Test",
        )


def test_reject_malformed_and_missing_json() -> None:
    """Loading missing or corrupt JSON files must raise explicit errors."""
    with pytest.raises(FileNotFoundError):
        Rulebook.load("non_existent_path.json")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
        tmp.write("{ invalid json content ;;;")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(ValueError, match="Malformed JSON"):
            Rulebook.load(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Enabled vs Disabled Behavior
# ---------------------------------------------------------------------------

def test_disabled_rule_behavior(clean_claim: Claim) -> None:
    """A disabled rule must not contribute points or alter the claim disposition."""
    rulebook = load_default_rulebook()

    # Create an exaggerated claim that would trigger RULE_REPAIR_BENCHMARK (weight 30.0)
    exaggerated_claim = Claim(
        **{
            **clean_claim.__dict__,
            "claimed_amount": 3000.0,
            "benchmark_repair_cost": 1000.0,
            "repair_line_items": (RepairLineItem("Side door respray", 3000.0, "PAINT"),),
        }
    )

    # With rule enabled: triggers 30.0 points -> REFER (25 <= score < 60)
    decision_enabled = rulebook.evaluate_claim(exaggerated_claim)
    assert decision_enabled.risk_score == 30.0
    assert decision_enabled.disposition == Disposition.REFER

    # Disable RULE_REPAIR_BENCHMARK
    disabled_rules = [
        RuleDefinition(
            rule_id=r.rule_id,
            template_id=r.template_id,
            description=r.description,
            enabled=False if r.rule_id == "RULE_REPAIR_BENCHMARK" else r.enabled,
            severity=r.severity,
            score_contribution=r.score_contribution,
            parameters=r.parameters,
        )
        for r in rulebook.rules
    ]
    rulebook_disabled = Rulebook(
        version="1.0.1",
        name=rulebook.name,
        description=rulebook.description,
        disposition_thresholds=rulebook.disposition_thresholds,
        rules=disabled_rules,
    )

    decision_disabled = rulebook_disabled.evaluate_claim(exaggerated_claim)
    assert decision_disabled.risk_score == 0.0
    assert decision_disabled.disposition == Disposition.APPROVE

    # Finding still exists for audit trail but marked as DISABLED
    benchmark_finding = next(f for f in decision_disabled.findings if f.check_name == "RULE_REPAIR_BENCHMARK")
    assert benchmark_finding.result == "DISABLED"
    assert benchmark_finding.risk_contribution == 0.0


# ---------------------------------------------------------------------------
# 4. Risk Scoring & Disposition Triage Boundaries
# ---------------------------------------------------------------------------

def test_clean_claim_scoring(clean_claim: Claim) -> None:
    """Clean claim receives 0.0 risk score and APPROVE disposition."""
    rulebook = load_default_rulebook()
    decision = rulebook.evaluate_claim(clean_claim)

    assert decision.risk_score == 0.0
    assert decision.disposition == Disposition.APPROVE
    assert len(decision.findings) == 6
    for f in decision.findings:
        assert f.result == "PASS"
        assert f.risk_contribution == 0.0


@pytest.mark.parametrize(
    "score,expected_disposition",
    [
        (0.0, Disposition.APPROVE),
        (24.9, Disposition.APPROVE),
        (25.0, Disposition.REFER),
        (45.0, Disposition.REFER),
        (59.9, Disposition.REFER),
        (60.0, Disposition.SUSPECTED_FRAUD),
        (85.0, Disposition.SUSPECTED_FRAUD),
        (100.0, Disposition.SUSPECTED_FRAUD),
    ],
)
def test_disposition_threshold_boundaries(score: float, expected_disposition: Disposition) -> None:
    """Centralised disposition thresholds must cleanly partition the [0, 100] range."""
    thresholds = DispositionThresholds(approve_below=25.0, refer_below=60.0, suspected_fraud_at_or_above=60.0)
    assert thresholds.determine_disposition(score) == expected_disposition


def test_independent_anomalies_add_without_double_counting(clean_claim: Claim) -> None:
    """Triggering multiple independent rules accumulates score up to maximum 100.0."""
    rulebook = load_default_rulebook()

    # Claim with 2 anomalies:
    # 1. Benchmark exaggeration (30.0 points)
    # 2. Prior damage reported (25.0 points)
    # Expected score = 55.0 -> REFER
    two_anomaly_claim = Claim(
        **{
            **clean_claim.__dict__,
            "claimed_amount": 3500.0,
            "benchmark_repair_cost": 1000.0,
            "prior_damage_reported": True,
            "repair_line_items": (RepairLineItem("Side door respray", 3500.0, "PAINT"),),
        }
    )
    dec2 = rulebook.evaluate_claim(two_anomaly_claim)
    assert dec2.risk_score == 55.0
    assert dec2.disposition == Disposition.REFER

    # Claim with 3 anomalies:
    # 1. Benchmark (30.0)
    # 2. Prior damage (25.0)
    # 3. Write-off CAT_A (40.0)
    # Expected score = 95.0 -> SUSPECTED_FRAUD
    three_anomaly_claim = Claim(
        **{
            **two_anomaly_claim.__dict__,
            "vehicle_write_off_status": VehicleWriteOffStatus.CAT_A.value,
        }
    )
    dec3 = rulebook.evaluate_claim(three_anomaly_claim)
    assert dec3.risk_score == 95.0
    assert dec3.disposition == Disposition.SUSPECTED_FRAUD

    # Claim with all 6 anomalies triggering:
    # Expected score clamped to 100.0
    all_anomaly_claim = Claim(
        **{
            **three_anomaly_claim.__dict__,
            "vehicle_registration_status": VehicleRegistrationStatus.UNREGISTERED.value,  # +35
            "policy_inception_date": "2026-04-08",
            "incident_date": "2026-04-10",
            "linked_parties": ("P1", "P2"),  # +35
            "repair_line_items": (
                RepairLineItem(part_or_operation="Unrelated scratch respray", cost=100.0, category="MISC"),
            ),  # +20
        }
    )
    dec_all = rulebook.evaluate_claim(all_anomaly_claim)
    assert dec_all.risk_score == 100.0  # Clamped
    assert dec_all.disposition == Disposition.SUSPECTED_FRAUD


def test_reproducibility_of_claim_decisions(clean_claim: Claim) -> None:
    """Evaluating the same claim twice against the same rulebook produces identical decisions."""
    rulebook = load_default_rulebook()

    test_claim = Claim(
        **{**clean_claim.__dict__, "claimed_amount": 2500.0, "benchmark_repair_cost": 1000.0}
    )

    dec1 = rulebook.evaluate_claim(test_claim, round_number=1)
    dec2 = rulebook.evaluate_claim(test_claim, round_number=1)

    assert dec1.risk_score == dec2.risk_score
    assert dec1.disposition == dec2.disposition
    assert dec1.findings == dec2.findings
    assert dec1.tool_trace == dec2.tool_trace


# ---------------------------------------------------------------------------
# 5. Safety Invariants (TI-01, TI-04)
# ---------------------------------------------------------------------------

def test_rulebook_does_not_import_ground_truth_vault() -> None:
    """
    CRITICAL INVARIANT (TI-01):
    shard.rules.rulebook must never import GroundTruthVault or hidden labels.
    """
    rulebook_file = Path(__file__).resolve().parent.parent / "shard" / "rules" / "rulebook.py"
    tree = ast.parse(rulebook_file.read_text(encoding="utf-8"), filename=str(rulebook_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "vault" not in alias.name.lower(), f"Forbidden import '{alias.name}' in rulebook.py"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert "vault" not in mod_name.lower(), f"Forbidden import from '{mod_name}' in rulebook.py"
            for alias in node.names:
                assert "groundtruth" not in alias.name.lower(), f"Forbidden symbol '{alias.name}' in rulebook.py"


def test_no_eval_or_exec_in_rulebook() -> None:
    """
    CRITICAL INVARIANT (TI-04):
    Rule execution must never invoke eval() or exec().
    """
    rulebook_file = Path(__file__).resolve().parent.parent / "shard" / "rules" / "rulebook.py"
    tree = ast.parse(rulebook_file.read_text(encoding="utf-8"), filename=str(rulebook_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec"), (
                f"Forbidden dynamic call '{node.func.id}' found in rulebook.py:{node.lineno}"
            )


def test_claim_decision_requires_non_empty_findings() -> None:
    """
    CRITICAL INVARIANT (TI-02):
    ClaimDecision must require a non-empty findings tuple. Empty findings raise ValueError.
    """
    with pytest.raises(ValueError, match="ClaimDecision must contain at least one Finding"):
        ClaimDecision(
            claim_id="CLM-EMPTY-01",
            risk_score=10.0,
            disposition=Disposition.APPROVE,
            findings=(),
        )

