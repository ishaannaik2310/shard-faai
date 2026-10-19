"""
Tests for Shard Critic Agent and Evaluation Engine (Phase 5).

Covers all 16 required scenarios:
1. Perfect classification.
2. All predictions approved.
3. All predictions flagged.
4. A known mixture of TP, FP, TN, and FN.
5. Exact precision, recall, F1, and false-positive-rate calculations.
6. Zero-denominator handling without exceptions.
7. Per-typology detection rates and case counts.
8. Detection of missing decisions.
9. Duplicate and unknown claim IDs.
10. Invalid risk scores.
11. Empty findings detection.
12. Incorrect disposition thresholds.
13. Strict ground-truth isolation (AST inspection across investigation modules).
14. Evaluation changes when labels change, while investigation decisions remain unchanged.
15. Reproducible structured reports for identical inputs.
16. Phase 4 planner-rationale regression and evidence-provenance verification.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pytest

from shard.agents.critic import CriticAgent
from shard.agents.investigator import InvestigationPipeline
from shard.agents.planner import PlannerAgent
from shard.agents.researcher import ResearchAgent
from shard.core.models import (
    Claim,
    ClaimDecision,
    ConsistencyViolation,
    CriticReport,
    Disposition,
    Finding,
    FraudTypology,
    GroundTruthRecord,
    IncidentType,
    RepairLineItem,
    TypologyMetrics,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)
from shard.data.vault import GroundTruthVault
from shard.rules.rulebook import Rulebook, load_default_rulebook


# ---------------------------------------------------------------------------
# Fixture Helpers
# ---------------------------------------------------------------------------

def make_dummy_decision(
    claim_id: str,
    risk_score: float,
    disposition: Disposition,
    findings: Optional[Tuple[Finding, ...]] = None,
    tool_trace: Optional[Tuple[dict, ...]] = None,
) -> ClaimDecision:
    """Construct a valid ClaimDecision for test scenarios."""
    if findings is None:
        findings = (
            Finding(
                check_name="RULE_REPAIR_BENCHMARK",
                result="TRIGGERED" if risk_score >= 25.0 else "PASSED",
                risk_contribution=risk_score,
                detail=f"Test check contribution {risk_score}",
            ),
        )
    if tool_trace is None:
        tool_trace = (
            {
                "step": 1,
                "rule_id": "RULE_REPAIR_BENCHMARK",
                "triggered": risk_score >= 25.0,
                "points": risk_score,
            },
        )
    return ClaimDecision(
        claim_id=claim_id,
        risk_score=risk_score,
        disposition=disposition,
        findings=findings,
        tool_trace=tool_trace,
        round_number=1,
    )


# ---------------------------------------------------------------------------
# 1. Perfect Classification
# ---------------------------------------------------------------------------

def test_perfect_classification() -> None:
    """Critic evaluates a run where all fraud is flagged and all genuine approved."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True, fraud_typology=FraudTypology.EXAGGERATED_REPAIR.value),
        GroundTruthRecord("CLM-02", is_fraud=True, fraud_typology=FraudTypology.STAGED_COLLISION.value),
        GroundTruthRecord("CLM-03", is_fraud=False),
        GroundTruthRecord("CLM-04", is_fraud=False),
    ])

    decisions = [
        make_dummy_decision("CLM-01", 75.0, Disposition.SUSPECTED_FRAUD),
        make_dummy_decision("CLM-02", 40.0, Disposition.REFER),
        make_dummy_decision("CLM-03", 10.0, Disposition.APPROVE),
        make_dummy_decision("CLM-04", 15.0, Disposition.APPROVE),
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    assert report.tp == 2
    assert report.fp == 0
    assert report.tn == 2
    assert report.fn == 0
    assert report.total_evaluated == 4
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.fpr == 0.0
    assert report.accuracy == 1.0
    assert report.disposition_counts == {"APPROVE": 2, "REFER": 1, "SUSPECTED_FRAUD": 1}
    assert len(report.consistency_violations) == 0


# ---------------------------------------------------------------------------
# 2. All Predictions Approved
# ---------------------------------------------------------------------------

def test_all_predictions_approved() -> None:
    """Critic evaluates a run where all claims are approved (TP=0, FP=0)."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True),
        GroundTruthRecord("CLM-02", is_fraud=True),
        GroundTruthRecord("CLM-03", is_fraud=False),
        GroundTruthRecord("CLM-04", is_fraud=False),
    ])

    decisions = [
        make_dummy_decision("CLM-01", 10.0, Disposition.APPROVE),
        make_dummy_decision("CLM-02", 15.0, Disposition.APPROVE),
        make_dummy_decision("CLM-03", 5.0, Disposition.APPROVE),
        make_dummy_decision("CLM-04", 0.0, Disposition.APPROVE),
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    assert report.tp == 0
    assert report.fp == 0
    assert report.tn == 2
    assert report.fn == 2
    assert report.total_evaluated == 4
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.fpr == 0.0
    assert report.accuracy == 0.5


# ---------------------------------------------------------------------------
# 3. All Predictions Flagged
# ---------------------------------------------------------------------------

def test_all_predictions_flagged() -> None:
    """Critic evaluates a run where all claims are flagged (TN=0, FN=0)."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True),
        GroundTruthRecord("CLM-02", is_fraud=True),
        GroundTruthRecord("CLM-03", is_fraud=False),
        GroundTruthRecord("CLM-04", is_fraud=False),
    ])

    decisions = [
        make_dummy_decision("CLM-01", 70.0, Disposition.SUSPECTED_FRAUD),
        make_dummy_decision("CLM-02", 50.0, Disposition.REFER),
        make_dummy_decision("CLM-03", 35.0, Disposition.REFER),
        make_dummy_decision("CLM-04", 80.0, Disposition.SUSPECTED_FRAUD),
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    assert report.tp == 2
    assert report.fp == 2
    assert report.tn == 0
    assert report.fn == 0
    assert report.total_evaluated == 4
    assert report.precision == 0.5
    assert report.recall == 1.0
    assert report.f1 == 0.6667
    assert report.fpr == 1.0
    assert report.accuracy == 0.5


# ---------------------------------------------------------------------------
# 4. Known Mixture of TP, FP, TN, and FN
# ---------------------------------------------------------------------------

def test_known_mixture_evaluation() -> None:
    """Critic computes exact expected metrics for a known mixture."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-F1", is_fraud=True),
        GroundTruthRecord("CLM-F2", is_fraud=True),
        GroundTruthRecord("CLM-F3", is_fraud=True),
        GroundTruthRecord("CLM-G1", is_fraud=False),
        GroundTruthRecord("CLM-G2", is_fraud=False),
        GroundTruthRecord("CLM-G3", is_fraud=False),
    ])

    decisions = [
        make_dummy_decision("CLM-F1", 85.0, Disposition.SUSPECTED_FRAUD),  # TP
        make_dummy_decision("CLM-F2", 45.0, Disposition.REFER),            # TP
        make_dummy_decision("CLM-F3", 15.0, Disposition.APPROVE),          # FN
        make_dummy_decision("CLM-G1", 10.0, Disposition.APPROVE),          # TN
        make_dummy_decision("CLM-G2", 20.0, Disposition.APPROVE),          # TN
        make_dummy_decision("CLM-G3", 35.0, Disposition.REFER),            # FP
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    assert report.tp == 2
    assert report.fn == 1
    assert report.tn == 2
    assert report.fp == 1
    assert report.total_evaluated == 6
    assert report.precision == 0.6667
    assert report.recall == 0.6667
    assert report.f1 == 0.6667
    assert report.fpr == 0.3333
    assert report.accuracy == 0.6667


# ---------------------------------------------------------------------------
# 5. Exact Precision, Recall, F1, and False-Positive-Rate Calculations
# ---------------------------------------------------------------------------

def test_exact_metric_calculations() -> None:
    """Verify precision, recall, f1, and fpr on an asymmetric count setup."""
    # TP=3, FP=1, TN=4, FN=2 (Total = 10)
    records = []
    for i in range(1, 4):
        records.append(GroundTruthRecord(f"CLM-TP-{i}", is_fraud=True))
    for i in range(1, 3):
        records.append(GroundTruthRecord(f"CLM-FN-{i}", is_fraud=True))
    for i in range(1, 5):
        records.append(GroundTruthRecord(f"CLM-TN-{i}", is_fraud=False))
    records.append(GroundTruthRecord("CLM-FP-1", is_fraud=False))

    vault = GroundTruthVault(records)

    decisions = [
        make_dummy_decision("CLM-TP-1", 70.0, Disposition.SUSPECTED_FRAUD),
        make_dummy_decision("CLM-TP-2", 65.0, Disposition.SUSPECTED_FRAUD),
        make_dummy_decision("CLM-TP-3", 30.0, Disposition.REFER),
        make_dummy_decision("CLM-FN-1", 10.0, Disposition.APPROVE),
        make_dummy_decision("CLM-FN-2", 15.0, Disposition.APPROVE),
        make_dummy_decision("CLM-TN-1", 5.0, Disposition.APPROVE),
        make_dummy_decision("CLM-TN-2", 10.0, Disposition.APPROVE),
        make_dummy_decision("CLM-TN-3", 12.0, Disposition.APPROVE),
        make_dummy_decision("CLM-TN-4", 18.0, Disposition.APPROVE),
        make_dummy_decision("CLM-FP-1", 50.0, Disposition.REFER),
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    assert report.tp == 3
    assert report.fp == 1
    assert report.tn == 4
    assert report.fn == 2
    assert report.total_evaluated == 10

    # Exact mathematical formulas:
    # Precision = 3 / (3 + 1) = 0.75
    # Recall = 3 / (3 + 2) = 0.6
    # F1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 0.9 / 1.35 = 0.6667
    # FPR = 1 / (1 + 4) = 0.2
    # Accuracy = (3 + 4) / 10 = 0.7
    assert report.precision == 0.75
    assert report.recall == 0.6
    assert report.f1 == 0.6667
    assert report.fpr == 0.2
    assert report.accuracy == 0.7


# ---------------------------------------------------------------------------
# 6. Zero-Denominator Handling
# ---------------------------------------------------------------------------

def test_zero_denominator_handling() -> None:
    """Critic returns 0.0 for all zero-denominator cases without raising exceptions."""
    critic = CriticAgent()

    # Case A: Empty decisions and empty vault
    empty_vault = GroundTruthVault()
    report_empty = critic.evaluate([], empty_vault)
    assert report_empty.total_evaluated == 0
    assert report_empty.precision == 0.0
    assert report_empty.recall == 0.0
    assert report_empty.f1 == 0.0
    assert report_empty.fpr == 0.0
    assert report_empty.accuracy == 0.0

    # Case B: All genuine, all approved -> TP=0, FP=0, FN=0, TN=2
    # Denominators for Precision (TP+FP=0) and Recall (TP+FN=0) are zero
    gen_vault = GroundTruthVault([
        GroundTruthRecord("CLM-G1", is_fraud=False),
        GroundTruthRecord("CLM-G2", is_fraud=False),
    ])
    gen_decisions = [
        make_dummy_decision("CLM-G1", 10.0, Disposition.APPROVE),
        make_dummy_decision("CLM-G2", 15.0, Disposition.APPROVE),
    ]
    report_gen = critic.evaluate(gen_decisions, gen_vault)
    assert report_gen.precision == 0.0
    assert report_gen.recall == 0.0
    assert report_gen.f1 == 0.0
    assert report_gen.fpr == 0.0
    assert report_gen.accuracy == 1.0


# ---------------------------------------------------------------------------
# 7. Per-Typology Detection Rates and Case Counts
# ---------------------------------------------------------------------------

def test_per_typology_detection_rates() -> None:
    """Critic reports detailed breakdown and case counts for every fraud typology."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True, fraud_typology="EXAGGERATED_REPAIR"),
        GroundTruthRecord("CLM-02", is_fraud=True, fraud_typology="EXAGGERATED_REPAIR"),
        GroundTruthRecord("CLM-03", is_fraud=True, fraud_typology="EXAGGERATED_REPAIR"),
        GroundTruthRecord("CLM-04", is_fraud=True, fraud_typology="STAGED_COLLISION"),
        GroundTruthRecord("CLM-05", is_fraud=True, fraud_typology="STAGED_COLLISION"),
        GroundTruthRecord("CLM-06", is_fraud=True, fraud_typology="GHOST_VEHICLE"),
        GroundTruthRecord("CLM-07", is_fraud=False),
    ])

    decisions = [
        make_dummy_decision("CLM-01", 70.0, Disposition.SUSPECTED_FRAUD),  # Caught
        make_dummy_decision("CLM-02", 40.0, Disposition.REFER),            # Caught
        make_dummy_decision("CLM-03", 10.0, Disposition.APPROVE),          # Missed
        make_dummy_decision("CLM-04", 80.0, Disposition.SUSPECTED_FRAUD),  # Caught
        make_dummy_decision("CLM-05", 55.0, Disposition.REFER),            # Caught
        make_dummy_decision("CLM-06", 12.0, Disposition.APPROVE),          # Missed
        make_dummy_decision("CLM-07", 14.0, Disposition.APPROVE),          # Genuine TN
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    # Exaggerated repair: 2 of 3 caught
    exag = report.typology_metrics["EXAGGERATED_REPAIR"]
    assert exag.total_cases == 3
    assert exag.caught_cases == 2
    assert exag.missed_cases == 1
    assert exag.detection_rate == 0.6667

    # Staged collision: 2 of 2 caught (100%)
    staged = report.typology_metrics["STAGED_COLLISION"]
    assert staged.total_cases == 2
    assert staged.caught_cases == 2
    assert staged.missed_cases == 0
    assert staged.detection_rate == 1.0

    # Ghost vehicle: 0 of 1 caught
    ghost = report.typology_metrics["GHOST_VEHICLE"]
    assert ghost.total_cases == 1
    assert ghost.caught_cases == 0
    assert ghost.missed_cases == 1
    assert ghost.detection_rate == 0.0

    # Missed typology counts aggregated for Rule-Writer
    assert report.missed_typology_counts == {
        "EXAGGERATED_REPAIR": 1,
        "GHOST_VEHICLE": 1,
    }


# ---------------------------------------------------------------------------
# 8. Missing Decisions
# ---------------------------------------------------------------------------

def test_missing_decisions_detection() -> None:
    """Critic detects when target claims have no corresponding decision."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True),
        GroundTruthRecord("CLM-02", is_fraud=False),
        GroundTruthRecord("CLM-03", is_fraud=True),
    ])

    # Only provide decision for CLM-01
    decisions = [
        make_dummy_decision("CLM-01", 80.0, Disposition.SUSPECTED_FRAUD),
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    missing_violations = [
        v for v in report.consistency_violations if v.violation_type == "MISSING_DECISION"
    ]
    missing_ids = {v.claim_id for v in missing_violations}
    assert missing_ids == {"CLM-02", "CLM-03"}


# ---------------------------------------------------------------------------
# 9. Duplicate and Unknown Claim IDs
# ---------------------------------------------------------------------------

def test_duplicate_and_unknown_claim_ids() -> None:
    """Critic flags duplicate decisions and unknown claim IDs without crashing."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True),
        GroundTruthRecord("CLM-02", is_fraud=False),
    ])

    decisions = [
        make_dummy_decision("CLM-01", 80.0, Disposition.SUSPECTED_FRAUD),
        make_dummy_decision("CLM-01", 80.0, Disposition.SUSPECTED_FRAUD),  # Duplicate
        make_dummy_decision("CLM-UNKNOWN-99", 50.0, Disposition.REFER),    # Unknown ID
        make_dummy_decision("CLM-02", 10.0, Disposition.APPROVE),
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    types = [v.violation_type for v in report.consistency_violations]
    assert "DUPLICATE_DECISION" in types
    assert "UNKNOWN_CLAIM_ID" in types

    # Duplicate should not double-count in metrics
    assert report.total_evaluated == 2
    assert report.tp == 1
    assert report.tn == 1


# ---------------------------------------------------------------------------
# 10. Invalid Risk Scores
# ---------------------------------------------------------------------------

def test_invalid_risk_scores_detected() -> None:
    """Critic detects out-of-range risk scores."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True),
    ])

    # Construct decision with bypassed post_init to simulate out-of-bounds score
    decision = object.__new__(ClaimDecision)
    object.__setattr__(decision, "claim_id", "CLM-01")
    object.__setattr__(decision, "risk_score", 150.0)  # Invalid score > 100.0
    object.__setattr__(decision, "disposition", Disposition.SUSPECTED_FRAUD)
    object.__setattr__(
        decision,
        "findings",
        (Finding("RULE_REPAIR_BENCHMARK", "FLAGGED", 150.0, "Score overflow"),),
    )
    object.__setattr__(decision, "tool_trace", ({"step": 1, "rule_id": "RULE_REPAIR_BENCHMARK"},))
    object.__setattr__(decision, "round_number", 1)

    critic = CriticAgent()
    report = critic.evaluate([decision], vault)

    score_violations = [
        v for v in report.consistency_violations if v.violation_type == "INVALID_RISK_SCORE"
    ]
    assert len(score_violations) == 1
    assert "150.0" in score_violations[0].explanation


# ---------------------------------------------------------------------------
# 11. Empty Findings Detection
# ---------------------------------------------------------------------------

def test_empty_findings_detected() -> None:
    """Critic detects decisions that violate TI-02 by lacking findings."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True),
    ])

    decision = object.__new__(ClaimDecision)
    object.__setattr__(decision, "claim_id", "CLM-01")
    object.__setattr__(decision, "risk_score", 80.0)
    object.__setattr__(decision, "disposition", Disposition.SUSPECTED_FRAUD)
    object.__setattr__(decision, "findings", ())  # Empty findings
    object.__setattr__(decision, "tool_trace", ())
    object.__setattr__(decision, "round_number", 1)

    critic = CriticAgent()
    report = critic.evaluate([decision], vault)

    finding_violations = [
        v for v in report.consistency_violations if v.violation_type == "EMPTY_FINDINGS"
    ]
    assert len(finding_violations) == 1
    assert "TI-02" in finding_violations[0].explanation


# ---------------------------------------------------------------------------
# 12. Incorrect Disposition Thresholds
# ---------------------------------------------------------------------------

def test_incorrect_disposition_thresholds() -> None:
    """Critic detects when disposition contradicts the active Rulebook thresholds."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True),
        GroundTruthRecord("CLM-02", is_fraud=False),
    ])

    # Rulebook thresholds: < 25.0 APPROVE, 25.0-59.9 REFER, >= 60.0 SUSPECTED_FRAUD
    decisions = [
        # Score 15.0 should be APPROVE, but is marked SUSPECTED_FRAUD
        make_dummy_decision("CLM-01", 15.0, Disposition.SUSPECTED_FRAUD),
        # Score 80.0 should be SUSPECTED_FRAUD, but is marked APPROVE
        make_dummy_decision("CLM-02", 80.0, Disposition.APPROVE),
    ]

    critic = CriticAgent()
    report = critic.evaluate(decisions, vault)

    thresh_violations = [
        v for v in report.consistency_violations if v.violation_type == "DISPOSITION_THRESHOLD_MISMATCH"
    ]
    assert len(thresh_violations) == 2


# ---------------------------------------------------------------------------
# 13. Ground-Truth Isolation Across Boundaries (AST Inspection)
# ---------------------------------------------------------------------------

def test_ground_truth_isolation_ast_check() -> None:
    """
    Verify that investigation-facing modules (Planner, Researcher, Investigator,
    Rulebook, Tools) NEVER import GroundTruthVault or shard.data.vault.
    Only Critic and test/orchestrator modules are permitted.
    """
    repo_root = Path(__file__).resolve().parent.parent

    prohibited_modules = [
        repo_root / "shard" / "agents" / "planner.py",
        repo_root / "shard" / "agents" / "researcher.py",
        repo_root / "shard" / "agents" / "investigator.py",
        repo_root / "shard" / "rules" / "rulebook.py",
        repo_root / "shard" / "rules" / "tools.py",
        repo_root / "shard" / "rules" / "templates.py",
        repo_root / "shard" / "core" / "models.py",
    ]

    for mod_path in prohibited_modules:
        assert mod_path.exists(), f"Module {mod_path} does not exist"
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "vault" not in alias.name, (
                        f"Prohibited import in {mod_path.name}: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "vault" not in node.module, (
                        f"Prohibited from-import in {mod_path.name}: {node.module}"
                    )
                for alias in node.names:
                    assert "GroundTruthVault" != alias.name, (
                        f"Prohibited symbol import in {mod_path.name}: {alias.name}"
                    )


# ---------------------------------------------------------------------------
# 14. Evaluation Changes With Labels While Decisions Remain Unchanged
# ---------------------------------------------------------------------------

def test_label_mutation_alters_evaluation_not_investigation() -> None:
    """
    Changing hidden ground-truth labels changes the Critic's evaluation metrics,
    but does NOT alter the upstream investigation decision or evidence.
    """
    clean_claim = Claim(
        claim_id="CLM-ISO-01",
        policy_id="POL-ISO-01",
        claimant_id="CLMT-ISO-01",
        policy_inception_date="2025-01-01",
        incident_date="2026-01-01",
        filing_date="2026-01-05",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Standard car park collision.",
        vehicle_registration="GF19 XKR",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
        prior_damage_reported=False,
        claimed_amount=1100.0,
        benchmark_repair_cost=1050.0,
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(RepairLineItem("Bumper cover", 1100.0, "PARTS"),),
        linked_parties=(),
    )

    # Run genuine investigation pipeline
    pipeline = InvestigationPipeline()
    res = pipeline.process(clean_claim)
    assert res.decision.disposition == Disposition.APPROVE
    assert res.decision.risk_score == 0.0

    critic = CriticAgent()

    # Scenario A: Vault says claim is genuine -> True Negative (Accuracy = 1.0)
    vault_genuine = GroundTruthVault([GroundTruthRecord("CLM-ISO-01", is_fraud=False)])
    report_a = critic.evaluate([res.decision], vault_genuine)
    assert report_a.tn == 1
    assert report_a.fn == 0
    assert report_a.accuracy == 1.0

    # Scenario B: Vault says claim was secretly fraud -> False Negative (Accuracy = 0.0)
    vault_fraud = GroundTruthVault([GroundTruthRecord("CLM-ISO-01", is_fraud=True, fraud_typology="PRE_EXISTING_DAMAGE")])
    report_b = critic.evaluate([res.decision], vault_fraud)
    assert report_b.tn == 0
    assert report_b.fn == 1
    assert report_b.accuracy == 0.0

    # Critical invariant: The investigation decision is completely untouched
    assert res.decision.risk_score == 0.0
    assert res.decision.disposition == Disposition.APPROVE


# ---------------------------------------------------------------------------
# 15. Reproducible Structured Reports for Identical Inputs
# ---------------------------------------------------------------------------

def test_critic_report_reproducibility() -> None:
    """Critic produces identical structured dictionary serialisations for identical inputs."""
    vault = GroundTruthVault([
        GroundTruthRecord("CLM-01", is_fraud=True, fraud_typology="EXAGGERATED_REPAIR"),
        GroundTruthRecord("CLM-02", is_fraud=False),
    ])

    decisions = [
        make_dummy_decision("CLM-01", 75.0, Disposition.SUSPECTED_FRAUD),
        make_dummy_decision("CLM-02", 15.0, Disposition.APPROVE),
    ]

    critic = CriticAgent()
    report1 = critic.evaluate(decisions, vault, round_number=1)
    report2 = critic.evaluate(decisions, vault, round_number=1)

    d1 = report1.to_dict()
    d2 = report2.to_dict()

    # Disregard generated_at timestamps for reproducibility check
    d1.pop("generated_at")
    d2.pop("generated_at")

    assert d1 == d2
    assert d1["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert d1["metrics"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 16. Regression & Provenance Fix Verification
# ---------------------------------------------------------------------------

def test_phase4_regressions_verified() -> None:
    """
    Verify both Phase 4 fixes in the evaluation context:
    1. Planner rationale does not misstate tenure.
    2. Evidence items are accurately attributed to claim-submitted evidence.
    """
    planner = PlannerAgent()
    researcher = ResearchAgent()

    test_claim = Claim(
        claim_id="CLM-P4-FIX",
        policy_id="POL-P4",
        claimant_id="CLMT-P4",
        policy_inception_date="2025-01-01",
        incident_date="2025-11-20",  # 323 days tenure
        filing_date="2025-11-22",
        incident_type="COLLISION",
        incident_description="Front bumper scuff.",
        vehicle_registration="WM21 YYY",
        vehicle_make="BMW",
        vehicle_model="3 Series",
        vehicle_year=2021,
        vehicle_registration_status="VALID",
        vehicle_write_off_status="CLEAN",
        prior_damage_reported=False,
        claimed_amount=1350.0,
        benchmark_repair_cost=1000.0,
        repair_provider_id="PRV-FLAG-902",  # Triggered provider
        repair_line_items=(RepairLineItem("Bumper respray", 1350.0, "PAINT"),),
        linked_parties=(),
    )

    plan = planner.plan(test_claim)
    reason = plan.task_reasons["RULE_STAGED_COLLISION_INDICATORS"]

    # Triggered by flagged provider, NOT rapid inception
    assert "flagged provider 'PRV-FLAG-902'" in reason
    assert "rapid inception" not in reason.lower()

    # Evidence bundle accurately attributes claim-submitted sources
    bundle = researcher.gather_evidence(test_claim, plan)
    assert bundle.get("repair_cost_benchmark").source == "CLAIM_SUBMITTED_ESTIMATICS_BENCHMARK"
    assert bundle.get("vehicle_registration").source == "CLAIM_SUBMITTED_VEHICLE_RECORD"
    assert bundle.get("vehicle_write_off_status").source == "CLAIM_SUBMITTED_WRITE_OFF_RECORD"
