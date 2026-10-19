"""
Tests for Shard Bounded Rule-Writer Agent and Validation Gate (Phase 6).

Covers all 16 required test scenarios:
1. Justified diagnosis proposes a relevant supported parameter change.
2. Unsupported diagnoses cannot cause arbitrary rule modifications.
3. Candidate values stay within configured bounds.
4. Per-round parameter changes obey maximum step size.
5. Maximum proposals per round are enforced.
6. A candidate exceeding the false-positive budget is rejected.
7. A candidate breaking the golden regression set is rejected.
8. A structurally invalid candidate is rejected.
9. A rejected candidate cannot mutate the active rulebook.
10. An accepted candidate increments the version exactly once.
11. A candidate with no demonstrated justification is not applied.
12. Identical inputs produce identical proposals and validation outcomes.
13. Audit records accurately report accepted and rejected changes.
14. No eval() or exec() appears anywhere in the rule-writing pipeline.
15. The validation set remains reproducible and is not regenerated per candidate.
16. The Rule-Writer does not receive or import the Critic's raw ground-truth vault.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from shard.agents.rule_writer import RuleWriterAgent, RuleWriterOutput
from shard.core.models import (
    Claim,
    CriticFeedbackSummary,
    Disposition,
    FraudTypology,
    IncidentType,
    RepairLineItem,
    RuleAuditRecord,
    RuleProposal,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)
from shard.rules.rulebook import Rulebook, load_default_rulebook
from shard.rules.validator import (
    RuleValidator,
    build_golden_regression_set,
    get_default_validation_dataset,
)


# ---------------------------------------------------------------------------
# 1. Justified Diagnosis Proposes Relevant Parameter Change
# ---------------------------------------------------------------------------

def test_justified_diagnosis_proposes_relevant_parameter() -> None:
    """Missed EXAGGERATED_REPAIR diagnosis proposes tightening RULE_REPAIR_BENCHMARK."""
    book = load_default_rulebook()
    writer = RuleWriterAgent()

    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.EXAGGERATED_REPAIR.value: 3},
        typology_detection_rates={FraudTypology.EXAGGERATED_REPAIR.value: 0.25},
        total_evaluated=10,
        overall_recall=0.70,
    )

    proposals = writer.generate_proposals(book, feedback, round_number=2)
    assert len(proposals) == 1

    prop = proposals[0]
    assert prop.rule_id == "RULE_REPAIR_BENCHMARK"
    assert prop.parameter_name == "max_markup_ratio"
    assert prop.old_value == 1.35
    assert prop.candidate_value == 1.30  # Tightened by step size 0.05
    assert "EXAGGERATED_REPAIR" in prop.rationale


# ---------------------------------------------------------------------------
# 2. Unsupported Diagnoses Cause No Arbitrary Modifications
# ---------------------------------------------------------------------------

def test_unsupported_diagnoses_cause_no_arbitrary_modifications() -> None:
    """Diagnoses with no supported numerical tuning parameter yield no proposals."""
    book = load_default_rulebook()
    writer = RuleWriterAgent()

    # PRE_EXISTING_DAMAGE has no parameterized threshold in default rulebook
    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.PRE_EXISTING_DAMAGE.value: 2},
        typology_detection_rates={FraudTypology.PRE_EXISTING_DAMAGE.value: 0.0},
        total_evaluated=10,
        overall_recall=0.80,
    )

    proposals = writer.generate_proposals(book, feedback, round_number=2)
    assert len(proposals) == 0


# ---------------------------------------------------------------------------
# 3. Candidate Values Stay Within Configured Bounds
# ---------------------------------------------------------------------------

def test_candidate_values_stay_within_bounds() -> None:
    """Validation Gate A rejects candidate parameter values outside permitted min/max."""
    book = load_default_rulebook()
    validator = RuleValidator()

    # Out of bounds: max_markup_ratio below min 1.10
    out_of_bounds_proposal = RuleProposal(
        proposal_id="PROP-TEST-01",
        round_number=2,
        rule_id="RULE_REPAIR_BENCHMARK",
        template_id="TPL_COST_BENCHMARK",
        parameter_name="max_markup_ratio",
        old_value=1.35,
        candidate_value=1.05,  # Below min bound 1.10
        rationale="Over-aggressive tightening",
        target_typology=FraudTypology.EXAGGERATED_REPAIR.value,
    )

    passed, res, _ = validator.validate(book, out_of_bounds_proposal)
    assert passed is False
    assert res.gate_failed == "GATE_A_CONFIG"
    assert "outside allowed bounds" in res.rejection_reason


# ---------------------------------------------------------------------------
# 4. Per-Round Parameter Changes Obey Maximum Step Size
# ---------------------------------------------------------------------------

def test_per_round_changes_obey_step_size() -> None:
    """Validation Gate A rejects proposals taking a step larger than max_step_size."""
    book = load_default_rulebook()
    validator = RuleValidator()

    # Step size 0.20 exceeds max_step_size 0.05
    large_step_proposal = RuleProposal(
        proposal_id="PROP-TEST-02",
        round_number=2,
        rule_id="RULE_REPAIR_BENCHMARK",
        template_id="TPL_COST_BENCHMARK",
        parameter_name="max_markup_ratio",
        old_value=1.35,
        candidate_value=1.15,  # 0.20 jump
        rationale="Excessive single-round jump",
        target_typology=FraudTypology.EXAGGERATED_REPAIR.value,
    )

    passed, res, _ = validator.validate(book, large_step_proposal)
    assert passed is False
    assert res.gate_failed == "GATE_A_CONFIG"
    assert "exceeds maximum permitted step" in res.rejection_reason


# ---------------------------------------------------------------------------
# 5. Maximum Proposals Per Round Enforced
# ---------------------------------------------------------------------------

def test_max_proposals_per_round_enforced() -> None:
    """Agent enforces max_proposals_per_round constraint regardless of feedback count."""
    book = load_default_rulebook()
    writer = RuleWriterAgent(max_proposals_per_round=1)

    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={
            FraudTypology.EXAGGERATED_REPAIR.value: 5,
            FraudTypology.STAGED_COLLISION.value: 3,
            FraudTypology.OPPORTUNISTIC_INFLATION.value: 2,
        },
        typology_detection_rates={},
        total_evaluated=20,
        overall_recall=0.5,
    )

    proposals = writer.generate_proposals(book, feedback, round_number=2)
    assert len(proposals) == 1
    # Highest miss count (EXAGGERATED_REPAIR) prioritized
    assert proposals[0].target_typology == FraudTypology.EXAGGERATED_REPAIR.value


# ---------------------------------------------------------------------------
# 6. Candidate Exceeding FP Budget Is Rejected
# ---------------------------------------------------------------------------

def test_candidate_exceeding_fp_budget_rejected() -> None:
    """Validation Gate B rejects candidate rules that push false-positive rate over budget."""
    book = load_default_rulebook()

    # Create a validation dataset containing a borderline genuine claim (1.32x markup)
    # Under current rule (max_markup_ratio = 1.35), 1.32x is approved (score 0.0)
    # Under candidate rule (max_markup_ratio = 1.30), 1.32x is flagged (score 30.0 -> REFER)
    borderline_claim = Claim(
        claim_id="VAL-BORDERLINE-01",
        policy_id="POL-V-01",
        claimant_id="CLMT-V-01",
        policy_inception_date="2025-01-01",
        incident_date="2026-03-01",
        filing_date="2026-03-03",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Authentic claim with legitimate dealer parts markup.",
        vehicle_registration="GF19 XKR",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
        prior_damage_reported=False,
        claimed_amount=1320.0,
        benchmark_repair_cost=1000.0,  # 1.32x markup
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(RepairLineItem("Bumper cover OEM", 1320.0, "PARTS"),),
        linked_parties=(),
    )

    # Standard 10% budget, but 1 of 1 claim flagged -> 100% FP rate > 10% budget!
    budget_validator = RuleValidator(fp_budget=0.10, validation_claims=(borderline_claim,))

    proposal = RuleProposal(
        proposal_id="PROP-TEST-03",
        round_number=2,
        rule_id="RULE_REPAIR_BENCHMARK",
        template_id="TPL_COST_BENCHMARK",
        parameter_name="max_markup_ratio",
        old_value=1.35,
        candidate_value=1.30,
        rationale="Overly aggressive threshold tightening",
        target_typology=FraudTypology.EXAGGERATED_REPAIR.value,
    )

    passed, res, new_book = budget_validator.validate(book, proposal)
    assert passed is False
    assert res.gate_failed == "GATE_B_FP_BUDGET"
    assert res.validation_fp_rate == 1.0
    assert "exceeds allowable budget" in res.rejection_reason
    assert new_book is None


# ---------------------------------------------------------------------------
# 7. Candidate Breaking Golden Regression Set Rejected
# ---------------------------------------------------------------------------

def test_candidate_breaking_golden_set_rejected() -> None:
    """Validation Gate C rejects candidates that fail clean approval or fraud detection."""
    book = load_default_rulebook()

    # Custom validator with a golden suite containing an impossible expectation
    broken_golden_suite = (
        (
            Claim(
                claim_id="BROKEN-CLAIM-01",
                policy_id="P1",
                claimant_id="C1",
                policy_inception_date="2025-01-01",
                incident_date="2026-01-01",
                filing_date="2026-01-05",
                incident_type=IncidentType.COLLISION.value,
                incident_description="Test claim",
                vehicle_registration="AB12 CDE",
                vehicle_make="Ford",
                vehicle_model="Focus",
                vehicle_year=2020,
                vehicle_registration_status="VALID",
                vehicle_write_off_status="CLEAN",
                prior_damage_reported=False,
                claimed_amount=100.0,
                benchmark_repair_cost=100.0,
                repair_provider_id="PRV-NORM-101",
                repair_line_items=(),
                linked_parties=(),
            ),
            "FLAGGED",  # Expects flagged, but clean claim will APPROVE -> regression failure!
            "RULE_REPAIR_BENCHMARK",
        ),
    )

    validator = RuleValidator(fp_budget=0.50)
    proposal = RuleProposal(
        proposal_id="PROP-TEST-04",
        round_number=2,
        rule_id="RULE_REPAIR_BENCHMARK",
        template_id="TPL_COST_BENCHMARK",
        parameter_name="max_markup_ratio",
        old_value=1.35,
        candidate_value=1.30,
        rationale="Regression check test",
        target_typology=FraudTypology.EXAGGERATED_REPAIR.value,
    )

    # Monkeypatch golden set in validator
    import shard.rules.validator as val_module
    orig_fn = val_module.build_golden_regression_set
    val_module.build_golden_regression_set = lambda: broken_golden_suite
    try:
        passed, res, _ = validator.validate(book, proposal)
        assert passed is False
        assert res.gate_failed == "GATE_C_REGRESSION"
    finally:
        val_module.build_golden_regression_set = orig_fn


# ---------------------------------------------------------------------------
# 8. Structurally Invalid Candidate Rejected
# ---------------------------------------------------------------------------

def test_structurally_invalid_candidate_rejected() -> None:
    """Validation Gate A rejects unknown rules and non-existent parameters."""
    book = load_default_rulebook()
    validator = RuleValidator()

    invalid_prop = RuleProposal(
        proposal_id="PROP-TEST-05",
        round_number=2,
        rule_id="UNKNOWN_RULE_99",
        template_id="UNKNOWN_TEMPLATE",
        parameter_name="non_existent_param",
        old_value=10,
        candidate_value=20,
        rationale="Invalid structural proposal",
        target_typology="UNKNOWN",
    )

    passed, res, _ = validator.validate(book, invalid_prop)
    assert passed is False
    assert res.gate_failed == "GATE_A_CONFIG"


# ---------------------------------------------------------------------------
# 9. Rejected Candidate Cannot Mutate Active Rulebook
# ---------------------------------------------------------------------------

def test_rejected_candidate_does_not_mutate_active_rulebook() -> None:
    """A rejected proposal leaves active Rulebook version, rules and history untouched."""
    book = load_default_rulebook()
    initial_version = book.version
    initial_markup = book.get_rule("RULE_REPAIR_BENCHMARK").parameters["max_markup_ratio"]
    initial_history_len = len(book.history)

    # Force a rejection with 0.0 FP budget
    strict_writer = RuleWriterAgent(validator=RuleValidator(fp_budget=0.0))
    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.EXAGGERATED_REPAIR.value: 4},
        typology_detection_rates={},
        total_evaluated=10,
        overall_recall=0.6,
    )

    out = strict_writer.propose_and_validate(book, feedback, round_number=2)

    # Even if rejected, verify original book object was NOT mutated
    assert book.version == initial_version
    assert book.get_rule("RULE_REPAIR_BENCHMARK").parameters["max_markup_ratio"] == initial_markup
    assert len(book.history) == initial_history_len


# ---------------------------------------------------------------------------
# 10. Accepted Candidate Increments Version Exactly Once
# ---------------------------------------------------------------------------

def test_accepted_candidate_increments_version_once() -> None:
    """An accepted candidate increments rulebook version and logs acceptance in history."""
    book = load_default_rulebook()
    initial_version = book.version
    writer = RuleWriterAgent(validator=RuleValidator(fp_budget=0.20))

    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.EXAGGERATED_REPAIR.value: 3},
        typology_detection_rates={FraudTypology.EXAGGERATED_REPAIR.value: 0.25},
        total_evaluated=10,
        overall_recall=0.75,
    )

    out = writer.propose_and_validate(book, feedback, round_number=2)

    assert out.accepted is True
    assert out.active_rulebook.version == "1.1.0"
    assert out.active_rulebook.get_rule("RULE_REPAIR_BENCHMARK").parameters["max_markup_ratio"] == 1.30

    # History logged exactly once
    assert len(out.active_rulebook.history) == len(book.history) + 1
    latest_hist = out.active_rulebook.history[-1]
    assert latest_hist["version"] == "1.1.0"
    assert latest_hist["status"] == "ACCEPTED"


# ---------------------------------------------------------------------------
# 11. Candidate With No Demonstrated Justification Not Applied
# ---------------------------------------------------------------------------

def test_no_justification_yields_no_proposal() -> None:
    """Empty feedback or 0 misses generates no proposals and leaves rulebook unchanged."""
    book = load_default_rulebook()
    writer = RuleWriterAgent()

    out = writer.propose_and_validate(book, feedback=None, round_number=1)
    assert out.accepted is False
    assert len(out.proposals) == 0
    assert out.active_rulebook.version == book.version


# ---------------------------------------------------------------------------
# 12. Identical Inputs Produce Identical Proposals and Outcomes
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_identical_outcomes() -> None:
    """Rule-Writer is strictly deterministic across identical rulebook and feedback."""
    book = load_default_rulebook()
    writer1 = RuleWriterAgent(validator=RuleValidator(fp_budget=0.20))
    writer2 = RuleWriterAgent(validator=RuleValidator(fp_budget=0.20))

    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.EXAGGERATED_REPAIR.value: 3},
        typology_detection_rates={FraudTypology.EXAGGERATED_REPAIR.value: 0.25},
        total_evaluated=10,
        overall_recall=0.75,
    )

    out1 = writer1.propose_and_validate(book, feedback, round_number=2)
    out2 = writer2.propose_and_validate(book, feedback, round_number=2)

    assert out1.accepted == out2.accepted
    assert [p.to_dict() for p in out1.proposals] == [p.to_dict() for p in out2.proposals]
    assert out1.active_rulebook.version == out2.active_rulebook.version


# ---------------------------------------------------------------------------
# 13. Audit Records Accurately Report Accepted and Rejected Changes
# ---------------------------------------------------------------------------

def test_audit_records_structure() -> None:
    """Audit records record proposal ID, versions, values and rationale for audit trails."""
    book = load_default_rulebook()
    writer = RuleWriterAgent(validator=RuleValidator(fp_budget=0.20))

    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.EXAGGERATED_REPAIR.value: 3},
        typology_detection_rates={FraudTypology.EXAGGERATED_REPAIR.value: 0.25},
        total_evaluated=10,
        overall_recall=0.75,
    )

    out = writer.propose_and_validate(book, feedback, round_number=2)
    assert len(out.audit_records) == 1

    rec = out.audit_records[0]
    assert isinstance(rec, RuleAuditRecord)
    assert rec.proposal_id.startswith("PROP-R02")
    assert rec.rule_id == "RULE_REPAIR_BENCHMARK"
    assert rec.parameter_name == "max_markup_ratio"
    assert rec.old_value == 1.35
    assert rec.candidate_value == 1.30
    assert rec.status == "ACCEPTED"
    assert rec.rulebook_version_before == "1.0.0"
    assert rec.rulebook_version_after == "1.1.0"


# ---------------------------------------------------------------------------
# 14. No eval() or exec() in Rule-Writer Pipeline
# ---------------------------------------------------------------------------

def test_no_eval_or_exec_in_rule_writer_pipeline() -> None:
    """Static AST check confirms zero eval() or exec() calls in rule_writer or validator."""
    files = [
        Path(__file__).resolve().parent.parent / "shard" / "agents" / "rule_writer.py",
        Path(__file__).resolve().parent.parent / "shard" / "rules" / "validator.py",
    ]

    for fpath in files:
        assert fpath.exists()
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec"), (
                    f"Prohibited {node.func.id}() call found in {fpath.name}"
                )


# ---------------------------------------------------------------------------
# 15. Validation Set Remains Reproducible
# ---------------------------------------------------------------------------

def test_validation_set_reproducibility() -> None:
    """The genuine validation dataset is identical and deterministic across calls."""
    set1 = get_default_validation_dataset()
    set2 = get_default_validation_dataset()

    assert len(set1) == 40
    assert len(set2) == 40
    assert [c.claim_id for c in set1] == [c.claim_id for c in set2]
    assert [c.claimed_amount for c in set1] == [c.claimed_amount for c in set2]


# ---------------------------------------------------------------------------
# 16. Rule-Writer Does Not Import GroundTruthVault
# ---------------------------------------------------------------------------

def test_rule_writer_does_not_import_ground_truth_vault() -> None:
    """Verify via AST that RuleWriterAgent and RuleValidator never import GroundTruthVault."""
    files = [
        Path(__file__).resolve().parent.parent / "shard" / "agents" / "rule_writer.py",
        Path(__file__).resolve().parent.parent / "shard" / "rules" / "validator.py",
    ]

    for fpath in files:
        assert fpath.exists()
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "vault" not in alias.name, f"Prohibited import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "vault" not in node.module, f"Prohibited from-import: {node.module}"
                for alias in node.names:
                    assert alias.name != "GroundTruthVault", f"Prohibited symbol: {alias.name}"
