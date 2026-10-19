"""
Tests for Shard Audit Logging Engine.

Validates:
1. JSONL file creation, formatting, and valid JSON per line.
2. Monotonic sequence numbering (1, 2, 3...) without gaps or duplicates.
3. Ground-truth boundary isolation (TI-01, TI-09): hidden labels (is_fraud, vault_record)
   are strictly stripped from audit event payloads.
4. 100% ClaimDecision audit coverage: every processed claim decision yields an individual event.
5. EXPERIMENT_INIT, ROUND_START, ROUND_END, EXPERIMENT_END event lifecycle logging.
6. CRITIC_REPORT logging with confusion matrix, metrics, and aggregate feedback.
7. RULE_PROPOSAL and RULE_VALIDATION logging with gate audit records.
8. FRAUDSTER_ADAPTATION logging with weight distribution shifts.
9. Canonical dictionary representation without dynamic timestamps for bit-identical reproducibility diffs.
10. Input validation and error handling for malformed events or payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from shard.core.models import (
    ClaimDecision,
    CriticReport,
    Disposition,
    Finding,
    FraudsterAdaptationRecord,
    RuleAuditRecord,
    RuleProposal,
)
from shard.logging.audit import AuditEvent, AuditLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    return tmp_path / "test_audit.jsonl"


@pytest.fixture
def sample_decision() -> ClaimDecision:
    return ClaimDecision(
        claim_id="CLM-2026-TEST-001",
        risk_score=78.5,
        disposition=Disposition.SUSPECTED_FRAUD,
        findings=[
            Finding(
                check_name="repair_cost_benchmark",
                result="FLAGGED: Exceeds benchmark by 45%",
                risk_contribution=35.0,
                detail="Repair estimate: £4500 vs benchmark: £3100",
            ),
            Finding(
                check_name="vehicle_verification",
                result="FLAGGED: Vehicle registered recently",
                risk_contribution=25.0,
                detail="Registration age under 30 days",
            ),
        ],
        tool_trace=[
            {"step": "verify_vehicle", "status": "COMPLETED"},
            {"step": "benchmark_repair_cost", "status": "COMPLETED"},
        ],
        round_number=1,
    )


@pytest.fixture
def sample_critic_report() -> CriticReport:
    return CriticReport(
        total_evaluated=20,
        tp=5,
        fp=1,
        tn=12,
        fn=2,
        precision=0.8333,
        recall=0.7143,
        f1=0.7692,
        fpr=0.0769,
        accuracy=0.85,
        disposition_counts={
            "APPROVE": 12,
            "REFER": 2,
            "SUSPECTED_FRAUD": 6,
        },
        typology_metrics={},
        missed_typology_counts={
            "GHOST_VEHICLE": 2,
        },
        consistency_violations=(),
        round_number=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_audit_logger_in_memory_and_sequence():
    """Verify in-memory event creation and strictly monotonic sequence numbers."""
    logger = AuditLogger()
    assert logger.event_count == 0

    e1 = logger.log_event("TEST_EVENT_A", {"key": "val1"}, round_number=1)
    e2 = logger.log_event("TEST_EVENT_B", {"key": "val2"}, round_number=1)
    e3 = logger.log_event("TEST_EVENT_C", {"key": "val3"}, round_number=2)

    assert logger.event_count == 3
    assert e1.sequence == 1
    assert e2.sequence == 2
    assert e3.sequence == 3
    assert [e.sequence for e in logger.get_events()] == [1, 2, 3]


def test_audit_logger_jsonl_file_output(temp_log_file: Path):
    """Verify that JSON Lines are appended to disk and each line is valid JSON."""
    logger = AuditLogger(log_path=temp_log_file, run_id="RUN-TEST-001")
    assert temp_log_file.exists()

    logger.log_experiment_init({"base_seed": 42, "max_rounds": 3})
    logger.log_round_start(round_number=1, rulebook_version="v1.0.0", seed=42)
    logger.log_round_end(
        round_number=1,
        claims_count=10,
        active_version="v1.0.0",
        continuation_reason="CONTINUE",
    )

    with open(temp_log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 3
    for idx, line in enumerate(lines, start=1):
        data = json.loads(line)
        assert data["sequence"] == idx
        assert data["run_id"] == "RUN-TEST-001"
        assert "event_type" in data
        assert "timestamp" in data
        assert "payload" in data


def test_audit_logger_sanitizes_ground_truth_leakage():
    """Verify that prohibited ground-truth keys (is_fraud, vault_record) are stripped (TI-01)."""
    logger = AuditLogger()
    dirty_payload = {
        "claim_id": "CLM-999",
        "is_fraud": True,
        "is_Fraud": True,
        "ground_truth": {"typology": "GHOST_VEHICLE"},
        "vault_record": "SECRET",
        "raw_gt": "PROHIBITED",
        "safe_feature": "REPAIR_INFLATION",
        "nested_details": {
            "score": 85.0,
            "is_fraud": False,
            "legitimate_nested_info": "VALID",
        },
    }

    event = logger.log_event("TEST_SANITIZE", dirty_payload, round_number=1)
    payload = event.payload

    assert "safe_feature" in payload
    assert payload["safe_feature"] == "REPAIR_INFLATION"
    assert "is_fraud" not in payload
    assert "is_Fraud" not in payload
    assert "ground_truth" not in payload
    assert "vault_record" not in payload
    assert "raw_gt" not in payload
    assert "is_fraud" not in payload["nested_details"]
    assert payload["nested_details"]["legitimate_nested_info"] == "VALID"


def test_audit_logger_claim_decision_coverage(sample_decision: ClaimDecision):
    """Verify logging of ClaimDecision and decision coverage calculation (TI-02, TI-09)."""
    logger = AuditLogger()
    event = logger.log_claim_decision(sample_decision, round_number=1)

    assert event.event_type == "CLAIM_DECISION"
    assert event.payload["claim_id"] == "CLM-2026-TEST-001"
    assert event.payload["risk_score"] == 78.5
    assert event.payload["disposition"] == "SUSPECTED_FRAUD"
    assert event.payload["findings_count"] == 2
    assert len(event.payload["triggered_rules"]) == 2
    assert event.payload["tool_trace_steps"] == 2

    coverage = logger.decision_coverage()
    assert coverage["decision_events_count"] == 1
    assert coverage["total_audit_events"] == 1


def test_audit_logger_critic_report(sample_critic_report: CriticReport):
    """Verify logging of CriticReport with metrics and confusion matrix."""
    logger = AuditLogger()
    event = logger.log_critic_report(sample_critic_report, round_number=1)

    assert event.event_type == "CRITIC_REPORT"
    payload = event.payload
    assert payload["total_evaluated"] == 20
    assert payload["tp"] == 5
    assert payload["fp"] == 1
    assert payload["tn"] == 12
    assert payload["fn"] == 2
    assert payload["precision"] == 0.8333
    assert payload["recall"] == 0.7143
    assert payload["f1"] == 0.7692
    assert payload["fpr"] == 0.0769
    assert payload["accuracy"] == 0.85
    assert payload["missed_typology_counts"] == {"GHOST_VEHICLE": 2}
    assert payload["disposition_counts"]["APPROVE"] == 12


def test_audit_logger_rule_proposal_and_validation():
    """Verify logging of RuleProposal and RuleAuditRecord."""
    logger = AuditLogger()

    proposal = RuleProposal(
        proposal_id="PROP-R02-01",
        round_number=2,
        rule_id="RULE-GHOST-01",
        template_id="ghost_vehicle_threshold",
        parameter_name="min_days_since_reg",
        old_value=7,
        candidate_value=14,
        rationale="Critic reported 2 missed ghost vehicle claims.",
        target_typology="GHOST_VEHICLE",
    )
    p_event = logger.log_rule_proposal(proposal, round_number=2)
    assert p_event.event_type == "RULE_PROPOSAL"
    assert p_event.payload["rule_id"] == "RULE-GHOST-01"
    assert p_event.payload["parameter_name"] == "min_days_since_reg"

    audit_rec = RuleAuditRecord(
        proposal_id="PROP-R02-01",
        round_number=2,
        rule_id="RULE-GHOST-01",
        parameter_name="min_days_since_reg",
        old_value=7,
        candidate_value=14,
        rationale="Critic reported 2 missed ghost vehicle claims.",
        status="ACCEPTED",
        gate_failed=None,
        rejection_reason="",
        validation_fp_rate=0.06,
        rulebook_version_before="v1.0.0",
        rulebook_version_after="v1.1.0",
        timestamp="2026-10-15T00:00:00Z",
    )
    v_event = logger.log_rule_validation(audit_rec, round_number=2)
    assert v_event.event_type == "RULE_VALIDATION"
    assert v_event.payload["status"] == "ACCEPTED"
    assert v_event.payload["rulebook_version_after"] == "v1.1.0"
    assert v_event.payload["validation_fp_rate"] == 0.06


def test_audit_logger_fraudster_adaptation():
    """Verify logging of Fraudster attack weight adaptation."""
    logger = AuditLogger()
    record = FraudsterAdaptationRecord(
        round_number=2,
        base_seed=42,
        derived_seed=42002,
        prior_weights={"REPAIR_INFLATION": 0.4, "GHOST_VEHICLE": 0.2},
        updated_weights={"REPAIR_INFLATION": 0.3, "GHOST_VEHICLE": 0.3},
        feedback_summary={"top_missed_typology": "GHOST_VEHICLE"},
        adapted=True,
    )
    event = logger.log_fraudster_adaptation(record, round_number=2)
    assert event.event_type == "FRAUDSTER_ADAPTATION"
    assert event.payload["adapted"] is True
    assert event.payload["feedback_summary"]["top_missed_typology"] == "GHOST_VEHICLE"
    assert event.payload["updated_weights"]["GHOST_VEHICLE"] == 0.3


def test_audit_logger_experiment_lifecycle_events():
    """Verify EXPERIMENT_INIT and EXPERIMENT_END lifecycle events."""
    logger = AuditLogger()

    init_event = logger.log_experiment_init({"max_rounds": 5, "seed": 42})
    assert init_event.event_type == "EXPERIMENT_INIT"
    assert init_event.round_number == 0

    end_event = logger.log_experiment_end(
        final_stop_reason="TARGET_MET_RECALL_AND_FP_RATE",
        target_reached=True,
        total_rounds=3,
        total_claims=90,
        final_rulebook_version="v1.2.0",
        cumulative_metrics={"recall": 0.88, "precision": 0.90, "f1": 0.89},
    )
    assert end_event.event_type == "EXPERIMENT_END"
    assert end_event.payload["target_reached"] is True
    assert end_event.payload["total_rounds"] == 3
    assert end_event.payload["final_rulebook_version"] == "v1.2.0"


def test_audit_event_canonical_dict_reproducibility():
    """Verify to_canonical_dict strips dynamic timestamp for deterministic comparisons."""
    logger = AuditLogger(run_id="RUN-STATIC")
    e1 = logger.log_event("EVENT_A", {"score": 42}, round_number=1)

    full_dict = e1.to_dict()
    assert "timestamp" in full_dict
    assert full_dict["timestamp"] != ""

    canonical = e1.to_canonical_dict()
    assert "timestamp" not in canonical
    assert canonical["sequence"] == 1
    assert canonical["event_type"] == "EVENT_A"
    assert canonical["run_id"] == "RUN-STATIC"
    assert canonical["payload"] == {"score": 42}


def test_audit_logger_input_validation():
    """Verify that invalid event types or payloads raise appropriate errors."""
    logger = AuditLogger()

    with pytest.raises(ValueError, match="event_type must be a non-empty string"):
        logger.log_event("", {"key": "val"})

    with pytest.raises(TypeError, match="payload must be a dictionary"):
        logger.log_event("VALID_TYPE", "invalid_string_payload")  # type: ignore

    with pytest.raises(TypeError, match="payload must be a dictionary"):
        logger.log_event("VALID_TYPE", [1, 2, 3])  # type: ignore
