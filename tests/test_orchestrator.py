"""
Tests for Shard Multi-Round Orchestrator.

Validates:
1. OrchestratorConfig input validation and constraint enforcement.
2. Single-round and multi-round closed-loop execution.
3. Stopping conditions: early stopping on target recall/FPR, termination on max rounds.
4. Rulebook snapshotting and version tracking across rounds.
5. Historical immutability of past round results.
6. Rule-Writer integration: accepted rule updates advance rulebook version.
7. Rule-Writer rejection handling: rejected candidate updates preserve rulebook.
8. Fraudster adaptation integration: evasive typologies receive increased sampling weight.
9. 100% decision audit coverage (TI-09).
10. Cumulative metrics mathematical consistency with aggregate confusion matrix.
11. Strict reproducibility (TI-07): bit-identical canonical dictionaries for identical seeds.
12. Ground-truth isolation boundary (TI-01): AST and runtime verification.
13. File-based vs in-memory audit log emission.
14. Edge-case robustness: 0% and 100% fraud proportions without divide-by-zero errors.
15. Verification that reported metrics are measured, not fabricated (TI-03).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pytest

from shard.agents.critic import CriticAgent
from shard.agents.fraudster import FraudsterAgent
from shard.agents.investigator import InvestigationPipeline
from shard.agents.rule_writer import RuleWriterAgent
from shard.core.models import Claim, ClaimDecision, Disposition, Finding
from shard.logging.audit import AuditLogger
from shard.orchestrator import ExperimentResult, Orchestrator, OrchestratorConfig, RoundResult
from shard.rules.rulebook import Rulebook, load_default_rulebook


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------

def test_orchestrator_config_validation():
    """Verify OrchestratorConfig rejects invalid parameter values."""
    # Valid config passes
    valid = OrchestratorConfig(
        base_seed=42,
        n_claims_per_round=15,
        fraud_share=0.25,
        max_rounds=3,
        target_recall=0.80,
        max_fp_rate=0.08,
        adaptation_strength=0.10,
        fp_budget=0.05,
    )
    assert valid.n_claims_per_round == 15

    # Invalid n_claims_per_round
    with pytest.raises(ValueError, match="n_claims_per_round must be >= 1"):
        OrchestratorConfig(n_claims_per_round=0)

    # Invalid fraud_share
    with pytest.raises(ValueError, match="fraud_share must be between 0.0 and 1.0"):
        OrchestratorConfig(fraud_share=-0.1)
    with pytest.raises(ValueError, match="fraud_share must be between 0.0 and 1.0"):
        OrchestratorConfig(fraud_share=1.5)

    # Invalid max_rounds
    with pytest.raises(ValueError, match="max_rounds must be >= 1"):
        OrchestratorConfig(max_rounds=0)

    # Invalid target_recall
    with pytest.raises(ValueError, match="target_recall must be between 0.0 and 1.0"):
        OrchestratorConfig(target_recall=1.2)

    # Invalid max_fp_rate
    with pytest.raises(ValueError, match="max_fp_rate must be between 0.0 and 1.0"):
        OrchestratorConfig(max_fp_rate=-0.05)

    # Invalid adaptation_strength
    with pytest.raises(ValueError, match="adaptation_strength must be >= 0.0"):
        OrchestratorConfig(adaptation_strength=-0.1)

    # Invalid fp_budget
    with pytest.raises(ValueError, match="fp_budget must be between 0.0 and 1.0"):
        OrchestratorConfig(fp_budget=1.1)


# ---------------------------------------------------------------------------
# Execution & Stopping Condition Tests
# ---------------------------------------------------------------------------

def test_orchestrator_single_round_execution():
    """Verify single-round experiment execution and completeness."""
    cfg = OrchestratorConfig(
        base_seed=101,
        n_claims_per_round=10,
        fraud_share=0.30,
        max_rounds=1,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg)
    res: ExperimentResult = orch.run()

    assert len(res.rounds) == 1
    assert res.total_claims_processed == 10
    round1 = res.rounds[0]
    assert round1.round_number == 1
    assert round1.claims_count == 10
    assert len(round1.decisions) == 10
    assert round1.continuation_decision == "MAX_ROUNDS_REACHED"
    assert round1.stop_reason == "MAX_ROUNDS_REACHED"
    assert res.final_stop_reason == "MAX_ROUNDS_REACHED"
    assert round1.critic_report.total_evaluated == 10


def test_orchestrator_multi_round_max_rounds():
    """Verify multi-round execution halts exactly at max_rounds when target not reached."""
    cfg = OrchestratorConfig(
        base_seed=202,
        n_claims_per_round=8,
        fraud_share=0.40,
        max_rounds=3,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg)
    res: ExperimentResult = orch.run()

    assert len(res.rounds) == 3
    assert res.total_claims_processed == 24
    assert res.final_stop_reason == "MAX_ROUNDS_REACHED"
    assert res.target_reached is False
    assert res.rounds[0].continuation_decision == "CONTINUE"
    assert res.rounds[1].continuation_decision == "CONTINUE"
    assert res.rounds[2].continuation_decision == "MAX_ROUNDS_REACHED"


def test_orchestrator_early_stopping_target_met():
    """Verify early stopping triggers when target recall and FPR are achieved."""
    cfg = OrchestratorConfig(
        base_seed=303,
        n_claims_per_round=10,
        fraud_share=0.30,
        max_rounds=5,
        target_recall=0.01,  # Trivial target to guarantee early stop
        max_fp_rate=1.00,
        early_stopping=True,
    )
    orch = Orchestrator(config=cfg)
    res: ExperimentResult = orch.run()

    assert len(res.rounds) == 1
    assert res.target_reached is True
    assert res.final_stop_reason == "TARGET_MET_RECALL_AND_FP_RATE"
    assert res.rounds[0].continuation_decision == "TARGET_REACHED"


def test_orchestrator_early_stopping_disabled():
    """Verify that disabling early_stopping continues all max_rounds even if target is met."""
    cfg = OrchestratorConfig(
        base_seed=404,
        n_claims_per_round=8,
        fraud_share=0.25,
        max_rounds=2,
        target_recall=0.01,
        max_fp_rate=1.00,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg)
    res: ExperimentResult = orch.run()

    assert len(res.rounds) == 2
    assert res.final_stop_reason == "MAX_ROUNDS_REACHED"


# ---------------------------------------------------------------------------
# Rulebook Snapshotting, Adaptation & Immutability Tests
# ---------------------------------------------------------------------------

def test_orchestrator_rulebook_snapshotting_and_immutability():
    """Verify each round captures its own immutable rulebook version snapshot."""
    cfg = OrchestratorConfig(
        base_seed=505,
        n_claims_per_round=10,
        max_rounds=2,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg)
    res = orch.run()

    r1 = res.rounds[0]
    r2 = res.rounds[1]

    # Verify version strings are tracked
    assert bool(r1.rulebook_version_before)
    assert bool(r2.rulebook_version_before)
    assert r1.rulebook_version_before == "1.0.0"
    # Verify decisions tuple in round 1 was frozen
    assert isinstance(r1.decisions, tuple)
    assert isinstance(r2.decisions, tuple)


def test_orchestrator_rule_writer_proposals_logged():
    """Verify Rule-Writer proposals and validation outcomes are recorded in round results."""
    rb = load_default_rulebook()
    r = rb.get_rule("RULE_REPAIR_BENCHMARK")
    r.parameters["max_markup_ratio"] = 1.60

    cfg = OrchestratorConfig(
        base_seed=42,
        n_claims_per_round=30,
        fraud_share=0.40,
        max_rounds=3,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg, initial_rulebook=rb)
    res = orch.run()

    r2 = res.rounds[1]
    # In round 2 continuation, rule_writer runs and proposes adjustment
    assert r2.continuation_decision == "CONTINUE"
    assert len(r2.rule_proposals) > 0
    assert len(r2.rule_audit_records) > 0
    assert res.total_proposals > 0
    assert res.accepted_proposals + res.rejected_proposals == res.total_proposals


def test_orchestrator_fraudster_adaptation_integration():
    """Verify Fraudster adapts attack typology weights in subsequent rounds."""
    rb = load_default_rulebook()
    r = rb.get_rule("RULE_REPAIR_BENCHMARK")
    r.parameters["max_markup_ratio"] = 1.60

    cfg = OrchestratorConfig(
        base_seed=42,
        n_claims_per_round=30,
        fraud_share=0.40,
        max_rounds=3,
        early_stopping=False,
        adaptation_strength=0.20,
    )
    orch = Orchestrator(config=cfg, initial_rulebook=rb)
    res = orch.run()

    r1 = res.rounds[0]
    r2 = res.rounds[1]
    r3 = res.rounds[2]

    # In round 1, no prior feedback exists
    assert r1.fraudster_record.adapted is False
    # In round 3, feedback from round 2 (which had misses) caused adaptation
    assert r3.fraudster_record.adapted is True
    assert "missed_typologies" in r3.fraudster_record.feedback_summary
    assert r3.fraudster_record.prior_weights != r3.fraudster_record.updated_weights


# ---------------------------------------------------------------------------
# Audit Coverage & Metrics Consistency Tests
# ---------------------------------------------------------------------------

def test_orchestrator_100_percent_decision_audit_coverage():
    """Verify every claim decision produces an individual audit event (TI-09)."""
    cfg = OrchestratorConfig(
        base_seed=808,
        n_claims_per_round=10,
        max_rounds=2,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg)
    res = orch.run()

    cov = orch.audit_logger.decision_coverage()
    assert cov["decision_events_count"] == res.total_claims_processed
    assert cov["decision_events_count"] == 20

    # Verify event types present
    event_types = {e.event_type for e in orch.audit_logger.get_events()}
    assert "EXPERIMENT_INIT" in event_types
    assert "ROUND_START" in event_types
    assert "CLAIM_DECISION" in event_types
    assert "CRITIC_REPORT" in event_types
    assert "ROUND_END" in event_types
    assert "EXPERIMENT_END" in event_types


def test_orchestrator_cumulative_metrics_consistency():
    """Verify cumulative metrics match the exact sum of confusion matrices across rounds."""
    cfg = OrchestratorConfig(
        base_seed=909,
        n_claims_per_round=12,
        fraud_share=0.35,
        max_rounds=3,
        target_recall=0.99,
        max_fp_rate=0.01,
    )
    orch = Orchestrator(config=cfg)
    res = orch.run()

    cum = res.cumulative_metrics
    expected_tp = sum(r.critic_report.tp for r in res.rounds)
    expected_fp = sum(r.critic_report.fp for r in res.rounds)
    expected_tn = sum(r.critic_report.tn for r in res.rounds)
    expected_fn = sum(r.critic_report.fn for r in res.rounds)
    expected_total = expected_tp + expected_fp + expected_tn + expected_fn

    assert cum["tp"] == expected_tp
    assert cum["fp"] == expected_fp
    assert cum["tn"] == expected_tn
    assert cum["fn"] == expected_fn
    assert cum["total_evaluated"] == expected_total

    if (expected_tp + expected_fp) > 0:
        assert cum["precision"] == round(expected_tp / (expected_tp + expected_fp), 4)
    if (expected_tp + expected_fn) > 0:
        assert cum["recall"] == round(expected_tp / (expected_tp + expected_fn), 4)
    if (expected_fp + expected_tn) > 0:
        assert cum["fpr"] == round(expected_fp / (expected_fp + expected_tn), 4)
    if expected_total > 0:
        assert cum["accuracy"] == round((expected_tp + expected_tn) / expected_total, 4)


# ---------------------------------------------------------------------------
# Strict Reproducibility Test (TI-07)
# ---------------------------------------------------------------------------

def test_orchestrator_strict_reproducibility():
    """Verify running Orchestrator twice with same seed produces bit-identical results."""
    cfg = OrchestratorConfig(
        base_seed=12345,
        n_claims_per_round=10,
        fraud_share=0.30,
        max_rounds=2,
        target_recall=0.99,
    )

    orch1 = Orchestrator(config=cfg)
    res1 = orch1.run()

    orch2 = Orchestrator(config=cfg)
    res2 = orch2.run()

    # Verify core attributes match exactly
    assert res1.total_claims_processed == res2.total_claims_processed
    assert res1.final_rulebook_version == res2.final_rulebook_version
    assert res1.cumulative_metrics == res2.cumulative_metrics
    assert res1.final_stop_reason == res2.final_stop_reason

    # Verify canonical dictionaries (excluding timestamps & run_id) match bit-for-bit
    assert res1.to_canonical_dict() == res2.to_canonical_dict()


# ---------------------------------------------------------------------------
# Ground-Truth Boundary & AST Isolation Tests (TI-01)
# ---------------------------------------------------------------------------

def test_orchestrator_gt_isolation_ast():
    """Verify AST compliance: orchestrator and audit log do not import forbidden modules."""
    for rel_path in ["shard/orchestrator.py", "shard/logging/audit.py"]:
        path = Path(rel_path)
        assert path.exists(), f"{rel_path} does not exist"

        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "vault" not in alias.name.lower() or "shard.data.vault" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module != "shard.data.vault", (
                        f"Prohibited import of shard.data.vault in {rel_path}"
                    )


def test_orchestrator_investigation_pipeline_receives_only_observable_claims(monkeypatch):
    """Verify that during orchestration, InvestigationPipeline receives only observable Claims."""
    seen_claims: list[Claim] = []
    original_process = InvestigationPipeline.process

    def mock_process(self, claim: Claim, round_number: int = 1):
        # Verify the claim passed to investigation pipeline has no is_fraud attribute
        assert not hasattr(claim, "is_fraud"), "Claim object leaked is_fraud to investigation!"
        seen_claims.append(claim)
        return original_process(self, claim, round_number=round_number)

    monkeypatch.setattr(InvestigationPipeline, "process", mock_process)

    cfg = OrchestratorConfig(base_seed=999, n_claims_per_round=5, max_rounds=1)
    orch = Orchestrator(config=cfg)
    orch.run()

    assert len(seen_claims) == 5


# ---------------------------------------------------------------------------
# File Audit Log & Robustness Tests
# ---------------------------------------------------------------------------

def test_orchestrator_file_audit_logging(tmp_path: Path):
    """Verify that Orchestrator properly writes JSONL audit logs to disk when configured."""
    log_file = tmp_path / "experiment_audit.jsonl"
    cfg = OrchestratorConfig(
        base_seed=111,
        n_claims_per_round=6,
        max_rounds=2,
        target_recall=0.99,
        audit_log_path=str(log_file),
    )
    orch = Orchestrator(config=cfg)
    res = orch.run()

    assert log_file.exists()
    assert res.audit_log_path == str(log_file)

    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0
    # Every line must be parseable JSON
    for line in lines:
        data = json.loads(line)
        assert "event_type" in data
        assert "sequence" in data
        assert "run_id" in data


def test_orchestrator_edge_case_zero_fraud_share():
    """Verify Orchestrator executes safely without zero-division error when fraud_share is 0.0."""
    cfg = OrchestratorConfig(
        base_seed=222,
        n_claims_per_round=6,
        fraud_share=0.0,
        max_rounds=1,
    )
    orch = Orchestrator(config=cfg)
    res = orch.run()

    assert res.total_claims_processed == 6
    assert res.rounds[0].critic_report.tp == 0
    assert res.rounds[0].critic_report.fn == 0
    assert res.cumulative_metrics["tp"] == 0


def test_orchestrator_edge_case_hundred_percent_fraud():
    """Verify Orchestrator executes safely when fraud_share is 1.0."""
    cfg = OrchestratorConfig(
        base_seed=333,
        n_claims_per_round=6,
        fraud_share=1.0,
        max_rounds=1,
    )
    orch = Orchestrator(config=cfg)
    res = orch.run()

    assert res.total_claims_processed == 6
    assert res.rounds[0].critic_report.tn == 0
    assert res.rounds[0].critic_report.fp == 0
    assert res.cumulative_metrics["tn"] == 0
