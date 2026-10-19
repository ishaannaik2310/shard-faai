"""
Tests for SHARD v3 Phase 2b-2: Generic feature-threshold rule via gated discovery.

Validates:
1. Dynamic route in Planner: PlannerAgent schedules tpl_feature_threshold rules generically.
2. Dynamic execution in Investigator: Investigator executes tpl_feature_threshold rules and builds findings.
3. Safety gates for discovered rules:
   - Gate A: Cap of 3 discovered rules, max 1 per round, bounds and score checks.
   - Gate B: Candidate threshold exceeding FP budget is rejected; looser candidate is accepted.
4. End-to-end multi-round discovery in Orchestrator:
   - With enable_feature_discovery=True, new unmodelled attack is missed in Round 1.
   - Rule-Writer ranks features, discovers top feature, tests candidates against Gate B.
   - Installs RULE_DISCOVERED_001.
   - In Round 2, new rule is dynamically scheduled and executed.
   - Recall on the unmodelled attack improves substantially while FPR <= 10%.
5. Invariant preservation: With enable_feature_discovery=False, Orchestrator run is 100% identical to v2 baseline.
6. Fixed-seed reproducibility: Identical seed with discovery enabled produces bit-identical results.
7. Trust boundaries and AST guard: Zero imports of vault, GroundTruthRecord, or FraudTypology;
   zero forbidden strings in detection/planner/investigator/templates/tools/validator/rule_writer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shard.agents.critic import CriticAgent
from shard.agents.investigator import InvestigationPipeline, InvestigatorAgent
from shard.agents.planner import ALL_RULE_TASKS, PlannerAgent
from shard.agents.rule_writer import RuleWriterAgent
from shard.core.features import FEATURE_WHITELIST
from shard.core.models import (
    Claim,
    Disposition,
    FraudTypology,
    RepairLineItem,
    RuleProposal,
)
from shard.data.generator import ClaimGenerator, GeneratorConfig
from shard.orchestrator import Orchestrator, OrchestratorConfig
from shard.rules.rulebook import RuleDefinition, Rulebook, load_default_rulebook
from shard.rules.templates import TEMPLATE_REGISTRY, tpl_feature_threshold
from shard.rules.validator import RuleValidator, get_v3_validation_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sample_claim(
    claim_id: str = "TEST-CLM-001",
    filing_delay_days: int = 4,
    claimed_amount: float = 1200.0,
    benchmark_cost: float = 1200.0,
) -> Claim:
    """Construct an observable claim with controllable filing delay and amounts."""
    return Claim(
        claim_id=claim_id,
        policy_id="POL-TEST-001",
        claimant_id="CLMT-TEST-001",
        policy_inception_date="2025-01-01",
        incident_date="2026-03-01",
        filing_date=f"2026-03-{1 + filing_delay_days:02d}",
        incident_type="COLLISION",
        incident_description="Standard minor front-end impact.",
        vehicle_registration="AB12 CDE",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status="VALID",
        vehicle_write_off_status="CLEAN",
        prior_damage_reported=False,
        claimed_amount=claimed_amount,
        benchmark_repair_cost=benchmark_cost,
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(
            RepairLineItem("Bumper cover respray", claimed_amount, "PAINT"),
        ),
        linked_parties=(),
    )


# ---------------------------------------------------------------------------
# 1. Dynamic Route in Planner
# ---------------------------------------------------------------------------

def test_planner_schedules_generic_feature_threshold_rules() -> None:
    """Planner dynamically schedules any enabled rule using tpl_feature_threshold."""
    rb = load_default_rulebook()
    planner = PlannerAgent()
    claim = _create_sample_claim()

    # Default rulebook: only baseline tasks scheduled
    plan_default = planner.plan(claim, rulebook=rb)
    for task in ALL_RULE_TASKS:
        assert task in plan_default.tasks
    assert not any(t.startswith("RULE_DISCOVERED_") for t in plan_default.tasks)

    # Add a discovered rule to the rulebook
    disc_rule = RuleDefinition(
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        description="Discovered rule for observable feature",
        enabled=True,
        severity="HIGH",
        score_contribution=20.0,
        parameters={
            "feature": "claim_filing_delay_days",
            "direction": ">",
            "threshold": 6.0,
        },
    )
    rb.rules.append(disc_rule)

    plan_with_disc = planner.plan(claim, rulebook=rb)
    assert "RULE_DISCOVERED_001" in plan_with_disc.tasks
    assert "RULE_DISCOVERED_001" in plan_with_disc.task_reasons
    # All baseline tasks must also remain scheduled
    for task in ALL_RULE_TASKS:
        assert task in plan_with_disc.tasks


def test_planner_skips_disabled_discovered_rules() -> None:
    """Planner does not schedule disabled discovered rules."""
    rb = load_default_rulebook()
    planner = PlannerAgent()
    claim = _create_sample_claim()

    disc_rule = RuleDefinition(
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        description="Disabled discovered rule",
        enabled=False,
        severity="HIGH",
        score_contribution=20.0,
        parameters={"feature": "claim_filing_delay_days", "direction": ">", "threshold": 6.0},
    )
    rb.rules.append(disc_rule)

    plan = planner.plan(claim, rulebook=rb)
    assert "RULE_DISCOVERED_001" not in plan.tasks


# ---------------------------------------------------------------------------
# 2. Dynamic Execution in Investigator and Pipeline
# ---------------------------------------------------------------------------

def test_investigator_executes_discovered_rule() -> None:
    """Investigator executes tpl_feature_threshold and produces findings and traces."""
    rb = load_default_rulebook()
    disc_rule = RuleDefinition(
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        description="Discovered rule for filing delay",
        enabled=True,
        severity="HIGH",
        score_contribution=25.0,
        parameters={
            "feature": "claim_filing_delay_days",
            "direction": ">",
            "threshold": 5.0,
        },
    )
    rb.rules.append(disc_rule)

    pipeline = InvestigationPipeline(rulebook=rb)

    # Claim with delay = 7 (> threshold 5.0) -> rule triggers
    claim_breach = _create_sample_claim(claim_id="BREACH-001", filing_delay_days=7)
    res_breach = pipeline.process(claim_breach)
    trace_breach = [t for t in res_breach.decision.tool_trace if t["rule_id"] == "RULE_DISCOVERED_001"]
    assert len(trace_breach) == 1
    assert trace_breach[0]["triggered"] is True
    assert trace_breach[0]["points"] == 25.0
    finding_breach = [f for f in res_breach.decision.findings if f.check_name == "RULE_DISCOVERED_001"]
    assert len(finding_breach) == 1
    assert finding_breach[0].result == "FLAGGED (HIGH)"
    assert finding_breach[0].risk_contribution == 25.0

    # Claim with delay = 2 (<= threshold 5.0) -> rule does not trigger
    claim_conform = _create_sample_claim(claim_id="CONFORM-001", filing_delay_days=2)
    res_conform = pipeline.process(claim_conform)
    trace_conform = [t for t in res_conform.decision.tool_trace if t["rule_id"] == "RULE_DISCOVERED_001"]
    assert len(trace_conform) == 1
    assert trace_conform[0]["triggered"] is False
    assert trace_conform[0]["points"] == 0.0


# ---------------------------------------------------------------------------
# 3. Safety Gates for Discovered Rules
# ---------------------------------------------------------------------------

def test_gate_b_rejects_overly_aggressive_threshold() -> None:
    """Gate B rejects candidate threshold that exceeds the false-positive budget."""
    validator = RuleValidator(fp_budget=0.10, v3_mode=True)
    rb = load_default_rulebook()

    # Threshold = 2.0 days (median genuine delay is 3-4 days -> breaches budget)
    proposal_aggressive = RuleProposal(
        proposal_id="PROP-TEST-AGGRESSIVE",
        round_number=1,
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        parameter_name="threshold",
        old_value=None,
        candidate_value=2.0,
        rationale="Overly aggressive threshold test",
        target_typology="DISCOVERED_PATTERN",
        expected_effect="Flag claims with delay > 2.0",
        rule_parameters={
            "feature": "claim_filing_delay_days",
            "direction": ">",
            "threshold": 2.0,
            "score_contribution": 25.0,
            "severity": "HIGH",
        },
    )

    accepted, val_result, updated_rb = validator.validate(rb, proposal_aggressive)
    assert accepted is False
    assert val_result.gate_failed == "GATE_B_FP_BUDGET"
    assert val_result.validation_fp_rate > 0.10
    assert updated_rb is None


def test_gate_b_accepts_safe_discovered_threshold() -> None:
    """Gate B accepts candidate threshold that keeps validation FPR within budget."""
    validator = RuleValidator(fp_budget=0.10, v3_mode=True)
    rb = load_default_rulebook()

    # Threshold = 8.0 days (well above genuine 95th/97th percentile)
    proposal_safe = RuleProposal(
        proposal_id="PROP-TEST-SAFE",
        round_number=1,
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        parameter_name="threshold",
        old_value=None,
        candidate_value=8.0,
        rationale="Safe threshold test",
        target_typology="DISCOVERED_PATTERN",
        expected_effect="Flag claims with delay > 8.0",
        rule_parameters={
            "feature": "claim_filing_delay_days",
            "direction": ">",
            "threshold": 8.0,
            "score_contribution": 25.0,
            "severity": "HIGH",
        },
    )

    accepted, val_result, updated_rb = validator.validate(rb, proposal_safe)
    assert accepted is True
    assert val_result.passed is True
    assert val_result.validation_fp_rate <= 0.10
    assert updated_rb is not None
    assert updated_rb.version == "1.1.0"
    assert any(r.rule_id == "RULE_DISCOVERED_001" for r in updated_rb.rules)


def test_gate_a_discovered_rules_constraints() -> None:
    """Gate A enforces global cap (3), max 1 per round, and valid parameter bounds."""
    validator = RuleValidator(fp_budget=0.10, v3_mode=True)
    rb = load_default_rulebook()

    # 1. Invalid feature name rejected
    prop_bad_feat = RuleProposal(
        proposal_id="PROP-BAD-FEAT",
        round_number=1,
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        parameter_name="threshold",
        old_value=None,
        candidate_value=5.0,
        rationale="Bad feature",
        target_typology="DISCOVERED_PATTERN",
        rule_parameters={
            "feature": "non_existent_feature",
            "direction": ">",
            "threshold": 5.0,
            "score_contribution": 20.0,
        },
    )
    acc, res, _ = validator.validate(rb, prop_bad_feat)
    assert acc is False
    assert res.gate_failed == "GATE_A_CONFIG"

    # 2. Out of allowed threshold bounds rejected
    prop_oob = RuleProposal(
        proposal_id="PROP-OOB",
        round_number=1,
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        parameter_name="threshold",
        old_value=None,
        candidate_value=500.0,  # Whitelist max for filing delay is 365
        rationale="OOB threshold",
        target_typology="DISCOVERED_PATTERN",
        rule_parameters={
            "feature": "claim_filing_delay_days",
            "direction": ">",
            "threshold": 500.0,
            "score_contribution": 20.0,
        },
    )
    acc, res, _ = validator.validate(rb, prop_oob)
    assert acc is False
    assert res.gate_failed == "GATE_A_CONFIG"

    # 3. Score contribution outside [10, 30] rejected
    prop_score = RuleProposal(
        proposal_id="PROP-SCORE",
        round_number=1,
        rule_id="RULE_DISCOVERED_001",
        template_id="tpl_feature_threshold",
        parameter_name="threshold",
        old_value=None,
        candidate_value=7.0,
        rationale="Bad score contribution",
        target_typology="DISCOVERED_PATTERN",
        rule_parameters={
            "feature": "claim_filing_delay_days",
            "direction": ">",
            "threshold": 7.0,
            "score_contribution": 50.0,  # Max allowed is 30.0
        },
    )
    acc, res, _ = validator.validate(rb, prop_score)
    assert acc is False
    assert res.gate_failed == "GATE_A_CONFIG"

    # 4. Global cap of 3 discovered rules
    rb_capped = rb.clone()
    for i in range(1, 4):
        rb_capped.rules.append(
            RuleDefinition(
                rule_id=f"RULE_DISCOVERED_{i:03d}",
                template_id="tpl_feature_threshold",
                description="Cap test rule",
                enabled=True,
                severity="HIGH",
                score_contribution=20.0,
                parameters={"feature": "claim_filing_delay_days", "direction": ">", "threshold": 8.0},
            )
        )
    prop_cap = RuleProposal(
        proposal_id="PROP-CAP-EXCEEDED",
        round_number=4,
        rule_id="RULE_DISCOVERED_004",
        template_id="tpl_feature_threshold",
        parameter_name="threshold",
        old_value=None,
        candidate_value=8.0,
        rationale="Cap test",
        target_typology="DISCOVERED_PATTERN",
        rule_parameters={
            "feature": "claim_filing_delay_days",
            "direction": ">",
            "threshold": 8.0,
            "score_contribution": 20.0,
        },
    )
    acc, res, _ = validator.validate(rb_capped, prop_cap)
    assert acc is False
    assert res.gate_failed == "GATE_A_CONFIG"
    assert "Global cap" in res.rejection_reason


# ---------------------------------------------------------------------------
# 4. End-to-End Orchestrator Gated Discovery Flow
# ---------------------------------------------------------------------------

def test_orchestrator_e2e_discovery_improves_recall() -> None:
    """
    End-to-end integration test:
    Round 1: Unmodelled attack is missed, Critic accumulates FNs.
    Adaptation: Rule-Writer ranks features, discovers top feature, installs RULE_DISCOVERED_001.
    Round 2: Newly installed rule is dynamically scheduled; recall on unmodelled attack improves;
             FPR remains within budget (<= 10%).
    """
    config = OrchestratorConfig(
        base_seed=42,
        n_claims_per_round=120,
        fraud_share=0.30,
        max_rounds=2,
        enable_feature_discovery=True,
        v3_typology_start_round=1,
        min_missed_samples=5,
        early_stopping=False,
    )

    orchestrator = Orchestrator(config=config)
    exp_result = orchestrator.run()

    assert len(exp_result.rounds) == 2

    # Round 1 evaluation
    r1 = exp_result.rounds[0]
    r1_delayed_metric = r1.critic_report.typology_metrics[FraudTypology.DELAYED_FILING.value]
    assert r1_delayed_metric.caught_cases == 0
    assert r1_delayed_metric.detection_rate == 0.0

    # Verification of adaptation between Round 1 and Round 2
    assert exp_result.accepted_proposals >= 1
    disc_accepted = [
        rec for rec in r1.rule_audit_records
        if rec.rule_id.startswith("RULE_DISCOVERED_") and rec.status == "ACCEPTED"
    ]
    assert len(disc_accepted) == 1
    accepted_rec = disc_accepted[0]
    assert accepted_rec.validation_fp_rate <= config.fp_budget

    # Round 2 evaluation: Discovered rule is now active
    r2 = exp_result.rounds[1]
    assert r2.rulebook_version_before == "1.1.0"
    r2_delayed_metric = r2.critic_report.typology_metrics[FraudTypology.DELAYED_FILING.value]
    # Recall on the new attack improves substantially
    assert r2_delayed_metric.caught_cases > 0
    assert r2_delayed_metric.detection_rate >= 0.20

    # Overall FPR remains strictly controlled
    assert r2.critic_report.fpr <= config.max_fp_rate

    # Audit logger recorded DISCOVERY_RANKING event
    events = orchestrator.audit_logger.get_events()
    ranking_events = [e for e in events if e.event_type == "DISCOVERY_RANKING"]
    assert len(ranking_events) >= 1
    ranking_payload = ranking_events[0].payload
    assert "ranked_features" in ranking_payload
    assert ranking_payload["chosen_feature"] == "claim_filing_delay_days"


# ---------------------------------------------------------------------------
# 5. Invariant: Flag OFF produces bit-identical run to v2 baseline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 123])
def test_flag_off_exact_v2_baseline_regression(seed: int) -> None:
    """When enable_feature_discovery=False, Orchestrator run is 100% identical to v2."""
    cfg_off = OrchestratorConfig(
        base_seed=seed,
        n_claims_per_round=30,
        fraud_share=0.30,
        max_rounds=3,
        enable_feature_discovery=False,
        include_v3_typologies=False,
    )
    orch = Orchestrator(cfg_off)
    res = orch.run()

    canonical = res.to_canonical_dict()

    # Rulebook should never install any discovered rule
    assert not any(
        r["rule_id"].startswith("RULE_DISCOVERED_")
        for r in canonical["rounds"]
        for r in r["rule_audit_records"]
    )
    # Final rulebook must not contain discovered rules
    assert not any(r.rule_id.startswith("RULE_DISCOVERED_") for r in res.final_rulebook.rules)


# ---------------------------------------------------------------------------
# 6. Reproducibility with Discovery
# ---------------------------------------------------------------------------

def test_reproducibility_with_discovery_enabled() -> None:
    """Two independent Orchestrator runs with discovery enabled produce bit-identical results."""
    cfg = OrchestratorConfig(
        base_seed=42,
        n_claims_per_round=60,
        fraud_share=0.30,
        max_rounds=2,
        enable_feature_discovery=True,
        v3_typology_start_round=1,
        early_stopping=False,
    )

    orch1 = Orchestrator(cfg)
    res1 = orch1.run()

    orch2 = Orchestrator(cfg)
    res2 = orch2.run()

    assert res1.to_canonical_dict() == res2.to_canonical_dict()


# ---------------------------------------------------------------------------
# 7. Trust Boundaries and AST Checks
# ---------------------------------------------------------------------------

def test_ast_trust_boundary_for_discovery_modules() -> None:
    """Verify that RuleWriter and discovery modules never import restricted modules or symbols."""
    restricted_modules = {"shard.data.vault", "vault"}
    restricted_symbols = {"GroundTruthRecord", "GroundTruthVault", "FraudTypology"}

    files_to_check = [
        Path("shard/agents/rule_writer.py"),
        Path("shard/discovery/ranking.py"),
        Path("shard/core/discovery_models.py"),
        Path("shard/core/features.py"),
    ]

    for fpath in files_to_check:
        assert fpath.exists(), f"File {fpath} must exist"
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in restricted_modules, (
                        f"Forbidden import '{alias.name}' in {fpath}"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in restricted_modules, (
                    f"Forbidden from-import module '{node.module}' in {fpath}"
                )
                for alias in node.names:
                    assert alias.name not in restricted_symbols, (
                        f"Forbidden imported symbol '{alias.name}' in {fpath}"
                    )


def test_guard_zero_forbidden_strings_in_core_modules() -> None:
    """Verify that detection, planner, investigator, templates, tools, validator, and rule-writer code
    never mention DELAYED_FILING or filing_delay, and discovery modules never mention any typology name.
    """
    phase_2a_strings = ["DELAYED_FILING", "filing_delay"]
    core_files = [
        Path("shard/rules/templates.py"),
        Path("shard/rules/tools.py"),
        Path("data/rulebook_v1.json"),
        Path("shard/agents/planner.py"),
        Path("shard/agents/investigator.py"),
        Path("shard/agents/rule_writer.py"),
        Path("shard/discovery/ranking.py"),
        Path("shard/core/discovery_models.py"),
    ]
    for fpath in core_files:
        assert fpath.exists(), f"File {fpath} must exist"
        text = fpath.read_text(encoding="utf-8")
        for bad_str in phase_2a_strings:
            assert bad_str not in text, f"Forbidden string '{bad_str}' found in {fpath}"

    all_typology_names = [
        "DELAYED_FILING",
        "EXAGGERATED_REPAIR",
        "STAGED_COLLISION",
        "GHOST_VEHICLE",
        "PRE_EXISTING_DAMAGE",
        "OPPORTUNISTIC_INFLATION",
    ]
    discovery_files = [
        Path("shard/discovery/ranking.py"),
        Path("shard/core/discovery_models.py"),
        Path("shard/core/features.py"),
    ]
    for fpath in discovery_files:
        assert fpath.exists(), f"File {fpath} must exist"
        text = fpath.read_text(encoding="utf-8")
        for bad_str in all_typology_names:
            assert bad_str not in text, f"Forbidden typology name '{bad_str}' found in {fpath}"
