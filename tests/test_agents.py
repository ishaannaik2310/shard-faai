"""
Tests for Shard Investigation Agents and Pipeline (Planner, Researcher, Investigator).

Covers:
1. Planner agent behaviour, contextual task ordering, determinism, and rationale.
2. Research agent behaviour, evidence structuring, provenance, and ID validation.
3. Investigator agent behaviour, planned tool execution, score calculation, and trace.
4. End-to-end multi-agent pipeline integration.
5. Strict ground-truth isolation (AST inspection) across all agent modules.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Set

import pytest

from shard.agents.investigator import (
    InvestigationPipeline,
    InvestigatorAgent,
    PipelineResult,
)
from shard.agents.planner import ALL_RULE_TASKS, PlannerAgent
from shard.agents.researcher import ResearchAgent
from shard.core.models import (
    Claim,
    ClaimDecision,
    Disposition,
    EvidenceBundle,
    EvidenceItem,
    IncidentType,
    InvestigationPlan,
    RepairLineItem,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)
from shard.data.generator import GeneratorConfig, generate_claims
from shard.rules.rulebook import load_default_rulebook


@pytest.fixture
def sample_clean_claim() -> Claim:
    """Standard observable clean motor insurance claim."""
    return Claim(
        claim_id="CLM-AGENT-001",
        policy_id="POL-100001",
        claimant_id="CLMT-20001",
        policy_inception_date="2025-06-01",
        incident_date="2026-03-20",
        filing_date="2026-03-23",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Low-speed rear contact in queue at roundabout.",
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
        repair_line_items=(
            RepairLineItem("Bumper skin replacement", 600.0, "PARTS"),
            RepairLineItem("Refinish and paint", 350.0, "PAINT"),
            RepairLineItem("Fitting labour 1.5 hrs", 150.0, "LABOUR"),
        ),
        linked_parties=(),
    )


# ---------------------------------------------------------------------------
# 1. Planner Agent Tests
# ---------------------------------------------------------------------------

def test_planner_produces_valid_plan(sample_clean_claim: Claim) -> None:
    """Planner produces a valid typed InvestigationPlan with all applicable checks."""
    planner = PlannerAgent()
    plan = planner.plan(sample_clean_claim)

    assert isinstance(plan, InvestigationPlan)
    assert plan.claim_id == sample_clean_claim.claim_id
    assert len(plan.tasks) == 6
    assert set(plan.tasks) == set(ALL_RULE_TASKS)
    assert len(plan.rationale) > 0
    assert len(plan.task_reasons) == 6
    for rule_id in ALL_RULE_TASKS:
        assert rule_id in plan.task_reasons


def test_planner_determinism(sample_clean_claim: Claim) -> None:
    """Calling planner multiple times on the same claim produces identical plans."""
    planner = PlannerAgent()
    plan1 = planner.plan(sample_clean_claim)
    plan2 = planner.plan(sample_clean_claim)

    assert plan1 == plan2
    assert plan1.tasks == plan2.tasks
    assert plan1.rationale == plan2.rationale


def test_planner_contextual_prioritization() -> None:
    """Planner contextually orders tasks based on observable indicators."""
    planner = PlannerAgent()

    # 1. Vehicle anomaly claim should prioritize vehicle checks first
    ghost_claim = Claim(
        claim_id="CLM-ANOM-01",
        policy_id="POL-01",
        claimant_id="C-01",
        policy_inception_date="2025-01-01",
        incident_date="2026-02-01",
        filing_date="2026-02-05",
        incident_type="COLLISION",
        incident_description="Damage reported.",
        vehicle_registration="XX99 YYY",
        vehicle_make="BMW",
        vehicle_model="3 Series",
        vehicle_year=2018,
        vehicle_registration_status=VehicleRegistrationStatus.UNREGISTERED.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CAT_A.value,
        prior_damage_reported=False,
        claimed_amount=2000.0,
        benchmark_repair_cost=1800.0,
        repair_provider_id="PRV-NORM-101",
    )
    plan_ghost = planner.plan(ghost_claim)
    assert plan_ghost.tasks[0] == "RULE_VEHICLE_REGISTRATION"
    assert plan_ghost.tasks[1] == "RULE_WRITE_OFF_STATUS"
    assert "IDENTITY_FIRST" in plan_ghost.rationale

    # 2. Staged collision indicators should prioritize collision checks first
    staged_claim = Claim(
        claim_id="CLM-ANOM-02",
        policy_id="POL-02",
        claimant_id="C-02",
        policy_inception_date="2026-03-01",
        incident_date="2026-03-05",  # 4 days tenure!
        filing_date="2026-03-06",
        incident_type="COLLISION",
        incident_description="Roundabout collision with multiple passengers.",
        vehicle_registration="GF19 XKR",
        vehicle_make="Audi",
        vehicle_model="A4",
        vehicle_year=2020,
        vehicle_registration_status="VALID",
        vehicle_write_off_status="CLEAN",
        prior_damage_reported=False,
        claimed_amount=1500.0,
        benchmark_repair_cost=1400.0,
        repair_provider_id="PRV-FLAG-901",
        linked_parties=("CLMT-9901", "CLMT-9902"),
    )
    plan_staged = planner.plan(staged_claim)
    assert plan_staged.tasks[0] == "RULE_STAGED_COLLISION_INDICATORS"
    assert "NETWORK_FIRST" in plan_staged.rationale


# ---------------------------------------------------------------------------
# 2. Research Agent Tests
# ---------------------------------------------------------------------------

def test_researcher_gathers_evidence_bundle(sample_clean_claim: Claim) -> None:
    """Researcher produces a complete typed EvidenceBundle with provenance for every item."""
    planner = PlannerAgent()
    researcher = ResearchAgent()

    plan = planner.plan(sample_clean_claim)
    bundle = researcher.gather_evidence(sample_clean_claim, plan)

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.claim_id == sample_clean_claim.claim_id
    assert bundle.is_complete is True
    assert len(bundle.items) >= 6

    # Verify provenance on items
    for key, item in bundle.items.items():
        assert isinstance(item, EvidenceItem)
        assert item.key == key
        assert len(item.source) > 0, f"Missing source provenance on {key}"
        assert item.is_available is True

    # Check specific synthetic evidence records (observably derived from claim dossier)
    assert bundle.get("repair_cost_benchmark").source == "CLAIM_SUBMITTED_ESTIMATICS_BENCHMARK"
    assert bundle.get("vehicle_registration").source == "CLAIM_SUBMITTED_VEHICLE_RECORD"
    assert bundle.get("vehicle_write_off_status").source == "CLAIM_SUBMITTED_WRITE_OFF_RECORD"
    assert bundle.get("repair_line_items").source == "CLAIM_SUBMISSION_GARAGE_QUOTE"
    assert bundle.get("prior_damage_history").source == "CLAIM_SUBMISSION_PRIOR_DAMAGE"
    assert bundle.get("policy_and_network_linkages").source == "CLAIM_SUBMITTED_POLICY_AND_NETWORK"


def test_planner_staged_collision_rationale_regression() -> None:
    """
    Regression test for Phase 4 Issue A:
    A claim with 336 days of tenure and a flagged provider must NOT report
    'rapid inception tenure' in its task rationale.
    """
    planner = PlannerAgent()
    long_tenure_flagged_claim = Claim(
        claim_id="CLM-REGRESSION-01",
        policy_id="POL-LONG-01",
        claimant_id="CLMT-01",
        policy_inception_date="2025-01-01",
        incident_date="2025-12-03",  # 336 days tenure
        filing_date="2025-12-05",
        incident_type="COLLISION",
        incident_description="Rear impact in car park.",
        vehicle_registration="AB12 CDE",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2020,
        vehicle_registration_status="VALID",
        vehicle_write_off_status="CLEAN",
        prior_damage_reported=False,
        claimed_amount=1200.0,
        benchmark_repair_cost=1100.0,
        repair_provider_id="PRV-FLAG-901",  # Trigger
        repair_line_items=(),
        linked_parties=(),
    )
    plan = planner.plan(long_tenure_flagged_claim)
    reason = plan.task_reasons["RULE_STAGED_COLLISION_INDICATORS"]

    # Must accurately reflect the flagged provider trigger
    assert "flagged provider 'PRV-FLAG-901'" in reason
    # Must NOT mischaracterise 336 days of tenure as rapid inception
    assert "rapid inception" not in reason.lower()
    assert "336d" not in reason

    # Conversely, a genuine rapid inception claim (<= 10 days) MUST report rapid inception
    rapid_claim = Claim(
        **{
            **long_tenure_flagged_claim.__dict__,
            "claim_id": "CLM-REGRESSION-02",
            "incident_date": "2025-01-06",  # 5 days tenure
            "repair_provider_id": "PRV-NORM-101",  # Standard provider
        }
    )
    plan_rapid = planner.plan(rapid_claim)
    reason_rapid = plan_rapid.task_reasons["RULE_STAGED_COLLISION_INDICATORS"]
    assert "rapid inception tenure (5d <= 10d)" in reason_rapid
    assert "flagged provider" not in reason_rapid


def test_researcher_validates_identifier_match(sample_clean_claim: Claim) -> None:
    """Researcher rejects mismatched claim and plan identifiers."""
    planner = PlannerAgent()
    researcher = ResearchAgent()

    plan = planner.plan(sample_clean_claim)
    mismatched_claim = Claim(**{**sample_clean_claim.__dict__, "claim_id": "CLM-DIFFERENT-999"})

    with pytest.raises(ValueError, match="Identifier mismatch"):
        researcher.gather_evidence(mismatched_claim, plan)


def test_researcher_handles_missing_information_explicitly(sample_clean_claim: Claim) -> None:
    """Researcher explicitly records when evidence is unavailable without crashing."""
    planner = PlannerAgent()
    researcher = ResearchAgent()

    # Claim with missing vehicle plate and zero benchmark
    incomplete_claim = Claim(
        **{
            **sample_clean_claim.__dict__,
            "vehicle_registration": "   ",
            "benchmark_repair_cost": 0.0,
        }
    )
    plan = planner.plan(incomplete_claim)
    bundle = researcher.gather_evidence(incomplete_claim, plan)

    assert bundle.is_complete is False
    assert bundle.get("vehicle_registration").is_available is False
    assert bundle.get("repair_cost_benchmark").is_available is False


# ---------------------------------------------------------------------------
# 3. Investigator Agent Tests
# ---------------------------------------------------------------------------

def test_investigator_executes_planned_tools(sample_clean_claim: Claim) -> None:
    """Investigator executes checks strictly from plan and scores via Rulebook."""
    planner = PlannerAgent()
    researcher = ResearchAgent()
    investigator = InvestigatorAgent()
    rulebook = load_default_rulebook()

    plan = planner.plan(sample_clean_claim)
    evidence = researcher.gather_evidence(sample_clean_claim, plan)

    decision = investigator.investigate(
        claim=sample_clean_claim,
        plan=plan,
        evidence=evidence,
        rulebook=rulebook,
    )

    assert isinstance(decision, ClaimDecision)
    assert decision.claim_id == sample_clean_claim.claim_id
    assert decision.risk_score == 0.0
    assert decision.disposition == Disposition.APPROVE
    assert len(decision.findings) == 6
    assert len(decision.tool_trace) == 6

    # Verify execution order matches the plan exactly
    trace_rule_ids = [t["rule_id"] for t in decision.tool_trace]
    assert trace_rule_ids == list(plan.tasks)


def test_investigator_rejects_mismatched_inputs(sample_clean_claim: Claim) -> None:
    """Investigator rejects any execution where claim, plan, or evidence IDs diverge."""
    planner = PlannerAgent()
    researcher = ResearchAgent()
    investigator = InvestigatorAgent()

    plan = planner.plan(sample_clean_claim)
    evidence = researcher.gather_evidence(sample_clean_claim, plan)

    diff_claim = Claim(**{**sample_clean_claim.__dict__, "claim_id": "CLM-DIFF"})
    with pytest.raises(ValueError, match="Identifier mismatch"):
        investigator.investigate(diff_claim, plan, evidence)


def test_investigator_handles_unknown_rule_in_plan(sample_clean_claim: Claim) -> None:
    """Investigator raises ValueError if a plan requests a rule missing from the rulebook."""
    researcher = ResearchAgent()
    investigator = InvestigatorAgent()

    bad_plan = InvestigationPlan(
        claim_id=sample_clean_claim.claim_id,
        tasks=("RULE_DOES_NOT_EXIST",),
        rationale="Bad plan",
    )
    evidence = researcher.gather_evidence(sample_clean_claim, bad_plan)

    with pytest.raises(ValueError, match="does not exist in Rulebook"):
        investigator.investigate(sample_clean_claim, bad_plan, evidence)


# ---------------------------------------------------------------------------
# 4. Multi-Agent Pipeline End-to-End Integration
# ---------------------------------------------------------------------------

def test_pipeline_collaboration_on_generated_claims() -> None:
    """
    Test complete 3-agent collaboration on seeded synthetic claims:
    Claim -> Planner -> Plan -> Researcher -> Evidence -> Investigator -> Decision
    """
    batch = generate_claims(GeneratorConfig(seed=42, n_claims=5, fraud_share=0.4))
    pipeline = InvestigationPipeline()

    assert len(batch.claims) == 5

    for claim in batch.claims:
        result = pipeline.process(claim)

        assert isinstance(result, PipelineResult)
        assert result.execution_order == ("Planner", "Researcher", "Investigator")

        # 1. Identifier agreement
        assert (
            result.claim.claim_id
            == result.plan.claim_id
            == result.evidence.claim_id
            == result.decision.claim_id
        )

        # 2. Plan integrity
        assert len(result.plan.tasks) == 6
        assert len(result.plan.rationale) > 0

        # 3. Evidence integrity
        assert len(result.evidence.items) >= 6

        # 4. Decision integrity
        assert 0.0 <= result.decision.risk_score <= 100.0
        assert result.decision.disposition in (
            Disposition.APPROVE,
            Disposition.REFER,
            Disposition.SUSPECTED_FRAUD,
        )
        assert len(result.decision.findings) > 0
        assert len(result.decision.tool_trace) == 6


# ---------------------------------------------------------------------------
# 5. Strict Ground-Truth Isolation (TI-01) Across Agent Modules
# ---------------------------------------------------------------------------

def test_agents_do_not_import_ground_truth_vault() -> None:
    """
    CRITICAL INVARIANT (TI-01):
    Planner, Researcher, and Investigator modules must never import GroundTruthVault,
    vault modules, or hidden evaluation metadata.
    """
    agents_dir = Path(__file__).resolve().parent.parent / "shard" / "agents"
    agent_files = [
        agents_dir / "planner.py",
        agents_dir / "researcher.py",
        agents_dir / "investigator.py",
        agents_dir / "__init__.py",
    ]

    for py_file in agent_files:
        assert py_file.exists(), f"Agent file {py_file} does not exist"
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "vault" not in alias.name.lower(), (
                        f"Prohibited import '{alias.name}' in {py_file.name}:{node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert "vault" not in mod_name.lower(), (
                    f"Prohibited import from '{mod_name}' in {py_file.name}:{node.lineno}"
                )
                for alias in node.names:
                    assert "groundtruth" not in alias.name.lower(), (
                        f"Prohibited import symbol '{alias.name}' in {py_file.name}:{node.lineno}"
                    )
