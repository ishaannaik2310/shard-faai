"""
Tests for Shard Fraudster Agent (Phase 6).

Covers all 11 required test scenarios:
1. Initial weights follow documented defaults.
2. Missed typologies increase in relative sampling weight.
3. Typologies with lower miss rates do not receive an unjustified increase.
4. Weights are bounded and normalized (sum to 1.0).
5. Identical inputs produce identical weights and claim batches (seed determinism).
6. Different seeds produce different synthetic batches.
7. Missing previous-round feedback is handled explicitly.
8. Zero-count typologies do not cause division errors.
9. Unknown typology identifiers in feedback are handled safely.
10. The agent does not access previous-round vault labels directly (AST check).
11. Generated claim objects do not contain hidden labels.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict

import pytest

from shard.agents.fraudster import FraudsterAgent, FraudsterRoundOutput
from shard.core.models import (
    Claim,
    CriticFeedbackSummary,
    CriticReport,
    FraudTypology,
    TypologyMetrics,
)
from shard.data.generator import DEFAULT_TYPOLOGY_WEIGHTS, GenerationBatch


# ---------------------------------------------------------------------------
# 1. Initial Weights Follow Documented Defaults
# ---------------------------------------------------------------------------

def test_initial_weights_follow_defaults() -> None:
    """Fraudster initializes with documented default weights summing to 1.0."""
    agent = FraudsterAgent(base_seed=42)
    assert agent.current_weights == pytest.approx(DEFAULT_TYPOLOGY_WEIGHTS, abs=1e-3)
    assert sum(agent.current_weights.values()) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Missed Typologies Increase in Relative Sampling Weight
# ---------------------------------------------------------------------------

def test_missed_typologies_increase_weight() -> None:
    """Typologies with high miss rates receive increased sampling weight in subsequent rounds."""
    agent = FraudsterAgent(base_seed=42, adaptation_strength=0.20)
    prior_weight_ghost = agent.current_weights[FraudTypology.GHOST_VEHICLE.value]

    # Feedback where GHOST_VEHICLE was heavily missed (detection rate 0.0)
    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.GHOST_VEHICLE.value: 3},
        typology_detection_rates={FraudTypology.GHOST_VEHICLE.value: 0.0},
        total_evaluated=10,
        overall_recall=0.70,
    )

    out = agent.generate_round(round_number=2, feedback=feedback)
    new_weight_ghost = out.config.typology_weights[FraudTypology.GHOST_VEHICLE.value]

    assert new_weight_ghost > prior_weight_ghost
    assert out.adaptation_record.adapted is True
    assert sum(out.config.typology_weights.values()) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Typologies With Lower Miss Rates Do Not Receive Unjustified Increase
# ---------------------------------------------------------------------------

def test_well_detected_typologies_do_not_increase() -> None:
    """Typologies with 0 misses (100% detection) do not receive weight increases."""
    agent = FraudsterAgent(base_seed=42, adaptation_strength=0.20)
    prior_staged = agent.current_weights[FraudTypology.STAGED_COLLISION.value]

    # Only GHOST_VEHICLE missed; STAGED_COLLISION 100% detected
    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={
            FraudTypology.GHOST_VEHICLE.value: 3,
            FraudTypology.STAGED_COLLISION.value: 0,
        },
        typology_detection_rates={
            FraudTypology.GHOST_VEHICLE.value: 0.0,
            FraudTypology.STAGED_COLLISION.value: 1.0,
        },
        total_evaluated=10,
        overall_recall=0.70,
    )

    out = agent.generate_round(round_number=2, feedback=feedback)
    new_staged = out.config.typology_weights[FraudTypology.STAGED_COLLISION.value]

    # STAGED_COLLISION should have shrunk relatively because GHOST_VEHICLE was boosted
    assert new_staged < prior_staged


# ---------------------------------------------------------------------------
# 4. Weights Are Bounded and Normalized
# ---------------------------------------------------------------------------

def test_weights_are_bounded_and_normalized() -> None:
    """Weights obey min_weight, max_weight constraints and strictly sum to 1.0."""
    max_w = 0.50
    min_w = 0.05
    agent = FraudsterAgent(
        base_seed=42,
        adaptation_strength=0.50,  # Aggressive boost
        max_weight=max_w,
        min_weight=min_w,
    )

    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.EXAGGERATED_REPAIR.value: 10},
        typology_detection_rates={FraudTypology.EXAGGERATED_REPAIR.value: 0.0},
        total_evaluated=10,
        overall_recall=0.0,
    )

    out = agent.generate_round(round_number=2, feedback=feedback)
    weights = out.config.typology_weights

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)
    for typ, w in weights.items():
        assert w <= max_w + 1e-4, f"Weight {w} exceeded max {max_w}"
        assert w >= min_w - 1e-4, f"Weight {w} fell below min {min_w}"


# ---------------------------------------------------------------------------
# 5. Seed Determinism Across Identical Inputs
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_identical_batches() -> None:
    """Same base seed, round number and feedback produce bit-identical claim batches."""
    feedback = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={FraudTypology.GHOST_VEHICLE.value: 2},
        typology_detection_rates={FraudTypology.GHOST_VEHICLE.value: 0.33},
        total_evaluated=10,
        overall_recall=0.6,
    )

    agent1 = FraudsterAgent(base_seed=1234)
    out1 = agent1.generate_round(round_number=2, feedback=feedback, n_claims=15)

    agent2 = FraudsterAgent(base_seed=1234)
    out2 = agent2.generate_round(round_number=2, feedback=feedback, n_claims=15)

    assert out1.config == out2.config
    assert out1.adaptation_record.to_dict() == out2.adaptation_record.to_dict()
    assert [c.claim_id for c in out1.batch.claims] == [c.claim_id for c in out2.batch.claims]
    assert [c.claimed_amount for c in out1.batch.claims] == [c.claimed_amount for c in out2.batch.claims]


# ---------------------------------------------------------------------------
# 6. Different Seeds Produce Different Batches
# ---------------------------------------------------------------------------

def test_different_seeds_produce_different_batches() -> None:
    """Different base seeds generate distinct claim batches."""
    agent_a = FraudsterAgent(base_seed=101)
    out_a = agent_a.generate_round(round_number=1, n_claims=20)

    agent_b = FraudsterAgent(base_seed=202)
    out_b = agent_b.generate_round(round_number=1, n_claims=20)

    ids_a = [c.claim_id for c in out_a.batch.claims]
    ids_b = [c.claim_id for c in out_b.batch.claims]
    assert ids_a != ids_b


# ---------------------------------------------------------------------------
# 7. Missing Previous-Round Feedback Handled Explicitly
# ---------------------------------------------------------------------------

def test_missing_feedback_handled_explicitly() -> None:
    """When feedback is None (e.g. Round 1), agent uses initial weights without error."""
    agent = FraudsterAgent(base_seed=42)
    out = agent.generate_round(round_number=1, feedback=None)

    assert out.adaptation_record.adapted is False
    assert out.adaptation_record.feedback_summary == {"status": "INITIAL_DISTRIBUTION"}
    assert out.config.typology_weights == agent.current_weights


# ---------------------------------------------------------------------------
# 8. Zero-Count Typologies Do Not Cause Division Errors
# ---------------------------------------------------------------------------

def test_zero_count_typologies_no_division_error() -> None:
    """Feedback reporting zero total misses or zero cases does not crash the agent."""
    agent = FraudsterAgent(base_seed=42)
    feedback_zero = CriticFeedbackSummary(
        round_number=1,
        missed_typology_counts={},
        typology_detection_rates={},
        total_evaluated=0,
        overall_recall=0.0,
    )
    out = agent.generate_round(round_number=2, feedback=feedback_zero)
    assert out.adaptation_record.adapted is False
    assert sum(out.config.typology_weights.values()) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 9. Unknown Typology Identifiers Handled Safely
# ---------------------------------------------------------------------------

def test_unknown_typology_identifiers_handled_safely() -> None:
    """Feedback containing unknown/legacy typology names is safely ignored."""
    agent = FraudsterAgent(base_seed=42)
    feedback_unknown = {
        "missed_typology_counts": {
            "NON_EXISTENT_FRAUD_TYPOLOGY": 5,
            FraudTypology.EXAGGERATED_REPAIR.value: 2,
        },
        "typology_detection_rates": {
            "NON_EXISTENT_FRAUD_TYPOLOGY": 0.0,
            FraudTypology.EXAGGERATED_REPAIR.value: 0.5,
        },
    }
    out = agent.generate_round(round_number=2, feedback=feedback_unknown)
    # The known typology EXAGGERATED_REPAIR should be boosted; unknown should not be added to weights
    assert "NON_EXISTENT_FRAUD_TYPOLOGY" not in out.config.typology_weights
    assert sum(out.config.typology_weights.values()) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 10. Agent Does Not Access Vault Labels Directly (AST Inspection)
# ---------------------------------------------------------------------------

def test_fraudster_does_not_import_ground_truth_vault() -> None:
    """Verify via AST that FraudsterAgent never imports GroundTruthVault."""
    mod_path = Path(__file__).resolve().parent.parent / "shard" / "agents" / "fraudster.py"
    assert mod_path.exists()
    tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "vault" not in alias.name, f"Prohibited import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "vault" not in node.module, f"Prohibited from-import: {node.module}"
            for alias in node.names:
                assert alias.name != "GroundTruthVault", f"Prohibited symbol: {alias.name}"


# ---------------------------------------------------------------------------
# 11. Generated Claims Do Not Expose Hidden Labels
# ---------------------------------------------------------------------------

def test_generated_claims_contain_no_hidden_labels() -> None:
    """Verify that claims generated by Fraudster contain zero hidden ground-truth fields."""
    agent = FraudsterAgent(base_seed=42)
    out = agent.generate_round(round_number=1, n_claims=20, fraud_share=0.4)

    for claim in out.batch.claims:
        assert isinstance(claim, Claim)
        assert not hasattr(claim, "is_fraud")
        assert not hasattr(claim, "fraud_typology")
        assert not hasattr(claim, "ground_truth")
