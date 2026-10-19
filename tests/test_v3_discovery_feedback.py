"""
Shard v3 — Tests for DiscoveryFeedback and Feature Ranking (Phase 2b-1).

Verifies:
1. Controlled synthetic fixture for feature ranking correctness (both > and < directions).
2. Order-independence of feature ranking (deterministic regardless of input dict order).
3. Insufficient sample guard (n_missed < min_missed_samples returns empty ranking).
4. Trust boundary AST enforcement (no vault, GroundTruthRecord, or FraudTypology imports).
5. Data model field inspection (zero claim IDs, labels, or typologies).
6. Forbidden strings guard test extended to discovery modules.
7. CriticAgent builds DiscoveryFeedback ONLY when enable_discovery=True.
8. Disjointness between discovery reference claims and Gate B validation corpora.
9. Empirical ranking behavior across generated seeds.
"""

from __future__ import annotations

import ast
from pathlib import Path
import pytest
from typing import Dict, List, Tuple

from shard.agents.critic import CriticAgent
from shard.agents.investigator import InvestigationPipeline
from shard.core.discovery_models import DiscoveryFeedback, FeatureAggregateStats
from shard.core.features import FEATURE_WHITELIST, extract_all_features, extract_feature
from shard.data.generator import GeneratorConfig, generate_claims
from shard.discovery.ranking import RankedFeature, rank_features
from shard.rules.rulebook import load_default_rulebook
from shard.rules.validator import GATE_B_SEEDS, GATE_B_V2_SEED, get_default_validation_dataset


# ---------------------------------------------------------------------------
# 1. Controlled Synthetic Fixture for Ranking Correctness
# ---------------------------------------------------------------------------

def test_controlled_synthetic_feature_ranking_greater_than_direction() -> None:
    """A synthetic feature with strong positive separation ranks first."""
    # Target feature: strong shift (> direction)
    target_missed = FeatureAggregateStats(
        count=20, mean=8.5, std=1.2, p5=6.0, p50=8.5, p95=10.0, p97=10.2, p99=10.5, min=5.5, max=11.0,
        tail_share_beyond_ref_p95=0.85,
    )
    target_ref = FeatureAggregateStats(
        count=50, mean=3.0, std=1.0, p5=1.5, p50=3.0, p95=5.0, p97=5.2, p99=5.5, min=1.0, max=6.0,
        tail_share_beyond_ref_p95=0.05,
    )

    # Noise feature: zero separation
    noise_missed = FeatureAggregateStats(
        count=20, mean=10.0, std=2.0, p5=7.0, p50=10.0, p95=13.0, p97=13.5, p99=14.0, min=6.0, max=15.0,
        tail_share_beyond_ref_p95=0.05,
    )
    noise_ref = FeatureAggregateStats(
        count=50, mean=10.1, std=2.1, p5=6.9, p50=10.0, p95=13.1, p97=13.6, p99=14.1, min=5.8, max=15.2,
        tail_share_beyond_ref_p95=0.05,
    )

    feedback = DiscoveryFeedback(
        round_number=1,
        n_missed=20,
        n_reference=50,
        features={
            "noise_feature": (noise_missed, noise_ref),
            "target_feature": (target_missed, target_ref),
        },
        min_missed_samples=8,
        has_sufficient_evidence=True,
    )

    ranked, status = rank_features(feedback)
    assert status == "OK"
    assert len(ranked) == 2
    assert ranked[0].feature_name == "target_feature"
    assert ranked[0].direction == ">"
    assert ranked[0].separation_score > ranked[1].separation_score


def test_controlled_synthetic_feature_ranking_less_than_direction() -> None:
    """A synthetic feature with strong negative shift (< direction) ranks correctly."""
    target_missed = FeatureAggregateStats(
        count=15, mean=15.0, std=5.0, p5=8.0, p50=15.0, p95=22.0, p97=24.0, p99=25.0, min=5.0, max=26.0,
        tail_share_beyond_ref_p95=0.80,
    )
    target_ref = FeatureAggregateStats(
        count=60, mean=120.0, std=20.0, p5=90.0, p50=120.0, p95=155.0, p97=160.0, p99=165.0, min=70.0, max=170.0,
        tail_share_beyond_ref_p95=0.05,
    )

    noise_missed = FeatureAggregateStats(
        count=15, mean=50.0, std=10.0, p5=35.0, p50=50.0, p95=65.0, p97=67.0, p99=70.0, min=30.0, max=72.0,
        tail_share_beyond_ref_p95=0.05,
    )
    noise_ref = FeatureAggregateStats(
        count=60, mean=49.5, std=10.5, p5=34.0, p50=50.0, p95=66.0, p97=68.0, p99=71.0, min=28.0, max=74.0,
        tail_share_beyond_ref_p95=0.05,
    )

    feedback = DiscoveryFeedback(
        round_number=1,
        n_missed=15,
        n_reference=60,
        features={
            "noise_feature": (noise_missed, noise_ref),
            "target_feature": (target_missed, target_ref),
        },
        min_missed_samples=8,
        has_sufficient_evidence=True,
    )

    ranked, status = rank_features(feedback)
    assert status == "OK"
    assert len(ranked) == 2
    assert ranked[0].feature_name == "target_feature"
    assert ranked[0].direction == "<"
    assert ranked[0].separation_score > ranked[1].separation_score


# ---------------------------------------------------------------------------
# 2. Order Independence in Feature Names
# ---------------------------------------------------------------------------

def test_feature_ranking_is_order_independent() -> None:
    """Feature ranking produces identical results regardless of dictionary insertion order."""
    stats_a = (
        FeatureAggregateStats(count=10, mean=5.0, std=1.0, p5=3.0, p50=5.0, p95=7.0, p97=7.2, p99=7.5, min=2.0, max=8.0),
        FeatureAggregateStats(count=30, mean=2.0, std=0.5, p5=1.0, p50=2.0, p95=3.0, p97=3.1, p99=3.2, min=0.5, max=3.5),
    )
    stats_b = (
        FeatureAggregateStats(count=10, mean=10.0, std=2.0, p5=7.0, p50=10.0, p95=13.0, p97=13.5, p99=14.0, min=6.0, max=15.0),
        FeatureAggregateStats(count=30, mean=10.0, std=2.0, p5=7.0, p50=10.0, p95=13.0, p97=13.5, p99=14.0, min=6.0, max=15.0),
    )
    stats_c = (
        FeatureAggregateStats(count=10, mean=100.0, std=15.0, p5=75.0, p50=100.0, p95=125.0, p97=128.0, p99=130.0, min=70.0, max=135.0),
        FeatureAggregateStats(count=30, mean=50.0, std=10.0, p5=35.0, p50=50.0, p95=65.0, p97=67.0, p99=70.0, min=30.0, max=72.0),
    )

    fb_order1 = DiscoveryFeedback(
        round_number=1, n_missed=10, n_reference=30,
        features={"feat_a": stats_a, "feat_b": stats_b, "feat_c": stats_c},
    )
    fb_order2 = DiscoveryFeedback(
        round_number=1, n_missed=10, n_reference=30,
        features={"feat_c": stats_c, "feat_b": stats_b, "feat_a": stats_a},
    )
    fb_order3 = DiscoveryFeedback(
        round_number=1, n_missed=10, n_reference=30,
        features={"feat_b": stats_b, "feat_a": stats_a, "feat_c": stats_c},
    )

    r1, _ = rank_features(fb_order1)
    r2, _ = rank_features(fb_order2)
    r3, _ = rank_features(fb_order3)

    assert [rf.feature_name for rf in r1] == [rf.feature_name for rf in r2] == [rf.feature_name for rf in r3]
    assert [rf.separation_score for rf in r1] == [rf.separation_score for rf in r2] == [rf.separation_score for rf in r3]


# ---------------------------------------------------------------------------
# 3. Insufficient Sample Guard
# ---------------------------------------------------------------------------

def test_insufficient_sample_guard() -> None:
    """When n_missed < min_missed_samples, ranking returns INSUFFICIENT_EVIDENCE."""
    stats = (
        FeatureAggregateStats(count=5, mean=8.0, std=1.0, p5=6.0, p50=8.0, p95=10.0, p97=10.2, p99=10.5, min=5.0, max=11.0),
        FeatureAggregateStats(count=40, mean=3.0, std=0.5, p5=2.0, p50=3.0, p95=4.0, p97=4.1, p99=4.2, min=1.5, max=4.5),
    )
    fb = DiscoveryFeedback(
        round_number=1,
        n_missed=5,
        n_reference=40,
        features={"f1": stats},
        min_missed_samples=8,
        has_sufficient_evidence=False,
    )
    ranked, status = rank_features(fb)
    assert status == "INSUFFICIENT_EVIDENCE"
    assert ranked == []


# ---------------------------------------------------------------------------
# 4. AST Trust Boundary Enforcement
# ---------------------------------------------------------------------------

def test_ast_trust_boundaries_for_discovery_modules() -> None:
    """
    Discovery code and Rule-Writer must NOT import or reference GroundTruthRecord,
    FraudTypology, or GroundTruthVault.
    """
    modules_to_check = [
        Path("shard/discovery/ranking.py"),
        Path("shard/core/discovery_models.py"),
        Path("shard/agents/rule_writer.py"),
    ]

    forbidden_symbols = {"GroundTruthRecord", "FraudTypology", "GroundTruthVault"}

    for mod_path in modules_to_check:
        assert mod_path.exists(), f"{mod_path} must exist"
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_symbols, (
                        f"Forbidden import '{alias.name}' in {mod_path}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                assert "vault" not in module_name, f"Forbidden import from vault in {mod_path}"
                for alias in node.names:
                    assert alias.name not in forbidden_symbols, (
                        f"Forbidden import '{alias.name}' in {mod_path}"
                    )
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_symbols, (
                    f"Forbidden name '{node.id}' in {mod_path}"
                )


# ---------------------------------------------------------------------------
# 5. Data Model Field Inspection (Zero Typology / ID Leakage)
# ---------------------------------------------------------------------------

def test_discovery_models_have_no_leakage_fields() -> None:
    """DiscoveryFeedback and FeatureAggregateStats must contain no ID, label, or typology fields."""
    import dataclasses

    fb_fields = {f.name for f in dataclasses.fields(DiscoveryFeedback)}
    stat_fields = {f.name for f in dataclasses.fields(FeatureAggregateStats)}

    forbidden_field_substrings = ["id", "typology", "label", "is_fraud", "ground_truth"]

    for f in fb_fields:
        for bad in forbidden_field_substrings:
            if bad == "id" and f == "has_sufficient_evidence":
                continue
            assert bad not in f.lower(), f"Forbidden field '{f}' in DiscoveryFeedback"

    for f in stat_fields:
        for bad in forbidden_field_substrings:
            assert bad not in f.lower(), f"Forbidden field '{f}' in FeatureAggregateStats"


# ---------------------------------------------------------------------------
# 6. Forbidden Strings Guard Test
# ---------------------------------------------------------------------------

def test_guard_no_forbidden_strings_in_discovery_engine() -> None:
    """Discovery ranking and discovery models must NOT mention DELAYED_FILING or filing_delay."""
    checked_files = [
        Path("shard/discovery/ranking.py"),
        Path("shard/core/discovery_models.py"),
    ]
    forbidden_strings = ["DELAYED_FILING", "filing_date", "filing_delay"]

    for file_path in checked_files:
        assert file_path.exists()
        content = file_path.read_text(encoding="utf-8")
        for bad_str in forbidden_strings:
            assert bad_str not in content, f"Forbidden string '{bad_str}' found in {file_path}"


# ---------------------------------------------------------------------------
# 7. CriticAgent builds DiscoveryFeedback ONLY when enable_discovery=True
# ---------------------------------------------------------------------------

def test_critic_discovery_feedback_emission_flag() -> None:
    """CriticAgent emits DiscoveryFeedback if and only if enable_discovery=True."""
    cfg = GeneratorConfig(seed=42, n_claims=30, fraud_share=0.3)
    batch = generate_claims(cfg)
    rb = load_default_rulebook()
    pipeline = InvestigationPipeline(rulebook=rb)

    decisions = [pipeline.process(claim=c, round_number=1).decision for c in batch.claims]

    # When enable_discovery=False: report.discovery_feedback must be None
    critic_off = CriticAgent(enable_discovery=False)
    rep_off = critic_off.evaluate(decisions, batch.vault, batch.claims, rb, round_number=1)
    assert rep_off.discovery_feedback is None

    # When enable_discovery=True: report.discovery_feedback must be present
    critic_on = CriticAgent(enable_discovery=True, min_missed_samples=2)
    rep_on = critic_on.evaluate(decisions, batch.vault, batch.claims, rb, round_number=1, round_seed=42)
    assert rep_on.discovery_feedback is not None
    assert isinstance(rep_on.discovery_feedback, DiscoveryFeedback)
    assert rep_on.discovery_feedback.n_reference > 0


# ---------------------------------------------------------------------------
# 8. Disjointness Verification Test
# ---------------------------------------------------------------------------

def test_disjointness_reference_and_gate_b_corpora() -> None:
    """Reference claims gathered by Critic are strictly disjoint from Gate B validation claims."""
    gate_b_claims = get_default_validation_dataset(seed=GATE_B_V2_SEED)
    gate_b_ids = {c.claim_id for c in gate_b_claims}

    # Generate round claims with seed 42
    cfg = GeneratorConfig(seed=42, n_claims=40, fraud_share=0.2)
    batch = generate_claims(cfg)
    critic = CriticAgent(enable_discovery=True)

    # Evaluate
    rb = load_default_rulebook()
    pipeline = InvestigationPipeline(rulebook=rb)
    decisions = [pipeline.process(claim=c, round_number=1).decision for c in batch.claims]
    critic.evaluate(decisions, batch.vault, batch.claims, rb, round_number=1, round_seed=42)

    ref_ids = {c.claim_id for c in critic._accumulated_reference_claims}
    assert ref_ids.isdisjoint(gate_b_ids), f"Overlapping IDs found: {ref_ids & gate_b_ids}"

    # Assert passing Gate B seed raises AssertionError
    with pytest.raises(AssertionError, match="matches Gate B validation seeds"):
        critic.evaluate(decisions, batch.vault, batch.claims, rb, round_number=1, round_seed=GATE_B_V2_SEED)


# ---------------------------------------------------------------------------
# 9. Empirical Ranking Across Seeds (Non-Forced Acceptance Rule)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 123, 456])
def test_empirical_ranking_with_v3_typologies(seed: int) -> None:
    """
    On v3 batches (flag on, start round 1), evaluate missed claims against reference claims.
    Reports empirical rankings across multiple seeds.
    """
    cfg = GeneratorConfig(
        seed=seed,
        n_claims=80,
        fraud_share=0.4,
        include_v3_typologies=True,
        v3_typology_start_round=1,
        round_number=1,
    )
    batch = generate_claims(cfg)
    rb = load_default_rulebook()
    pipeline = InvestigationPipeline(rulebook=rb)
    decisions = [pipeline.process(claim=c, round_number=1).decision for c in batch.claims]

    critic = CriticAgent(enable_discovery=True, min_missed_samples=4)
    report = critic.evaluate(decisions, batch.vault, batch.claims, rb, round_number=1, round_seed=seed)

    assert report.discovery_feedback is not None
    ranked, status = rank_features(report.discovery_feedback)
    assert status == "OK"
    assert len(ranked) == len(FEATURE_WHITELIST)
    # Target feature should separate strongly and rank high
    assert ranked[0].separation_score > 0.0
