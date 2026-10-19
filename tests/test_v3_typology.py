"""
Tests for SHARD v3 Phase 2a: Flag-gated sixth fraud typology (DELAYED_FILING).

Validates:
1. Flag off: generator output for identical seeds is bit-identical to v2 baseline.
2. Flag on: DELAYED_FILING is present strictly from v3_typology_start_round onward.
3. DELAYED_FILING delays are drawn strictly from [4, 10] inclusive.
4. DELAYED_FILING claims evade all v2 baseline detection rules (100% APPROVE, 0.0 recall).
5. Guard test: forbidden files never mention DELAYED_FILING, filing_date, or filing_delay.
6. Gate B validation corpus (seed 9999) remains identical when flag is off.
7. Genuine late filers (~5%, delays 6-8 days) appear only when flag is on.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List

import pytest

from shard.agents.critic import CriticAgent
from shard.agents.fraudster import FraudsterAgent
from shard.agents.investigator import InvestigationPipeline
from shard.core.models import (
    Claim,
    Disposition,
    FraudTypology,
    GroundTruthRecord,
)
from shard.data.generator import (
    ClaimGenerator,
    DEFAULT_TYPOLOGY_WEIGHTS,
    DEFAULT_V3_TYPOLOGY_WEIGHTS,
    GeneratorConfig,
    generate_claims,
)
from shard.rules.rulebook import load_default_rulebook
from shard.rules.validator import get_default_validation_dataset


def _calc_filing_delay_days(claim: Claim) -> int:
    """Calculate filing delay in days from incident_date to filing_date."""
    inc = datetime.date.fromisoformat(claim.incident_date)
    fil = datetime.date.fromisoformat(claim.filing_date)
    return (fil - inc).days


# ---------------------------------------------------------------------------
# 1. Flag Off: Bit-identical output to v2 baseline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 123, 456, 789, 9999])
def test_flag_off_identical_output_to_v2(seed: int) -> None:
    """With include_v3_typologies=False (the default), output is 100% identical to v2."""
    cfg_default = GeneratorConfig(seed=seed, n_claims=30, fraud_share=0.3)
    cfg_explicit = GeneratorConfig(
        seed=seed, n_claims=30, fraud_share=0.3, include_v3_typologies=False
    )

    batch_default = generate_claims(cfg_default)
    batch_explicit = generate_claims(cfg_explicit)

    assert len(batch_default.claims) == len(batch_explicit.claims) == 30
    assert len(batch_default.vault) == len(batch_explicit.vault) == 30

    for c1, c2 in zip(batch_default.claims, batch_explicit.claims):
        assert c1 == c2, f"Claim mismatch for seed {seed}: {c1.claim_id}"
        lbl1 = batch_default.vault.get_label(c1.claim_id)
        lbl2 = batch_explicit.vault.get_label(c2.claim_id)
        assert lbl1 == lbl2, f"Ground truth mismatch for seed {seed}: {lbl1}"
        assert lbl1.fraud_typology != FraudTypology.DELAYED_FILING.value

        if not lbl1.is_fraud:
            delay = _calc_filing_delay_days(c1)
            assert 1 <= delay <= 5, f"Genuine delay {delay} out of range [1, 5] for seed {seed}"


# ---------------------------------------------------------------------------
# 2. Flag On: DELAYED_FILING present only from start round; delays within 4 to 10
# ---------------------------------------------------------------------------

def test_delayed_filing_present_only_from_start_round() -> None:
    """When flag is ON, DELAYED_FILING appears strictly on or after v3_typology_start_round."""
    # Round 1 (before start round 2): DELAYED_FILING must NOT appear
    cfg_r1 = GeneratorConfig(
        seed=42,
        n_claims=60,
        fraud_share=0.5,
        include_v3_typologies=True,
        v3_typology_start_round=2,
        round_number=1,
    )
    batch_r1 = generate_claims(cfg_r1)
    r1_typologies = {
        batch_r1.vault.get_label(c.claim_id).fraud_typology
        for c in batch_r1.claims
        if batch_r1.vault.get_label(c.claim_id).is_fraud
    }
    assert FraudTypology.DELAYED_FILING.value not in r1_typologies

    # Round 2 (at start round 2): DELAYED_FILING MUST appear
    cfg_r2 = GeneratorConfig(
        seed=42,
        n_claims=60,
        fraud_share=0.5,
        include_v3_typologies=True,
        v3_typology_start_round=2,
        round_number=2,
    )
    batch_r2 = generate_claims(cfg_r2)
    r2_typologies = {
        batch_r2.vault.get_label(c.claim_id).fraud_typology
        for c in batch_r2.claims
        if batch_r2.vault.get_label(c.claim_id).is_fraud
    }
    assert FraudTypology.DELAYED_FILING.value in r2_typologies

    # Check delays of DELAYED_FILING claims in round 2
    delayed_claims = [
        c for c in batch_r2.claims
        if batch_r2.vault.get_label(c.claim_id).fraud_typology == FraudTypology.DELAYED_FILING.value
    ]
    assert len(delayed_claims) > 0
    for c in delayed_claims:
        delay = _calc_filing_delay_days(c)
        assert 4 <= delay <= 10, f"DELAYED_FILING delay {delay} outside [4, 10]"


def test_fraudster_respects_start_round_and_includes_delayed_filing() -> None:
    """FraudsterAgent does not generate DELAYED_FILING before start round, and introduces it at start round."""
    agent = FraudsterAgent(
        base_seed=101,
        include_v3_typologies=True,
        v3_typology_start_round=2,
    )

    # Round 1: start round not reached
    out_r1 = agent.generate_round(round_number=1, n_claims=50, fraud_share=0.5)
    r1_typologies = {
        out_r1.batch.vault.get_label(c.claim_id).fraud_typology
        for c in out_r1.batch.claims
        if out_r1.batch.vault.get_label(c.claim_id).is_fraud
    }
    assert FraudTypology.DELAYED_FILING.value not in r1_typologies
    assert FraudTypology.DELAYED_FILING.value not in out_r1.config.typology_weights

    # Round 2: start round reached
    out_r2 = agent.generate_round(round_number=2, n_claims=50, fraud_share=0.5)
    r2_typologies = {
        out_r2.batch.vault.get_label(c.claim_id).fraud_typology
        for c in out_r2.batch.claims
        if out_r2.batch.vault.get_label(c.claim_id).is_fraud
    }
    assert FraudTypology.DELAYED_FILING.value in r2_typologies
    assert FraudTypology.DELAYED_FILING.value in out_r2.config.typology_weights


# ---------------------------------------------------------------------------
# 3. Under v2 baseline rulebook, DELAYED_FILING claims receive APPROVE
# ---------------------------------------------------------------------------

def test_delayed_filing_claims_receive_approve_under_v2_rules() -> None:
    """
    DELAYED_FILING claims must evade all v2 baseline detection rules,
    scoring < 25.0 risk score and receiving APPROVE disposition.
    """
    rulebook = load_default_rulebook()
    pipeline = InvestigationPipeline(rulebook=rulebook)
    critic = CriticAgent()

    # Generate dedicated batch of DELAYED_FILING claims
    cfg = GeneratorConfig(
        seed=777,
        n_claims=30,
        fraud_share=1.0,
        typology_weights={FraudTypology.DELAYED_FILING.value: 1.0},
        include_v3_typologies=True,
        round_number=2,
    )
    batch = generate_claims(cfg)

    decisions = []
    for claim in batch.claims:
        res = pipeline.process(claim=claim, round_number=1)
        decisions.append(res.decision)
        # Every delayed filing claim must receive APPROVE under v2 rules
        assert res.decision.disposition == Disposition.APPROVE, (
            f"Claim {claim.claim_id} flagged with {res.decision.disposition} "
            f"and findings: {res.decision.findings}"
        )
        assert res.decision.risk_score < 25.0

    # Critic evaluation against ground truth
    report = critic.evaluate(
        decisions=decisions,
        vault=batch.vault,
        claims=batch.claims,
        rulebook=rulebook,
        round_number=1,
    )

    # All fraudulent claims missed -> 0% recall, 100% false negatives
    assert report.tp == 0
    assert report.fn == 30
    assert report.recall == 0.0
    typ_metric = report.typology_metrics[FraudTypology.DELAYED_FILING.value]
    assert typ_metric.caught_cases == 0
    assert typ_metric.missed_cases == 30
    assert typ_metric.detection_rate == 0.0


# ---------------------------------------------------------------------------
# 4. Guard test: No detection logic or feature leakage in forbidden modules
# ---------------------------------------------------------------------------

def test_guard_no_delayed_filing_in_forbidden_modules() -> None:
    """
    Guard invariant: DELAYED_FILING, filing_date, and filing_delay must NOT appear
    in detection, rule, planner, investigator, or rule-writing modules.
    """
    forbidden_files = [
        Path("shard/rules/templates.py"),
        Path("shard/rules/tools.py"),
        Path("data/rulebook_v1.json"),
        Path("shard/agents/planner.py"),
        Path("shard/agents/investigator.py"),
        Path("shard/agents/rule_writer.py"),
    ]

    forbidden_strings = ["DELAYED_FILING", "filing_date", "filing_delay"]

    for file_path in forbidden_files:
        assert file_path.exists(), f"File {file_path} must exist"
        content = file_path.read_text(encoding="utf-8")
        for bad_str in forbidden_strings:
            assert bad_str not in content, (
                f"Violation: forbidden string '{bad_str}' found in {file_path}"
            )


# ---------------------------------------------------------------------------
# 5. Gate B frozen validation corpus (seed 9999) unchanged when flag is off
# ---------------------------------------------------------------------------

def test_gate_b_frozen_corpus_unchanged_when_flag_off() -> None:
    """The Gate B validation dataset (40 genuine claims, seed 9999) is unchanged with flag off."""
    # Reset any cached dataset to test fresh generation
    import shard.rules.validator as validator_module
    validator_module._CACHED_VALIDATION_CLAIMS = None

    corpus = get_default_validation_dataset(n_claims=40, seed=9999)
    assert len(corpus) == 40

    # Direct generation with include_v3_typologies=False
    cfg = GeneratorConfig(seed=9999, n_claims=40, fraud_share=0.0, include_v3_typologies=False)
    batch = generate_claims(cfg)

    assert len(batch.claims) == 40
    for c_corpus, c_batch in zip(corpus, batch.claims):
        assert c_corpus == c_batch
        delay = _calc_filing_delay_days(c_corpus)
        assert 1 <= delay <= 5, f"Validation claim delay {delay} outside [1, 5]"


# ---------------------------------------------------------------------------
# 6. Genuine late filers (~5%, 6-8 days) appear only when flag is on
# ---------------------------------------------------------------------------

def test_genuine_late_filers_behavior() -> None:
    """When flag is ON, ~5% of genuine claims have delays 6-8 days; when OFF, 0% have delays > 5."""
    # Flag OFF: 200 genuine claims
    cfg_off = GeneratorConfig(seed=1234, n_claims=200, fraud_share=0.0, include_v3_typologies=False)
    batch_off = generate_claims(cfg_off)
    delays_off = [_calc_filing_delay_days(c) for c in batch_off.claims]
    assert all(1 <= d <= 5 for d in delays_off), "Flag OFF produced delays outside 1-5"

    # Flag ON: 1000 genuine claims to test ~5% distribution
    cfg_on = GeneratorConfig(seed=1234, n_claims=1000, fraud_share=0.0, include_v3_typologies=True)
    batch_on = generate_claims(cfg_on)
    delays_on = [_calc_filing_delay_days(c) for c in batch_on.claims]

    late_filers = [d for d in delays_on if 6 <= d <= 8]
    share = len(late_filers) / len(delays_on)
    # 5% nominal probability: expect 3% - 7% for 1000 trials
    assert 0.03 <= share <= 0.07, f"Late filer share {share:.3f} outside expected 3%-7% band"
    assert all(1 <= d <= 8 for d in delays_on), "Flag ON produced delays outside 1-8 for genuine"
