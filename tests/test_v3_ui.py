"""
Acceptance tests for SHARD v3 Streamlit UI integration and demonstration readiness.

Validates:
1. Presentation helper functions for v3 discovery ranking, candidate gate results,
   cross-round typology breakdown, and multi-round trajectory.
2. Streamlit AppTest toggle behaviour: defaults switch cleanly between v2 baseline and v3 demo preset.
3. Streamlit AppTest execution with v3 discovery enabled:
   - Dynamic discovery panel renders in Rulebook Evolution tab (Tab E).
   - Multi-round typology trajectory renders in Critic Report tab (Tab D).
   - Ground-truth isolation remains strictly preserved in Claims Explorer (TI-01).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.presentation import (
    build_claims_dataframe,
    build_cross_round_typology_dataframe,
    build_discovery_candidates_dataframe,
    build_discovery_ranking_dataframe,
    build_v3_metrics_progression_dataframe,
)
from shard.core.models import RuleAuditRecord
from shard.orchestrator import ExperimentResult, Orchestrator, OrchestratorConfig


APP_FILE = str((Path(__file__).parent.parent / "app" / "streamlit_app.py").resolve())


# ---------------------------------------------------------------------------
# Unit Tests for Presentation Helpers
# ---------------------------------------------------------------------------

def test_build_discovery_ranking_dataframe_empty():
    """Verify ranking dataframe handles empty or None payload gracefully."""
    df_none = build_discovery_ranking_dataframe(None)
    assert isinstance(df_none, pd.DataFrame)
    assert df_none.empty

    df_empty = build_discovery_ranking_dataframe({})
    assert isinstance(df_empty, pd.DataFrame)
    assert df_empty.empty


def test_build_discovery_ranking_dataframe_populated():
    """Verify ranking dataframe correctly formats real ranking payload."""
    payload: Dict[str, Any] = {
        "ranked_features": [
            {
                "feature_name": "claim_filing_delay_days",
                "separation_score": 4.3007,
                "standardized_mean_diff": 2.7623,
                "tail_separation_share": 0.7692,
                "direction": ">",
                "missed_mean": 7.69,
                "reference_mean": 3.21,
            },
            {
                "feature_name": "policy_tenure_days",
                "separation_score": 1.2285,
                "standardized_mean_diff": 0.7669,
                "tail_separation_share": 0.2308,
                "direction": ">",
                "missed_mean": 306.54,
                "reference_mean": 238.52,
            },
        ],
        "chosen_feature": "claim_filing_delay_days",
        "n_missed": 13,
        "n_reference": 252,
        "round_number": 3,
    }

    df = build_discovery_ranking_dataframe(payload)
    assert len(df) == 2
    assert "Rank" in df.columns
    assert "Feature" in df.columns
    assert "Separation Score" in df.columns
    assert "Outcome" in df.columns
    assert df.iloc[0]["Outcome"] == "SELECTED"
    assert df.iloc[1]["Outcome"] == "Evaluated"
    assert df.iloc[0]["Feature"] == "claim_filing_delay_days"


def test_build_discovery_candidates_dataframe():
    """Verify candidate dataframe formats validation gate outcomes."""
    rec_rejected = RuleAuditRecord(
        proposal_id="PROP-DISC-03-claim_filing_delay_days-p95",
        round_number=3,
        rule_id="RULE_DISCOVERED_001",
        parameter_name="threshold",
        old_value=None,
        candidate_value=5.0,
        rationale="Evaluating threshold 5.0",
        status="REJECTED",
        gate_failed="GATE_B_FP_BUDGET",
        rejection_reason="Candidate rulebook false-positive rate (27.0%) exceeds allowable budget (10.0%).",
        validation_fp_rate=0.27,
        rulebook_version_before="1.0.0",
        rulebook_version_after="1.0.0",
        timestamp="2026-10-20T00:00:00Z",
    )
    rec_accepted = RuleAuditRecord(
        proposal_id="PROP-DISC-03-claim_filing_delay_days-p97",
        round_number=3,
        rule_id="RULE_DISCOVERED_001",
        parameter_name="threshold",
        old_value=None,
        candidate_value=6.0,
        rationale="Discovered rule installed",
        status="ACCEPTED",
        gate_failed=None,
        rejection_reason="",
        validation_fp_rate=0.05,
        rulebook_version_before="1.0.0",
        rulebook_version_after="1.1.0",
        timestamp="2026-10-20T00:00:01Z",
    )

    df = build_discovery_candidates_dataframe([rec_rejected, rec_accepted])
    assert len(df) == 2
    assert "Proposal ID" in df.columns
    assert "Gate Outcome" in df.columns
    assert "Validation FPR" in df.columns
    assert df.iloc[0]["Status"] == "REJECTED"
    assert df.iloc[0]["Gate Outcome"] == "GATE_B_FP_BUDGET"
    assert df.iloc[1]["Status"] == "ACCEPTED"
    assert "Passed (Gates A–D)" in df.iloc[1]["Gate Outcome"]


def test_v3_orchestrator_execution_and_presentation():
    """Verify end-to-end execution of a 2-round v3 experiment and presentation tables."""
    cfg = OrchestratorConfig(
        base_seed=42,
        n_claims_per_round=30,
        fraud_share=0.30,
        max_rounds=2,
        enable_feature_discovery=True,
        v3_typology_start_round=2,
        min_missed_samples=4,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg)
    res = orch.run()

    cross_df = build_cross_round_typology_dataframe(res)
    assert not cross_df.empty
    assert "Typology" in cross_df.columns
    assert "Round 1" in cross_df.columns
    assert "Round 2" in cross_df.columns

    metrics_df = build_v3_metrics_progression_dataframe(res)
    assert len(metrics_df) == 2
    assert "DELAYED_FILING Catch Rate" in metrics_df.columns

    # Verify ground-truth isolation on Round 2 claims table
    claims_df = build_claims_dataframe(res.rounds[1])
    assert "is_fraud" not in claims_df.columns
    assert "fraud_typology" not in claims_df.columns


# ---------------------------------------------------------------------------
# Streamlit AppTest Harness Tests for v3
# ---------------------------------------------------------------------------

def test_apptest_v3_toggle_preset_behavior():
    """Verify that toggling the v3 checkbox switches defaults to the demonstration preset."""
    at = AppTest.from_file(APP_FILE, default_timeout=45)
    at.run(timeout=45)

    # Initial state (v3=False): baseline values
    claims_slider = next(s for s in at.slider if s.label == "Claims per Round")
    fraud_slider = next(s for s in at.slider if s.label == "Synthetic Fraud Share")
    rounds_slider = next(s for s in at.slider if s.label == "Maximum Rounds")
    early_stop_cb = next(c for c in at.checkbox if c.label == "Enable Early Stopping")

    assert claims_slider.value == 30
    assert fraud_slider.value == 0.40
    assert rounds_slider.value == 3
    assert early_stop_cb.value is True

    # Check v3 toggle
    v3_cb = next(c for c in at.checkbox if "SHARD v3 Feature Discovery" in c.label)
    v3_cb.check().run(timeout=45)

    # State with v3=True: demo preset values
    claims_slider = next(s for s in at.slider if s.label == "Claims per Round")
    fraud_slider = next(s for s in at.slider if s.label == "Synthetic Fraud Share")
    rounds_slider = next(s for s in at.slider if s.label == "Maximum Rounds")
    early_stop_cb = next(c for c in at.checkbox if c.label == "Enable Early Stopping")

    assert claims_slider.value == 120
    assert fraud_slider.value == 0.30
    assert rounds_slider.value == 4
    assert early_stop_cb.value is False

    # Check that typology start round input appears
    start_round_input = next(
        (n for n in at.number_input if "V3 Typology Start Round" in n.label),
        None,
    )
    assert start_round_input is not None
    assert start_round_input.value == 2


def test_apptest_v3_experiment_run_and_panel_render():
    """Verify running a small v3 experiment renders the v3 discovery panel and typology trajectory."""
    at = AppTest.from_file(APP_FILE, default_timeout=45)
    at.run(timeout=45)

    # Enable v3
    v3_cb = next(c for c in at.checkbox if "SHARD v3 Feature Discovery" in c.label)
    v3_cb.check().run(timeout=45)

    # Set smaller test values for quick test execution (2 rounds, 15 claims)
    for slider in at.slider:
        if slider.label == "Claims per Round":
            slider.set_value(15)
        elif slider.label == "Maximum Rounds":
            slider.set_value(2)

    # Click Run Loop
    run_btn = next(b for b in at.button if "Run Loop" in b.label)
    run_btn.click().run(timeout=45)

    assert not at.exception
    assert len(at.tabs) >= 6

    # Verify Discovery section header is present in markdown
    assert any("SHARD v3 Feature Discovery & Dynamic Adaptation" in str(m.value) for m in at.markdown)
    # Verify Multi-round typology trajectory is present in markdown
    assert any("Multi-Round Typology Detection Trajectory" in str(m.value) for m in at.markdown)
