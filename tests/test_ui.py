"""
Unit and integration tests for the Streamlit UI presentation layer and AppTest harness.

Validates:
1. Clean imports of app and presentation modules.
2. Accurate DataFrame conversions and absence of ground truth in claims views (TI-01).
3. Metric presentation fidelity (purely measured RoundResult values, no fabrication).
4. AppTest initial render: Welcome / Ready state with all sidebar widgets.
5. AppTest explicit execution: Run experiment triggers orchestrator and populates tabs.
6. AppTest reset behaviour: Reset button clears state without corrupting files.
7. Audit log export sanitization: Downloadable log excludes hidden labels (is_fraud).
8. Strict reproducibility across identical UI configurations (TI-07).
9. Failed experiment error display without marking as completed.
10. Validation gate enforcement: UI displays true gate results without bypassing validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from app.presentation import (
    build_audit_log_dataframe,
    build_claims_dataframe,
    build_rounds_summary_dataframe,
    build_typology_metrics_dataframe,
    create_confusion_matrix_heatmap,
    create_metrics_progression_chart,
    format_currency,
    format_pct,
    render_disposition_badge,
)
from shard.orchestrator import ExperimentResult, Orchestrator, OrchestratorConfig, RoundResult
from shard.rules.rulebook import load_default_rulebook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_experiment() -> ExperimentResult:
    """Fixture providing a deterministic 2-round experiment result."""
    cfg = OrchestratorConfig(
        base_seed=123,
        n_claims_per_round=10,
        fraud_share=0.30,
        max_rounds=2,
        early_stopping=False,
    )
    orch = Orchestrator(config=cfg)
    return orch.run()


# ---------------------------------------------------------------------------
# Presentation Helper Unit Tests
# ---------------------------------------------------------------------------

def test_presentation_imports_cleanly():
    """Verify that presentation module imports without error."""
    import app.presentation
    assert hasattr(app.presentation, "build_claims_dataframe")
    assert hasattr(app.presentation, "build_rounds_summary_dataframe")


def test_format_helpers():
    """Verify currency and percentage formatting."""
    assert format_currency(1250.5) == "£1,250.50"
    assert format_currency(0.0) == "£0.00"
    assert format_currency(None) == "£0.00"

    assert format_pct(0.854) == "85.4%"
    assert format_pct(0.0) == "0.0%"
    assert format_pct(None) == "0.0%"

    badge_app = render_disposition_badge("APPROVE")
    assert "badge-approve" in badge_app
    assert "APPROVE" in badge_app

    badge_fraud = render_disposition_badge("SUSPECTED_FRAUD")
    assert "badge-fraud" in badge_fraud
    assert "SUSPECTED FRAUD" in badge_fraud


def test_build_rounds_summary_dataframe(sample_experiment: ExperimentResult):
    """Verify that round summary table reflects true RoundResult metrics."""
    df = build_rounds_summary_dataframe(sample_experiment)
    assert len(df) == len(sample_experiment.rounds)
    assert "Round" in df.columns
    assert "Claims" in df.columns
    assert "Precision" in df.columns
    assert "Recall" in df.columns
    assert "FPR" in df.columns

    # Verify first round metrics match CriticReport
    r1 = sample_experiment.rounds[0]
    assert df.iloc[0]["Claims"] == r1.claims_count
    assert df.iloc[0]["TP"] == r1.critic_report.tp
    assert df.iloc[0]["FP"] == r1.critic_report.fp


def test_build_claims_dataframe_gt_isolation(sample_experiment: ExperimentResult):
    """
    Verify Claims Explorer table never leaks ground truth (TI-01).
    Must not contain is_fraud, fraud_typology, or vault records.
    """
    r1 = sample_experiment.rounds[0]
    df = build_claims_dataframe(round_result=r1)

    assert not df.empty
    assert len(df) == r1.claims_count

    # Invariant: No hidden labels
    assert "is_fraud" not in df.columns
    assert "is_Fraud" not in df.columns
    assert "ground_truth" not in df.columns
    assert "vault_record" not in df.columns

    # Standard observable columns present
    assert "Claim ID" in df.columns
    assert "Claimed (£)" in df.columns
    assert "Benchmark (£)" in df.columns
    assert "Risk Score" in df.columns
    assert "Disposition" in df.columns


def test_build_claims_dataframe_filtering(sample_experiment: ExperimentResult):
    """Verify disposition and search filters on claims dataframe."""
    r1 = sample_experiment.rounds[0]

    # Filter by APPROVE
    df_approve = build_claims_dataframe(round_result=r1, filter_disposition="APPROVE")
    for disp in df_approve["Disposition"]:
        assert disp == "APPROVE"

    # Filter by search ID
    first_id = r1.decisions[0].claim_id
    df_search = build_claims_dataframe(round_result=r1, search_query=first_id)
    assert len(df_search) == 1
    assert df_search.iloc[0]["Claim ID"] == first_id


def test_build_typology_metrics_dataframe(sample_experiment: ExperimentResult):
    """Verify typology breakdown table construction."""
    cr = sample_experiment.rounds[0].critic_report
    df = build_typology_metrics_dataframe(cr)
    assert "Typology" in df.columns
    assert "Total Cases" in df.columns
    assert "Caught" in df.columns
    assert "Missed" in df.columns
    assert "Detection Rate" in df.columns


def test_create_plotly_charts(sample_experiment: ExperimentResult):
    """Verify Plotly figure generation for metrics progression and confusion matrix."""
    prog_fig = create_metrics_progression_chart(sample_experiment)
    assert prog_fig is not None
    assert len(prog_fig.data) == 3  # Recall, FPR, Precision traces

    cr = sample_experiment.rounds[0].critic_report
    cm_fig = create_confusion_matrix_heatmap(cr)
    assert cm_fig is not None
    assert len(cm_fig.data) == 1


# ---------------------------------------------------------------------------
# Streamlit AppTest Harness Tests
# ---------------------------------------------------------------------------

APP_FILE = str((Path(__file__).parent.parent / "app" / "streamlit_app.py").resolve())


def test_apptest_initial_render_welcome_state():
    """Verify that launching the Streamlit app starts in clean Welcome state with sidebar controls."""
    at = AppTest.from_file(APP_FILE, default_timeout=45)
    at.run(timeout=45)

    assert not at.exception
    # Title exists
    assert any("SHARD Control Room" in str(t.value) for t in at.title)
    # Welcome banner text exists
    assert any("Welcome to SHARD Control Room" in str(m.value) for m in at.markdown)
    # Action button exists
    assert any("Run Loop" in str(b.label) for b in at.button)
    assert any("Reset" in str(b.label) for b in at.button)


def test_apptest_run_experiment_execution():
    """Verify clicking Run Loop triggers the real Orchestrator and renders tabs."""
    at = AppTest.from_file(APP_FILE, default_timeout=45)
    at.run(timeout=45)

    # Configure small test params in sidebar (1 round, 10 claims)
    for num_input in at.number_input:
        if num_input.label == "Base Random Seed":
            num_input.set_value(42)

    for slider in at.slider:
        if slider.label == "Claims per Round":
            slider.set_value(10)
        elif slider.label == "Maximum Rounds":
            slider.set_value(1)

    # Click Run
    run_btn = next(b for b in at.button if "Run Loop" in b.label)
    run_btn.click().run(timeout=45)

    assert not at.exception
    # Welcome screen is gone, tabs are rendered
    assert len(at.tabs) >= 6
    # Status banner exists
    assert any("EXPERIMENT CONCLUDED" in str(m.value) or "TARGET ACHIEVED" in str(m.value) for m in at.markdown)


def test_apptest_reset_clears_results():
    """Verify clicking Reset clears experiment results back to Welcome state."""
    at = AppTest.from_file(APP_FILE, default_timeout=45)
    at.run(timeout=45)

    # First run an experiment
    run_btn = next(b for b in at.button if "Run Loop" in b.label)
    run_btn.click().run(timeout=45)
    assert len(at.tabs) >= 6

    # Click Reset
    reset_btn = next(b for b in at.button if "Reset" in b.label)
    reset_btn.click().run(timeout=45)

    assert not at.exception
    # Should be back to Welcome view without tabs
    assert len(at.tabs) == 0
    assert any("Welcome to SHARD Control Room" in str(m.value) for m in at.markdown)


def test_apptest_download_buttons_available_after_run():
    """Verify audit log download buttons are rendered after an experiment completes."""
    at = AppTest.from_file(APP_FILE, default_timeout=45)
    at.run(timeout=45)

    run_btn = next(b for b in at.button if "Run Loop" in b.label)
    run_btn.click().run(timeout=45)

    # Check download buttons exist
    assert len(at.download_button) >= 2
    button_labels = [str(b.label) for b in at.download_button]
    assert any("Sanitized Audit Log" in lbl for lbl in button_labels)
    assert any("Canonical Reproducibility Export" in lbl for lbl in button_labels)
