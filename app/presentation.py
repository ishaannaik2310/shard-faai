"""
Shard — Presentation & Formatting Layer for Streamlit UI.

Provides pure presentation helpers, DataFrame transformations, color styling,
and Plotly chart builders for the SHARD Control Room dashboard.

CRITICAL INVARIANTS:
1. Pure Presentation (No Decision Logic): Formats and visualises data produced by
   the Orchestrator; does NOT duplicate scoring, thresholds, or evaluation logic.
2. Ground-Truth Isolation (TI-01): Investigator-facing tables and views NEVER include
   or reference GroundTruthRecord or hidden labels.
3. Decoupled Recommendations (TI-11): Dispositions are clearly styled as investigation
   triage recommendations (APPROVE, REFER, SUSPECTED_FRAUD), not confirmed legal verdicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import plotly.graph_objects as go

from shard.agents.investigator import PipelineResult
from shard.core.models import (
    Claim,
    ClaimDecision,
    CriticReport,
    Disposition,
    Finding,
    RuleAuditRecord,
    RuleProposal,
)
from shard.logging.audit import AuditEvent
from shard.orchestrator import ExperimentResult, RoundResult


# ---------------------------------------------------------------------------
# Visual Theme & Styling Constants
# ---------------------------------------------------------------------------

THEME_COLORS = {
    "bg_dark": "#0e1117",
    "surface": "#161b22",
    "surface_subtle": "#21262d",
    "border": "#30363d",
    "text_primary": "#c9d1d9",
    "text_muted": "#8b949e",
    "accent_blue": "#58a6ff",
    "approve": "#3fb950",
    "approve_bg": "#122a1b",
    "refer": "#d29922",
    "refer_bg": "#332612",
    "fraud": "#f85149",
    "fraud_bg": "#3c1618",
}

DISPOSITION_BADGE_STYLE = {
    "APPROVE": {
        "color": THEME_COLORS["approve"],
        "bg": THEME_COLORS["approve_bg"],
        "border": THEME_COLORS["approve"],
        "label": "APPROVE",
    },
    "REFER": {
        "color": THEME_COLORS["refer"],
        "bg": THEME_COLORS["refer_bg"],
        "border": THEME_COLORS["refer"],
        "label": "REFER (MANUAL REVIEW)",
    },
    "SUSPECTED_FRAUD": {
        "color": THEME_COLORS["fraud"],
        "bg": THEME_COLORS["fraud_bg"],
        "border": THEME_COLORS["fraud"],
        "label": "SUSPECTED FRAUD",
    },
}

CUSTOM_CSS = f"""
<style>
/* Main App Styling */
.main {{
    background-color: {THEME_COLORS['bg_dark']};
    color: {THEME_COLORS['text_primary']};
}}

/* KPI Metric Card */
.kpi-card {{
    background-color: {THEME_COLORS['surface']};
    border: 1px solid {THEME_COLORS['border']};
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
.kpi-title {{
    color: {THEME_COLORS['text_muted']};
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}}
.kpi-value {{
    color: #ffffff;
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
}}
.kpi-subtext {{
    color: {THEME_COLORS['text_muted']};
    font-size: 0.78rem;
    margin-top: 4px;
}}

/* Status Banner */
.status-banner {{
    border-radius: 6px;
    padding: 12px 18px;
    margin-bottom: 16px;
    border: 1px solid;
    font-size: 0.95rem;
}}
.status-target-met {{
    background-color: {THEME_COLORS['approve_bg']};
    border-color: {THEME_COLORS['approve']};
    color: #e6ffed;
}}
.status-max-rounds {{
    background-color: {THEME_COLORS['refer_bg']};
    border-color: {THEME_COLORS['refer']};
    color: #fff8c5;
}}
.status-failed {{
    background-color: {THEME_COLORS['fraud_bg']};
    border-color: {THEME_COLORS['fraud']};
    color: #ffebe9;
}}

/* Disposition Pills */
.badge-approve {{
    background-color: {THEME_COLORS['approve_bg']};
    color: {THEME_COLORS['approve']};
    border: 1px solid {THEME_COLORS['approve']};
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}}
.badge-refer {{
    background-color: {THEME_COLORS['refer_bg']};
    color: {THEME_COLORS['refer']};
    border: 1px solid {THEME_COLORS['refer']};
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}}
.badge-fraud {{
    background-color: {THEME_COLORS['fraud_bg']};
    color: {THEME_COLORS['fraud']};
    border: 1px solid {THEME_COLORS['fraud']};
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}}

/* Section Header */
.section-header {{
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    border-bottom: 1px solid {THEME_COLORS['border']};
    padding-bottom: 6px;
    margin-top: 18px;
    margin-bottom: 14px;
}}
</style>
"""


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_currency(amount: Optional[float]) -> str:
    """Format floating point currency to UK standard string."""
    if amount is None:
        return "£0.00"
    return f"£{amount:,.2f}"


def format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format ratio to percentage string."""
    if value is None:
        return "0.0%"
    return f"{value * 100:.{decimals}f}%"


def render_disposition_badge(disposition: str) -> str:
    """Produce HTML badge for a triage disposition recommendation."""
    disp_upper = disposition.upper()
    if disp_upper == "APPROVE":
        return f'<span class="badge-approve">APPROVE</span>'
    elif disp_upper == "REFER":
        return f'<span class="badge-refer">REFER</span>'
    elif disp_upper == "SUSPECTED_FRAUD":
        return f'<span class="badge-fraud">SUSPECTED FRAUD</span>'
    return f'<span class="badge-refer">{disposition}</span>'


# ---------------------------------------------------------------------------
# DataFrame Builders
# ---------------------------------------------------------------------------

def build_rounds_summary_dataframe(experiment: ExperimentResult) -> pd.DataFrame:
    """
    Construct a clean tabular history of all rounds in an experiment.
    All values are derived directly from actual RoundResult objects.
    """
    rows = []
    for r in experiment.rounds:
        cr = r.critic_report
        accepted = sum(1 for rec in r.rule_audit_records if rec.status == "ACCEPTED")
        rejected = sum(1 for rec in r.rule_audit_records if rec.status == "REJECTED")

        rows.append({
            "Round": r.round_number,
            "Claims": r.claims_count,
            "TP": cr.tp,
            "FP": cr.fp,
            "TN": cr.tn,
            "FN": cr.fn,
            "Precision": f"{cr.precision * 100:.1f}%",
            "Recall": f"{cr.recall * 100:.1f}%",
            "F1": f"{cr.f1 * 100:.1f}%",
            "FPR": f"{cr.fpr * 100:.1f}%",
            "Accuracy": f"{cr.accuracy * 100:.1f}%",
            "Rulebook Version": f"{r.rulebook_version_before} -> {r.rulebook_version_after}",
            "Proposals (Acc/Rej)": f"{len(r.rule_proposals)} ({accepted} acc, {rejected} rej)",
            "Continuation": r.stop_reason or r.continuation_decision,
        })
    return pd.DataFrame(rows)


def build_claims_dataframe(
    round_result: RoundResult,
    filter_disposition: Optional[str] = None,
    min_score: float = 0.0,
    search_query: str = "",
) -> pd.DataFrame:
    """
    Construct a tabular view of observable claims and triage decisions.

    CRITICAL INVARIANT (TI-01):
    Contains NO ground truth labels or hidden vault attributes.
    """
    rows = []
    claims_by_id = {c.claim_id: c for c in round_result.claims}

    for dec in round_result.decisions:
        disp_val = dec.disposition.value

        # Apply filters
        if filter_disposition and filter_disposition != "ALL" and disp_val != filter_disposition:
            continue
        if dec.risk_score < min_score:
            continue
        if search_query:
            q = search_query.lower()
            if q not in dec.claim_id.lower():
                continue

        claim = claims_by_id.get(dec.claim_id)
        claimed_amt = claim.claimed_amount if claim else 0.0
        benchmark_amt = claim.benchmark_repair_cost if claim else 0.0
        provider_id = claim.repair_provider_id if claim else "N/A"
        make_model = f"{claim.vehicle_make} {claim.vehicle_model}" if claim else "N/A"

        triggered_rules = [f.check_name for f in dec.findings if "FLAGGED" in f.result]

        rows.append({
            "Claim ID": dec.claim_id,
            "Vehicle": make_model,
            "Claimed (£)": round(claimed_amt, 2),
            "Benchmark (£)": round(benchmark_amt, 2),
            "Provider": provider_id,
            "Risk Score": round(dec.risk_score, 1),
            "Disposition": disp_val,
            "Findings Count": len(dec.findings),
            "Triggered Rules": ", ".join(triggered_rules) if triggered_rules else "None",
        })

    df = pd.DataFrame(rows)
    # Strictly enforce that ground-truth labels are not present
    assert "is_fraud" not in df.columns, "TI-01 Violation: is_fraud found in claims table"
    assert "vault_record" not in df.columns, "TI-01 Violation: vault_record found in claims table"
    return df


def build_typology_metrics_dataframe(report: CriticReport) -> pd.DataFrame:
    """Construct a breakdown of detection performance across synthetic fraud typologies."""
    rows = []
    for typology_name, m in report.typology_metrics.items():
        rows.append({
            "Typology": typology_name,
            "Total Cases": m.total_cases,
            "Caught": m.caught_cases,
            "Missed": m.missed_cases,
            "Detection Rate": f"{m.detection_rate * 100:.1f}%",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Typology", "Total Cases", "Caught", "Missed", "Detection Rate"])


def build_audit_log_dataframe(
    events: Sequence[AuditEvent],
    filter_round: Optional[int] = None,
    filter_type: Optional[str] = None,
) -> pd.DataFrame:
    """Format structured audit events for table display."""
    rows = []
    for e in events:
        if filter_round is not None and filter_round != 0 and e.round_number != filter_round:
            continue
        if filter_type and filter_type != "ALL" and e.event_type != filter_type:
            continue

        # Extract concise summary based on event type
        summary = ""
        p = e.payload
        if e.event_type == "EXPERIMENT_INIT":
            summary = f"Config: {p.get('config', {}).get('max_rounds')} rounds, seed={p.get('config', {}).get('base_seed')}"
        elif e.event_type == "ROUND_START":
            summary = f"Rulebook {p.get('rulebook_version')}, seed={p.get('round_seed')}"
        elif e.event_type == "CLAIM_DECISION":
            summary = f"Claim {p.get('claim_id')}: Score={p.get('risk_score')} -> {p.get('disposition')}"
        elif e.event_type == "CRITIC_REPORT":
            summary = f"Evaluated {p.get('total_evaluated')}: TP={p.get('tp')}, FP={p.get('fp')}, Recall={p.get('recall'):.2%}"
        elif e.event_type == "RULE_PROPOSAL":
            summary = f"Proposal on {p.get('rule_id')}.{p.get('parameter_name')}: {p.get('old_value')} -> {p.get('candidate_value')}"
        elif e.event_type == "RULE_VALIDATION":
            summary = f"Outcome: {p.get('status')} (Gate Failed: {p.get('gate_failed') or 'None'})"
        elif e.event_type == "FRAUDSTER_ADAPTATION":
            summary = f"Adapted: {p.get('adapted')} ({p.get('feedback_summary', {}).get('total_misses', 0)} misses observed)"
        elif e.event_type == "ROUND_END":
            summary = f"Processed {p.get('claims_processed')}, reason: {p.get('continuation_reason')}"
        elif e.event_type == "EXPERIMENT_END":
            summary = f"Stop: {p.get('final_stop_reason')}, Total Claims={p.get('total_claims')}"

        rows.append({
            "Seq": e.sequence,
            "Event Type": e.event_type,
            "Round": e.round_number,
            "Timestamp": e.timestamp[:19].replace("T", " ") if e.timestamp else "",
            "Summary": summary,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotly Chart Builders
# ---------------------------------------------------------------------------

def create_metrics_progression_chart(experiment: ExperimentResult) -> go.Figure:
    """
    Generate an interactive Plotly multi-round metrics progression chart.
    Uses TRUE RoundResult values from the executed experiment.
    """
    rounds = [r.round_number for r in experiment.rounds]
    recalls = [r.critic_report.recall * 100 for r in experiment.rounds]
    fprs = [r.critic_report.fpr * 100 for r in experiment.rounds]
    precisions = [r.critic_report.precision * 100 for r in experiment.rounds]

    fig = go.Figure()

    # Fraud Recall curve
    fig.add_trace(go.Scatter(
        x=rounds,
        y=recalls,
        mode="lines+markers",
        name="Fraud Recall (%)",
        line=dict(color=THEME_COLORS["approve"], width=3),
        marker=dict(size=9, symbol="circle"),
        hovertemplate="Round %{x}<br>Recall: %{y:.1f}%<extra></extra>",
    ))

    # False Positive Rate curve
    fig.add_trace(go.Scatter(
        x=rounds,
        y=fprs,
        mode="lines+markers",
        name="False Positive Rate (%)",
        line=dict(color=THEME_COLORS["fraud"], width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="Round %{x}<br>FPR: %{y:.1f}%<extra></extra>",
    ))

    # Precision curve
    fig.add_trace(go.Scatter(
        x=rounds,
        y=precisions,
        mode="lines+markers",
        name="Precision (%)",
        line=dict(color=THEME_COLORS["accent_blue"], width=2, dash="dot"),
        marker=dict(size=7, symbol="square"),
        hovertemplate="Round %{x}<br>Precision: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Measured Adversarial Feedback Trajectory", font=dict(color="#ffffff", size=15)),
        xaxis=dict(
            title="Experiment Round",
            tickmode="linear",
            tick0=1,
            dtick=1,
            gridcolor=THEME_COLORS["border"],
            color=THEME_COLORS["text_muted"],
        ),
        yaxis=dict(
            title="Metric (%)",
            range=[0, 105],
            gridcolor=THEME_COLORS["border"],
            color=THEME_COLORS["text_muted"],
        ),
        paper_bgcolor=THEME_COLORS["surface"],
        plot_bgcolor=THEME_COLORS["surface"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=THEME_COLORS["text_primary"]),
        ),
        margin=dict(l=40, r=30, t=50, b=40),
        height=340,
    )

    return fig


def create_confusion_matrix_heatmap(report: CriticReport) -> go.Figure:
    """Generate a clean 2x2 confusion matrix heatmap for a round's evaluation."""
    z = [
        [report.tn, report.fp],
        [report.fn, report.tp],
    ]
    x = ["Predicted Genuine (APPROVE)", "Predicted Suspect (REFER / FRAUD)"]
    y = ["Actual Genuine", "Actual Fraud"]

    annotations = []
    labels = [
        [f"TN: {report.tn}", f"FP: {report.fp}"],
        [f"FN: {report.fn}", f"TP: {report.tp}"],
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=x[j],
                y=y[i],
                text=labels[i][j],
                font=dict(color="#ffffff", size=14, family="sans-serif"),
                showarrow=False,
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, THEME_COLORS["surface_subtle"]], [1, THEME_COLORS["accent_blue"]]],
        showscale=False,
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix (Investigation Triage vs Ground Truth)", font=dict(color="#ffffff", size=14)),
        annotations=annotations,
        paper_bgcolor=THEME_COLORS["surface"],
        plot_bgcolor=THEME_COLORS["surface"],
        xaxis=dict(color=THEME_COLORS["text_primary"]),
        yaxis=dict(color=THEME_COLORS["text_primary"]),
        margin=dict(l=50, r=40, t=50, b=40),
        height=280,
    )

    return fig
