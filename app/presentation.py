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
    "bg_dark": "#f7f8fa",
    "surface": "#ffffff",
    "surface_subtle": "#f3f4f6",
    "border": "#e5e7eb",
    "border_subtle": "#d1d5db",
    "text_primary": "#111827",
    "text_secondary": "#4b5563",
    "text_muted": "#6b7280",
    "accent_blue": "#2563eb",
    "approve": "#16a34a",
    "approve_bg": "#ecfdf5",
    "approve_border": "#a7f3d0",
    "refer": "#d97706",
    "refer_bg": "#fffbeb",
    "refer_border": "#fde68a",
    "fraud": "#dc2626",
    "fraud_bg": "#fef2f2",
    "fraud_border": "#fecaca",
}

DISPOSITION_BADGE_STYLE = {
    "APPROVE": {
        "color": THEME_COLORS["approve"],
        "bg": THEME_COLORS["approve_bg"],
        "border": THEME_COLORS["approve_border"],
        "label": "APPROVE",
    },
    "REFER": {
        "color": THEME_COLORS["refer"],
        "bg": THEME_COLORS["refer_bg"],
        "border": THEME_COLORS["refer_border"],
        "label": "REFER (MANUAL REVIEW)",
    },
    "SUSPECTED_FRAUD": {
        "color": THEME_COLORS["fraud"],
        "bg": THEME_COLORS["fraud_bg"],
        "border": THEME_COLORS["fraud_border"],
        "label": "SUSPECTED FRAUD",
    },
}

CUSTOM_CSS = f"""
<style>
/* 1. Global Viewport & Canvas Background (Diagonal hatching as in screenshot) */
html, body, [data-testid="stAppViewContainer"], .stApp {{
    background-color: #f7f8fa !important;
    background-image: repeating-linear-gradient(
        -45deg,
        #f0f2f5,
        #f0f2f5 1px,
        #f7f8fa 1px,
        #f7f8fa 12px
    ) !important;
    color: #111827 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* 2. Main Content Container */
.main .block-container {{
    max-width: 1240px !important;
    padding-top: 1.25rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* 3. Top Architectural Brand Header */
.top-nav-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 9999px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}}
.brand-mark {{
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 700;
    font-size: 1.15rem;
    color: #111827;
    letter-spacing: -0.02em;
}}
.pillar-icon {{
    display: inline-flex;
    gap: 3px;
    align-items: center;
}}
.pillar-bar {{
    width: 6px;
    height: 18px;
    background: #111827;
    border-radius: 9999px;
}}
.nav-links-row {{
    display: flex;
    align-items: center;
    gap: 28px;
    font-size: 0.88rem;
    font-weight: 500;
    color: #4b5563;
}}
.nav-links-row span {{
    cursor: default;
    transition: color 0.15s ease;
}}
.nav-links-row span:hover {{
    color: #111827;
}}

/* 4. Framed Box with Technical Corner Brackets (from reference) */
.framed-box {{
    position: relative;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 38px 44px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}}
.bracket-tl {{
    position: absolute;
    top: -5px; left: -5px;
    width: 13px; height: 13px;
    border-top: 2px solid #111827;
    border-left: 2px solid #111827;
}}
.bracket-tr {{
    position: absolute;
    top: -5px; right: -5px;
    width: 13px; height: 13px;
    border-top: 2px solid #111827;
    border-right: 2px solid #111827;
}}
.bracket-bl {{
    position: absolute;
    bottom: -5px; left: -5px;
    width: 13px; height: 13px;
    border-bottom: 2px solid #111827;
    border-left: 2px solid #111827;
}}
.bracket-br {{
    position: absolute;
    bottom: -5px; right: -5px;
    width: 13px; height: 13px;
    border-bottom: 2px solid #111827;
    border-right: 2px solid #111827;
}}

/* 5. Typography Hierarchy */
.kicker {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 14px;
}}
.kicker-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #2563eb;
    display: inline-block;
}}
.display-headline {{
    font-size: 2.5rem;
    font-weight: 500;
    color: #111827;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
.body-lead {{
    font-size: 0.98rem;
    line-height: 1.6;
    color: #4b5563;
    margin-bottom: 24px;
}}

/* 6. Capsule Buttons */
.btn-capsule-dark {{
    display: inline-block;
    background-color: #111827;
    color: #ffffff !important;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 10px 24px;
    border-radius: 9999px;
    text-decoration: none;
    margin-right: 12px;
}}
.btn-capsule-light {{
    display: inline-block;
    background-color: #f3f4f6;
    color: #111827 !important;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 10px 24px;
    border-radius: 9999px;
    text-decoration: none;
}}

/* Streamlit Button Overrides */
div.stButton > button[kind="primary"], div.stButton > button {{
    border-radius: 9999px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 24px !important;
    transition: all 0.15s ease !important;
}}
div.stButton > button[kind="primary"] {{
    background-color: #111827 !important;
    color: #ffffff !important;
    border: 1px solid #111827 !important;
}}
div.stButton > button[kind="primary"]:hover {{
    background-color: #27272a !important;
    border-color: #27272a !important;
    box-shadow: 0 4px 12px rgba(17, 24, 39, 0.15) !important;
}}
div.stButton > button:not([kind="primary"]) {{
    background-color: #f3f4f6 !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb !important;
}}
div.stButton > button:not([kind="primary"]):hover {{
    background-color: #e5e7eb !important;
    color: #000000 !important;
}}

/* Streamlit Download Button Overrides */
div.stDownloadButton > button {{
    border-radius: 9999px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 22px !important;
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}}
div.stDownloadButton > button:hover {{
    background-color: #f9fafb !important;
    border-color: #9ca3af !important;
}}

/* 7. Capsule Tabs Bar (Reference style) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background-color: #ffffff !important;
    padding: 6px 10px !important;
    border-radius: 9999px !important;
    border: 1px solid #e5e7eb !important;
    margin-bottom: 24px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9999px !important;
    padding: 8px 18px !important;
    color: #4b5563 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    border: none !important;
    background-color: transparent !important;
    height: auto !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: #111827 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}}

/* 8. KPI Metric Cards */
.kpi-card {{
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.kpi-card:hover {{
    border-color: #d1d5db;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
}}
.kpi-title {{
    color: #6b7280;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}}
.kpi-value {{
    color: #111827;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.15;
}}
.kpi-subtext {{
    color: #9ca3af;
    font-size: 0.76rem;
    margin-top: 4px;
}}

/* 9. Status Banners */
.status-banner {{
    border-radius: 9999px;
    padding: 10px 22px;
    margin-bottom: 20px;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.status-target-met {{
    background-color: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}}
.status-max-rounds {{
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    color: #1e293b;
}}
.status-failed {{
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
}}

/* 10. Disposition Badges (Capsule pills) */
.badge-capsule {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.badge-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}}
.badge-approve {{
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}}
.dot-approve {{
    background-color: #10b981;
}}
.badge-refer {{
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}}
.dot-refer {{
    background-color: #f59e0b;
}}
.badge-fraud {{
    background-color: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}}
.dot-fraud {{
    background-color: #ef4444;
}}

/* 11. Section Header */
.section-header {{
    font-size: 1.15rem;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.01em;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 8px;
    margin-top: 24px;
    margin-bottom: 16px;
}}

/* 12. Right Visual Card with Atmospheric Lighting (Reference Hero) */
.hero-media-card {{
    position: relative;
    width: 100%;
    min-height: 290px;
    border-radius: 24px;
    background: radial-gradient(circle at 75% 65%, rgba(249, 115, 22, 0.8) 0%, rgba(59, 130, 246, 0.7) 40%, rgba(15, 23, 42, 0.95) 75%),
                linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    overflow: hidden;
    box-shadow: 0 12px 28px -6px rgba(15, 23, 42, 0.18);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 24px 28px;
    color: #ffffff;
}}
.hero-media-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0) 40%, rgba(15,23,42,0.85) 100%);
}}
.hero-media-badge {{
    position: relative;
    z-index: 2;
    align-self: flex-start;
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #ffffff;
    margin-bottom: auto;
}}
.hero-media-title {{
    position: relative;
    z-index: 2;
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
}}
.hero-media-desc {{
    position: relative;
    z-index: 2;
    font-size: 0.82rem;
    color: rgba(255, 255, 255, 0.8);
    line-height: 1.4;
}}

/* 13. Sidebar Styling */
[data-testid="stSidebar"] {{
    background-color: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: #111827 !important;
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
    color: #374151 !important;
}}

/* 14. Dataframe and Table Styling */
[data-testid="stDataFrame"] {{
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    overflow: hidden !important;
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
        return f'<span class="badge-capsule badge-approve"><span class="badge-dot dot-approve"></span>APPROVE</span>'
    elif disp_upper == "REFER":
        return f'<span class="badge-capsule badge-refer"><span class="badge-dot dot-refer"></span>REFER</span>'
    elif disp_upper == "SUSPECTED_FRAUD":
        return f'<span class="badge-capsule badge-fraud"><span class="badge-dot dot-fraud"></span>SUSPECTED FRAUD</span>'
    return f'<span class="badge-capsule badge-refer">{disposition}</span>'


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
        title=dict(text="Measured Adversarial Feedback Trajectory", font=dict(color="#111827", size=15)),
        xaxis=dict(
            title="Experiment Round",
            tickmode="linear",
            tick0=1,
            dtick=1,
            gridcolor="#f3f4f6",
            color="#4b5563",
        ),
        yaxis=dict(
            title="Metric (%)",
            range=[0, 105],
            gridcolor="#f3f4f6",
            color="#4b5563",
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#111827"),
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
                font=dict(color="#111827", size=14, family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
                showarrow=False,
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, "#f8fafc"], [1, "#dbeafe"]],
        showscale=False,
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix (Investigation Triage vs Ground Truth)", font=dict(color="#111827", size=14)),
        annotations=annotations,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(color="#111827"),
        yaxis=dict(color="#111827"),
        margin=dict(l=50, r=40, t=50, b=40),
        height=280,
    )

    return fig


def build_discovery_ranking_dataframe(payload: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Construct tabular representation of generic observable feature separation rankings."""
    if not payload or "ranked_features" not in payload:
        return pd.DataFrame(columns=[
            "Rank", "Feature", "Direction", "Separation Score", "SMD",
            "Tail Share", "Missed Mean", "Reference Mean", "Outcome",
        ])
    rows = []
    chosen = payload.get("chosen_feature")
    for rank, rf in enumerate(payload["ranked_features"], 1):
        fname = rf.get("feature_name", "")
        score = rf.get("separation_score", 0.0)
        smd = rf.get("standardized_mean_diff", rf.get("smd", 0.0))
        tail = rf.get("tail_separation_share", rf.get("tail_share_missed", 0.0))
        direction = rf.get("direction", "")
        missed_mean = rf.get("missed_mean", 0.0)
        ref_mean = rf.get("reference_mean", 0.0)
        status = "SELECTED" if fname == chosen else "Evaluated"
        rows.append({
            "Rank": rank,
            "Feature": fname,
            "Direction": direction,
            "Separation Score": f"{score:.4f}",
            "SMD": f"{smd:.4f}",
            "Tail Share": f"{tail * 100:.1f}%",
            "Missed Mean": f"{missed_mean:.2f}",
            "Reference Mean": f"{ref_mean:.2f}",
            "Outcome": status,
        })
    return pd.DataFrame(rows)


def build_discovery_candidates_dataframe(records: Sequence[Any]) -> pd.DataFrame:
    """Construct tabular representation of candidate thresholds and validation gate results."""
    rows = []
    disc_records = [
        r for r in records
        if getattr(r, "proposal_id", "").startswith("PROP-DISC-")
        or getattr(r, "rule_id", "").startswith("RULE_DISCOVERED_")
    ]
    for r in disc_records:
        rows.append({
            "Proposal ID": r.proposal_id,
            "Target Rule": r.rule_id,
            "Candidate Threshold": f"{r.candidate_value}",
            "Status": r.status,
            "Gate Outcome": r.gate_failed if r.gate_failed else "Passed (Gates A–D)",
            "Validation FPR": f"{r.validation_fp_rate * 100:.2f}%",
            "Version Transition": f"v{r.rulebook_version_before} ➔ v{r.rulebook_version_after}",
            "Details": r.rejection_reason if r.status == "REJECTED" else r.rationale,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "Proposal ID", "Target Rule", "Candidate Threshold", "Status",
        "Gate Outcome", "Validation FPR", "Version Transition", "Details",
    ])


def build_cross_round_typology_dataframe(res: ExperimentResult) -> pd.DataFrame:
    """Construct breakdown of fraud typology detection across all experiment rounds."""
    all_typologies = set()
    for r in res.rounds:
        for t in r.critic_report.typology_metrics.keys():
            all_typologies.add(t)
    ordered_typologies = sorted(list(all_typologies))
    rows = []
    for typ in ordered_typologies:
        row = {"Typology": typ}
        total_caught = 0
        total_cases = 0
        for r in res.rounds:
            m = r.critic_report.typology_metrics.get(typ)
            if m:
                row[f"Round {r.round_number}"] = f"{m.caught_cases}/{m.total_cases} ({m.detection_rate * 100:.0f}%)"
                total_caught += m.caught_cases
                total_cases += m.total_cases
            else:
                row[f"Round {r.round_number}"] = "—"
        overall_rate = (total_caught / total_cases * 100) if total_cases > 0 else 0.0
        row["Overall"] = f"{total_caught}/{total_cases} ({overall_rate:.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Typology", "Overall"])


def build_v3_metrics_progression_dataframe(res: ExperimentResult) -> pd.DataFrame:
    """Construct multi-round progression table including unmodelled typology catch rates."""
    rows = []
    for r in res.rounds:
        cr = r.critic_report
        delayed = cr.typology_metrics.get("DELAYED_FILING")
        delayed_str = f"{delayed.caught_cases}/{delayed.total_cases} ({delayed.detection_rate * 100:.0f}%)" if delayed else "—"
        rows.append({
            "Round": f"Round {r.round_number}",
            "Claims": r.claims_count,
            "Precision": f"{cr.precision * 100:.1f}%",
            "Recall": f"{cr.recall * 100:.1f}%",
            "FPR": f"{cr.fpr * 100:.1f}%",
            "DELAYED_FILING Catch Rate": delayed_str,
            "Rulebook Version": f"v{r.rulebook_version_after}",
        })
    return pd.DataFrame(rows)

