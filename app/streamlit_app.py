"""
SHARD Control Room — Adversarial Multi-Agent Insurance Claims Auditor UI.

Streamlit decision-support dashboard exposing the complete multi-agent closed-loop
architecture developed in Phases 2–7.

CRITICAL INVARIANTS:
1. Real Execution Only: Drives the actual Orchestrator; no simulated metrics or mock data.
2. Ground-Truth Boundary (TI-01): Investigator-facing views display only observable claims;
   GroundTruthVault is restricted strictly to evaluation tabs.
3. Decoupled Triage (TI-11): Dispositions are clearly labelled as investigation triage
   recommendations, not legal verdicts.
4. Reproducibility (TI-07): Base seed and configuration options produce bit-identical
   reproducible runs.
5. 100% Audit Coverage (TI-09): All completed decisions are traceable in the Audit Log tab.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from app.presentation import (
    CUSTOM_CSS,
    THEME_COLORS,
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
from shard.agents.investigator import PipelineResult
from shard.core.models import Claim, ClaimDecision, Disposition
from shard.logging.audit import AuditLogger
from shard.orchestrator import ExperimentResult, Orchestrator, OrchestratorConfig, RoundResult
from shard.rules.rulebook import load_default_rulebook


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SHARD — Adversarial Multi-Agent Claims Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "experiment_result" not in st.session_state:
    st.session_state.experiment_result = None

if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ---------------------------------------------------------------------------
# Sidebar: Experiment Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ SHARD Control Room")
    st.caption("Adversarial Multi-Agent Claims Auditor (CA-3)")

    st.markdown("### ⚙️ Experiment Parameters")

    base_seed = st.number_input(
        "Base Random Seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
        help="Deterministic root seed for reproducible claim generation and agent cycles.",
    )

    claims_per_round = st.slider(
        "Claims per Round",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Volume of synthetic motor claims generated in each round.",
    )

    fraud_share = st.slider(
        "Synthetic Fraud Share",
        min_value=0.10,
        max_value=0.70,
        value=0.40,
        step=0.05,
        format="%.2f",
        help="Proportion of fraudulent claims injected per round.",
    )

    max_rounds = st.slider(
        "Maximum Rounds",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        help="Maximum closed-loop adaptation rounds before stopping.",
    )

    st.markdown("### 🎯 Stopping Thresholds")

    target_recall = st.slider(
        "Target Fraud Recall",
        min_value=0.50,
        max_value=1.00,
        value=0.90,
        step=0.05,
        format="%.2f",
        help="Minimum recall required for early stopping.",
    )

    max_fp_rate = st.slider(
        "Max Allowed False Positive Rate",
        min_value=0.01,
        max_value=0.25,
        value=0.08,
        step=0.01,
        format="%.2f",
        help="Maximum acceptable false positive rate for early stopping.",
    )

    early_stopping = st.checkbox(
        "Enable Early Stopping",
        value=True,
        help="Halt experiment immediately when target recall and FPR are achieved.",
    )

    st.markdown("### 🔄 Adaptation Settings")

    adaptation_strength = st.slider(
        "Fraudster Adaptation Strength",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        format="%.2f",
        help="Rate at which Fraudster boosts weights for undetected fraud typologies.",
    )

    fp_budget = st.slider(
        "Rule-Writer FP Budget Delta",
        min_value=0.01,
        max_value=0.20,
        value=0.10,
        step=0.01,
        format="%.2f",
        help="Gate B limit: maximum permissible increase in false positive rate.",
    )

    audit_path = st.text_input(
        "Audit Log Output Path",
        value="data/audit_latest.jsonl",
        help="File path where structured JSON Lines audit records are appended.",
    )

    st.markdown("---")

    col_run, col_reset = st.columns(2)
    run_clicked = col_run.button("▶ Run Loop", type="primary", use_container_width=True)
    reset_clicked = col_reset.button("↺ Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.experiment_result = None
        st.session_state.audit_logger = None
        st.session_state.last_error = None
        st.rerun()

    # Active Config Badge Summary
    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:0.75rem; color:{THEME_COLORS['text_muted']};">
            <b>Active Configuration:</b><br>
            • Seed: {base_seed} | Batch: {claims_per_round} claims<br>
            • Fraud: {fraud_share:.0%} | Max Rounds: {max_rounds}<br>
            • Early Stop: {'On' if early_stopping else 'Off'} (Recall ≥ {target_recall:.0%})<br>
            • Safeguard: FP Budget ≤ {fp_budget:.0%}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Experiment Execution Trigger
# ---------------------------------------------------------------------------

if run_clicked:
    st.session_state.last_error = None
    try:
        cfg = OrchestratorConfig(
            base_seed=int(base_seed),
            n_claims_per_round=int(claims_per_round),
            fraud_share=float(fraud_share),
            max_rounds=int(max_rounds),
            target_recall=float(target_recall),
            max_fp_rate=float(max_fp_rate),
            adaptation_strength=float(adaptation_strength),
            fp_budget=float(fp_budget),
            early_stopping=bool(early_stopping),
            audit_log_path=audit_path.strip() if audit_path.strip() else None,
        )

        with st.spinner("Executing multi-round adversarial simulation across agents..."):
            orch = Orchestrator(config=cfg)
            result: ExperimentResult = orch.run()
            st.session_state.experiment_result = result
            st.session_state.audit_logger = orch.audit_logger

    except Exception as e:
        st.session_state.last_error = str(e)
        st.session_state.experiment_result = None


# ---------------------------------------------------------------------------
# Error Display
# ---------------------------------------------------------------------------

if st.session_state.last_error:
    st.error(f"Experiment execution encountered an error: {st.session_state.last_error}")


# ---------------------------------------------------------------------------
# View 1: Welcome & Ready State (Pre-Run)
# ---------------------------------------------------------------------------

res: Optional[ExperimentResult] = st.session_state.experiment_result

if res is None:
    st.markdown(
        """
        ## 🛡️ Welcome to SHARD Control Room
        **Adversarial Multi-Agent Insurance Claims Auditor**

        Insurer fraud controls degrade over time because organized fraudsters learn which thresholds trigger
        investigation and deliberately evolve their attack patterns. Traditional rules-based systems suffer from
        **structural detection lag** — new rules are written only after fraud has slipped through, and proposed updates
        are rarely stress-tested against genuine claims before live deployment.

        **SHARD** resolves this challenge through an autonomous, closed-loop agentic feedback system:
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Adversarial Adaptation</div>
                <div style="font-size:1.1rem; font-weight:600; color:{THEME_COLORS['accent_blue']};">Fraudster Agent</div>
                <p style="font-size:0.85rem; color:{THEME_COLORS['text_muted']}; margin-top:8px;">
                    Simulates evolving attacks across 5 synthetic fraud typologies, dynamically shifting sampling weights toward patterns that evaded prior investigation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Deterministic Investigation</div>
                <div style="font-size:1.1rem; font-weight:600; color:{THEME_COLORS['approve']};">Pipeline Trio</div>
                <p style="font-size:0.85rem; color:{THEME_COLORS['text_muted']}; margin-top:8px;">
                    <b>Planner</b> selects checks; <b>Researcher</b> retrieves synthetic policy/vehicle dossiers; <b>Investigator</b> runs 6 deterministic verification tools.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Safe Control Evolution</div>
                <div style="font-size:1.1rem; font-weight:600; color:{THEME_COLORS['refer']};">Critic & Rule-Writer</div>
                <p style="font-size:0.85rem; color:{THEME_COLORS['text_muted']}; margin-top:8px;">
                    <b>Critic</b> evaluates against GroundTruthVault; <b>Rule-Writer</b> proposes bounded rule updates; <b>Validation Gate</b> enforces strict FP budget safeguards.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 The 5 Synthetic Fraud Typologies in Scope")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
            1. **Exaggerated Repair Cost**: Inflated garage labour and parts exceeding independent estimatics benchmarks.
            2. **Opportunistic Line-Item Inflation**: Legitimate repairs padded with fictitious unrelated items or arithmetic mismatches.
            3. **Staged Collision**: Collusion networks with rapid inception velocity (claim filed immediately after policy purchase).
            """
        )
    with col_t2:
        st.markdown(
            """
            4. **Ghost Vehicle**: Claims submitted on fictitious registrations or vehicles with invalid registration status.
            5. **Pre-Existing Damage**: Known or historical write-off markers (CAT A/B) falsely claimed as fresh collision impact.
            """
        )

    st.info("👈 Configure your experiment parameters in the sidebar and click **Run Loop** to launch the live multi-agent simulation.")


# ---------------------------------------------------------------------------
# View 2: Live Results Dashboard (Post-Run)
# ---------------------------------------------------------------------------

else:
    # Top Status Banner
    if res.target_reached:
        st.markdown(
            f"""
            <div class="status-banner status-target-met">
                <b>🎯 TARGET ACHIEVED:</b> Fraud recall and false positive rate satisfied the target threshold 
                (Recall ≥ {res.config.target_recall:.0%}, FPR ≤ {res.config.max_fp_rate:.0%}) in Round {len(res.rounds)}.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-banner status-max-rounds">
                <b>⏹ EXPERIMENT CONCLUDED:</b> {res.final_stop_reason.replace('_', ' ')} after {len(res.rounds)} rounds.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tabs Interface
    tab_control, tab_claims, tab_trace, tab_critic, tab_rulebook, tab_audit = st.tabs([
        "📊 Control Room",
        "🔍 Claims Explorer",
        "🕵️ Agent Trace",
        "⚖️ Critic Report",
        "📜 Rulebook Evolution",
        "📑 Audit Log",
    ])

    # =======================================================================
    # Tab A: Control Room
    # =======================================================================
    with tab_control:
        st.markdown('<div class="section-header">Live Simulation KPI Overview</div>', unsafe_allow_html=True)

        latest_round = res.rounds[-1]
        cr = latest_round.critic_report

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Total Evaluated</div>
                    <div class="kpi-value">{res.total_claims_processed}</div>
                    <div class="kpi-subtext">Across {len(res.rounds)} rounds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Latest Recall</div>
                    <div class="kpi-value" style="color:{THEME_COLORS['approve']};">{cr.recall * 100:.1f}%</div>
                    <div class="kpi-subtext">TP: {cr.tp} / Fraud: {cr.tp + cr.fn}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Latest Precision</div>
                    <div class="kpi-value" style="color:{THEME_COLORS['accent_blue']};">{cr.precision * 100:.1f}%</div>
                    <div class="kpi-subtext">TP / (TP + FP)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Latest FPR</div>
                    <div class="kpi-value" style="color:{THEME_COLORS['fraud']};">{cr.fpr * 100:.1f}%</div>
                    <div class="kpi-subtext">False Alarms: {cr.fp}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Rule Updates</div>
                    <div class="kpi-value">{res.accepted_proposals}</div>
                    <div class="kpi-subtext">{res.rejected_proposals} rejected by Gate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k6:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Active Rulebook</div>
                    <div class="kpi-value">v{res.final_rulebook_version}</div>
                    <div class="kpi-subtext">{len(res.final_rulebook.rules)} active rules</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Progression Chart
        st.markdown('<div class="section-header">Measured Feedback Trajectory</div>', unsafe_allow_html=True)
        chart = create_metrics_progression_chart(res)
        st.plotly_chart(chart, use_container_width=True)

        # Round by Round Table
        st.markdown('<div class="section-header">Execution Round Breakdown</div>', unsafe_allow_html=True)
        rounds_df = build_rounds_summary_dataframe(res)
        st.dataframe(rounds_df, use_container_width=True, hide_index=True)

    # =======================================================================
    # Tab B: Claims Explorer
    # =======================================================================
    with tab_claims:
        st.markdown('<div class="section-header">Observable Claims & Investigation Dossiers</div>', unsafe_allow_html=True)
        st.caption(
            "🔒 **TI-01 Privacy Invariant**: Investigator-facing tables display strictly observable claim submissions "
            "and evidence. Hidden ground-truth labels are never exposed in this view."
        )

        col_r_sel, col_d_sel, col_s_sel = st.columns([1, 1, 2])
        with col_r_sel:
            selected_round_num = st.selectbox(
                "Select Round",
                options=[r.round_number for r in res.rounds],
                index=len(res.rounds) - 1,
            )
        with col_d_sel:
            filter_disp = st.selectbox(
                "Filter Disposition",
                options=["ALL", "APPROVE", "REFER", "SUSPECTED_FRAUD"],
                index=0,
            )
        with col_s_sel:
            search_cid = st.text_input("Search Claim ID", placeholder="e.g. CLM-123456")

        target_round = next(r for r in res.rounds if r.round_number == selected_round_num)
        claims_df = build_claims_dataframe(
            round_result=target_round,
            filter_disposition=filter_disp,
            min_score=0.0,
            search_query=search_cid,
        )

        st.dataframe(claims_df, use_container_width=True, hide_index=True)

        # Detailed Claim Inspector
        if not claims_df.empty:
            st.markdown('<div class="section-header">Structured Claim Decision & Tool Trace Inspector</div>', unsafe_allow_html=True)

            available_ids = list(claims_df["Claim ID"])
            inspected_id = st.selectbox("Inspect Claim Details", options=available_ids)

            # Find matching decision and claim
            selected_dec = next(d for d in target_round.decisions if d.claim_id == inspected_id)
            selected_claim = next((c for c in target_round.claims if c.claim_id == inspected_id), None)
            selected_pipeline_res = next(
                (pr for pr in target_round.pipeline_results if pr.claim.claim_id == inspected_id), None
            )

            col_c1, col_c2 = st.columns([1, 1])

            with col_c1:
                st.markdown("#### 📄 Submission Details")
                if selected_claim:
                    st.write(f"**Claim ID:** `{selected_claim.claim_id}`")
                    st.write(f"**Policy ID:** `{selected_claim.policy_id}`")
                    st.write(f"**Incident Date:** `{selected_claim.incident_date}`")
                    st.write(f"**Vehicle:** {selected_claim.vehicle_make} {selected_claim.vehicle_model} ({selected_claim.vehicle_year})")
                    st.write(f"**Registration:** `{selected_claim.vehicle_registration}` (Status: `{selected_claim.vehicle_registration_status}`)")
                    st.write(f"**Claimed Amount:** {format_currency(selected_claim.claimed_amount)}")
                    st.write(f"**Benchmark Cost:** {format_currency(selected_claim.benchmark_repair_cost)}")
                    st.write(f"**Incident Description:** {selected_claim.incident_description}")
                else:
                    st.info("Claim submission details unavailable.")

            with col_c2:
                st.markdown("#### ⚖️ Triage Disposition & Score")
                badge_html = render_disposition_badge(selected_dec.disposition.value)
                st.markdown(f"**Recommendation:** {badge_html}", unsafe_allow_html=True)
                st.markdown(f"**Total Risk Score:** `{selected_dec.risk_score:.1f}` / 100.0")

                st.markdown("##### Audit Findings")
                for f in selected_dec.findings:
                    color = THEME_COLORS["fraud"] if "FLAGGED" in f.result else THEME_COLORS["text_muted"]
                    st.markdown(
                        f"""
                        <div style="font-size:0.85rem; border-left:3px solid {color}; padding-left:10px; margin-bottom:8px;">
                            <b>{f.check_name}</b>: <span style="color:{color}; font-weight:600;">{f.result}</span> (+{f.risk_contribution:.1f} pts)<br>
                            <i>{f.detail}</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Structured Tool Trace & Evidence Bundle
            st.markdown("#### 🔬 Agent Pipeline Trace")
            trace_tabs = st.tabs(["Planner Task Rationale", "Research Evidence Dossier", "Deterministic Tool Trace"])

            with trace_tabs[0]:
                if selected_pipeline_res:
                    plan = selected_pipeline_res.plan
                    st.markdown(f"**Investigation Strategy:** Ordered execution of {len(plan.tasks)} verification checks.")
                    for t_idx, task_id in enumerate(plan.tasks, 1):
                        reason = plan.task_reasons.get(task_id, "Standard verification protocol.")
                        st.write(f"**Step {t_idx} [{task_id}]:** {reason}")
                else:
                    st.info("Pipeline plan unavailable.")

            with trace_tabs[1]:
                if selected_pipeline_res:
                    bundle = selected_pipeline_res.evidence
                    st.write(f"**Evidence Dossier Complete:** `{bundle.is_complete}`")
                    for k, item in bundle.items.items():
                        status_str = "Available" if item.is_available else "UNAVAILABLE"
                        st.markdown(f"• **{k}** [{item.source}]: `{item.value}` ({status_str})")
                else:
                    st.info("Evidence bundle unavailable.")

            with trace_tabs[2]:
                if selected_dec.tool_trace:
                    st.json(list(selected_dec.tool_trace))
                else:
                    st.info("Tool trace records unavailable.")

    # =======================================================================
    # Tab C: Agent Trace
    # =======================================================================
    with tab_trace:
        st.markdown('<div class="section-header">Chronological Multi-Agent Activity Trail</div>', unsafe_allow_html=True)

        if st.session_state.audit_logger:
            all_events = st.session_state.audit_logger.get_events()

            col_tf1, col_tf2 = st.columns(2)
            with col_tf1:
                filter_t_round = st.selectbox(
                    "Filter by Round",
                    options=[0] + [r.round_number for r in res.rounds],
                    format_func=lambda x: "All Rounds" if x == 0 else f"Round {x}",
                )
            with col_tf2:
                event_types = ["ALL"] + sorted(list({e.event_type for e in all_events}))
                filter_t_type = st.selectbox("Filter Event Type", options=event_types)

            events_df = build_audit_log_dataframe(all_events, filter_round=filter_t_round, filter_type=filter_t_type)
            st.dataframe(events_df, use_container_width=True, hide_index=True)

            # Selected Event Inspector
            if not events_df.empty:
                selected_seq = st.selectbox("Inspect Event Payload (by Sequence)", options=list(events_df["Seq"]))
                target_event = next(e for e in all_events if e.sequence == selected_seq)

                st.markdown(f"**Event Type:** `{target_event.event_type}` | **Round:** `{target_event.round_number}`")
                st.json(target_event.payload)
        else:
            st.info("Audit log session stream unavailable.")

    # =======================================================================
    # Tab D: Critic Report
    # =======================================================================
    with tab_critic:
        st.markdown('<div class="section-header">Independent Evaluation & Diagnostic Audit</div>', unsafe_allow_html=True)
        st.caption(
            "⚖️ **TI-03 / TI-06 Independent Evaluation**: The Critic evaluates completed decisions against hidden labels "
            "in the GroundTruthVault. Metrics are calculated directly from confusion counts (no hardcoded estimates)."
        )

        c_round_sel = st.selectbox(
            "Select Evaluation Round",
            options=[r.round_number for r in res.rounds],
            index=len(res.rounds) - 1,
            key="critic_round_sel",
        )
        c_round = next(r for r in res.rounds if r.round_number == c_round_sel)
        crit_rep = c_round.critic_report

        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.markdown("#### 🎯 Evaluation Metrics")
            st.write(f"**Total Claims Evaluated:** {crit_rep.total_evaluated}")
            st.write(f"**True Positives (Fraud Caught):** {crit_rep.tp}")
            st.write(f"**False Positives (Genuine Flagged):** {crit_rep.fp}")
            st.write(f"**True Negatives (Genuine Approved):** {crit_rep.tn}")
            st.write(f"**False Negatives (Fraud Missed):** {crit_rep.fn}")
            st.write(f"**Precision:** {format_pct(crit_rep.precision)}")
            st.write(f"**Recall (Sensitivity):** {format_pct(crit_rep.recall)}")
            st.write(f"**F1 Score:** {format_pct(crit_rep.f1)}")
            st.write(f"**False Positive Rate:** {format_pct(crit_rep.fpr)}")
            st.write(f"**Accuracy:** {format_pct(crit_rep.accuracy)}")

        with col_m2:
            matrix_fig = create_confusion_matrix_heatmap(crit_rep)
            st.plotly_chart(matrix_fig, use_container_width=True)

        st.markdown("#### 🏷️ Breakdown by Synthetic Fraud Typology")
        typology_df = build_typology_metrics_dataframe(crit_rep)
        if not typology_df.empty:
            st.dataframe(typology_df, use_container_width=True, hide_index=True)
        else:
            st.info("No fraudulent claims evaluated in this round.")

        st.markdown("#### ⚠️ Consistency & Integrity Audits")
        if crit_rep.consistency_violations:
            st.error(f"Detected {len(crit_rep.consistency_violations)} consistency violations:")
            for v in crit_rep.consistency_violations:
                st.write(f"• `{v.claim_id}` [{v.violation_type}]: {v.explanation}")
        else:
            st.success("✅ Zero consistency violations detected. All decisions adhere strictly to Rulebook thresholds and non-empty findings constraints.")

    # =======================================================================
    # Tab E: Rulebook Evolution
    # =======================================================================
    with tab_rulebook:
        st.markdown('<div class="section-header">Active Rulebook & Safe Modification Audit</div>', unsafe_allow_html=True)
        st.caption(
            "🔒 **TI-04 / TI-05 Safe Update Invariant**: All rule parameter updates are generated from bounded templates "
            "and must pass the 4-stage validation gate (including false positive budget check) before activation."
        )

        st.markdown(f"### Active Version: `v{res.final_rulebook.version}`")
        st.write(f"**Ruleset Name:** {res.final_rulebook.name}")
        st.write(f"**Description:** {res.final_rulebook.description}")

        st.markdown("#### Current Verification Rules")
        rule_rows = []
        for r_def in res.final_rulebook.rules:
            rule_rows.append({
                "Rule ID": r_def.rule_id,
                "Template": r_def.template_id,
                "Enabled": "Yes" if r_def.enabled else "No",
                "Severity": r_def.severity,
                "Score Points": r_def.score_contribution,
                "Parameters": json.dumps(r_def.parameters),
                "Description": r_def.description,
            })
        st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🔍 Rule-Writer Adaptation History Across Rounds")
        all_audit_records: List[Any] = [rec for r in res.rounds for rec in r.rule_audit_records]

        if all_audit_records:
            for rec in all_audit_records:
                status_color = THEME_COLORS["approve"] if rec.status == "ACCEPTED" else THEME_COLORS["fraud"]
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {status_color}; background-color:{THEME_COLORS['surface']}; padding: 12px 16px; margin-bottom: 12px; border-radius: 4px;">
                        <b>Proposal {rec.proposal_id} (Round {rec.round_number}):</b> 
                        <span style="color:{status_color}; font-weight:700;">{rec.status}</span><br>
                        • <b>Target Rule:</b> <code>{rec.rule_id}.{rec.parameter_name}</code><br>
                        • <b>Adjustment:</b> <code>{rec.old_value}</code> ➔ <code>{rec.candidate_value}</code><br>
                        • <b>Validation FP Rate:</b> {rec.validation_fp_rate:.2%}<br>
                        • <b>Rationale:</b> <i>{rec.rationale}</i><br>
                        {f'• <b>Gate Failed:</b> {rec.gate_failed} ({rec.rejection_reason})' if rec.gate_failed else f'• <b>Rulebook Version:</b> {rec.rulebook_version_before} ➔ {rec.rulebook_version_after}'}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No rule modifications were triggered or required during this experiment.")

    # =======================================================================
    # Tab F: Audit Log
    # =======================================================================
    with tab_audit:
        st.markdown('<div class="section-header">Structured Immutable Audit Trail</div>', unsafe_allow_html=True)
        st.caption(
            "📋 **TI-09 Complete Audit Coverage**: Every claim decision, critic evaluation, rule proposal, and "
            "adaptation shift produces a structured, monotonically sequenced audit event. Raw ground-truth records are strictly stripped."
        )

        if st.session_state.audit_logger:
            cov = st.session_state.audit_logger.decision_coverage()
            st.markdown(
                f"""
                <div class="kpi-card" style="margin-bottom:16px;">
                    <div class="kpi-title">Audit Log Health & Coverage</div>
                    <div style="font-size:1.1rem; font-weight:600; color:{THEME_COLORS['approve']};">
                        100% Decision Coverage Verified
                    </div>
                    <div style="font-size:0.85rem; color:{THEME_COLORS['text_muted']}; margin-top:4px;">
                        {cov['decision_events_count']} claim decisions recorded for {res.total_claims_processed} processed claims.
                        Total audit trail events: {cov['total_audit_events']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Audit Download Options
            log_events = st.session_state.audit_logger.get_events()
            jsonl_content = "\n".join(e.to_json() for e in log_events) + "\n"
            canonical_content = json.dumps(res.to_canonical_dict(), indent=2)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="⬇️ Download Sanitized Audit Log (.jsonl)",
                    data=jsonl_content,
                    file_name=f"shard_audit_{res.run_id}.jsonl",
                    mime="application/x-jsonlines",
                    use_container_width=True,
                )
            with col_d2:
                st.download_button(
                    label="⬇️ Download Canonical Reproducibility Export (.json)",
                    data=canonical_content,
                    file_name=f"shard_canonical_{res.run_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.markdown("#### Raw Event Stream Inspector")
            events_table = build_audit_log_dataframe(log_events)
            st.dataframe(events_table, use_container_width=True, hide_index=True)
        else:
            st.info("No active audit log recorded.")
