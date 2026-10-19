# SHARD — 5-Minute Live Demonstration Script & Operator Walkthrough

> **Document Type**: Technical Demonstration Guide & Examiner Viva Walkthrough  
> **Target Audience**: CA-3 Foundations of Agentic AI Assessors and Examiners  
> **Target Duration**: Exactly 5 Minutes (0:00 – 5:00)  
> **Application URI**: `http://localhost:8501` (Streamlit Control Room)  
> **Environment**: Pure Offline Python 3.14 (0 paid APIs, 0 cloud dependencies, 0 network calls)

---

## 1. Preflight Verification Checklist (T-Minus 2 Minutes)

Before opening the demonstration to examiners, verify local readiness:

1. **Terminal 1 (Test Suite Smoke Test)**:
   ```powershell
   python -m pytest tests/test_orchestrator.py -v
   ```
   *Expected Outcome*: All 20 tests pass in ~4 seconds.

2. **Terminal 2 (Launch Streamlit Control Room)**:
   ```powershell
   streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
   ```
   *Expected Outcome*: Application starts and renders at `http://localhost:8501`.

3. **Browser**:
   - Open `http://localhost:8501`.
   - Ensure the browser window is maximized (16:9 aspect ratio recommended).
   - Verify all 6 tabs are visible: **Control Room**, **Claims Explorer**, **Investigation Traces**, **Critic Evaluation**, **Rulebook Evolution**, **Audit Trail**.

---

## 2. Timed 5-Minute Demonstration Protocol

```
+----------------------------------------------------------------------------------------------------+
| 0:00 - 0:45 | Phase 1: Context, Problem Statement & Architecture Overview                          |
| 0:45 - 1:45 | Phase 2: Launching Simulation & Control Room Metrics (Seed 42, 30 Claims/Round)     |
| 1:45 - 2:45 | Phase 3: Deep-Dive on Claims Explorer & Verification Traces (Approve vs Fraud)       |
| 2:45 - 3:45 | Phase 4: Critic Report, Confusion Matrix & Trust Boundary Enforcement                |
| 3:45 - 4:30 | Phase 5: Adaptive Loop, Rule-Writer & Gate B False-Positive Budget Safeguard         |
| 4:30 - 5:00 | Phase 6: Tamper-Evident Audit Trail & Conclusion                                      |
+----------------------------------------------------------------------------------------------------+
```

---

### [0:00 – 0:45] Phase 1: Executive Context & Structural Problem

#### Operator Script:
> "Good morning examiners. Today I am demonstrating **SHARD**, an adversarial multi-agent insurance claims auditor built for the CA-3 Foundations of Agentic AI assignment.
>
> In motor insurance fraud detection, insurers face a structural dilemma: static rules degrade because organised fraud syndicates adapt to evade known thresholds. Worse, traditional systems lack pre-deployment safeguards to test whether a proposed rule update will flood adjusters with false positives.
>
> SHARD solves this with a **closed-loop adversarial multi-agent architecture**:
> - An adversarial **Fraudster Agent** dynamically adapts synthetic attack vectors across rounds.
> - An **Investigation Pipeline** (Planner, Researcher, Investigator) evaluates claims using 6 deterministic verification tools with zero ground-truth knowledge.
> - An independent **Critic Agent** queries a sealed Ground-Truth Vault to measure exact precision, recall, and missed typologies.
> - A **Rule-Writer Agent** proposes bounded parameter changes that must pass a **Multi-Gate Safety Validator (including Gate B False-Positive Budget Check)** before deployment.
>
> Let's observe this executing live in the Streamlit Control Room."

#### Visual Action:
- Stand on the **Control Room** tab. Point out the sidebar configuration panel.

---

### [0:45 – 1:45] Phase 2: Launching the Simulation & Control Room Overview

#### Visual Actions:
1. Direct the examiners' attention to the left sidebar:
   - **Base Random Seed**: Ensure set to `42` (guarantees bit-identical reproducibility under TI-07).
   - **Claims per Round**: Set to `30`.
   - **Max Simulation Rounds**: Set to `3`.
   - **Target Recall**: `0.85` (85%).
   - **Max Allowed FPR**: `0.10` (10%).
   - **FP Budget Delta**: `0.05` (5% maximum allowed surge in false alarms).
2. Click the primary button: **`Execute Adversarial Loop`**.
3. Watch the progress bar advance through Round 1, Round 2, and Round 3.

#### Operator Script:
> "I have initialized the orchestrator with fixed seed 42 and 30 claims per round. Notice that execution is fast because SHARD relies on deterministic verification tools and templates—no slow, non-deterministic LLM API calls or GPU dependencies.
>
> As execution finishes, the Control Room immediately populates with live session metrics:
> - **Total Claims Audited**: Exactly 90 claims processed across 3 rounds.
> - **Findings Recorded**: 540 discrete evidentiary findings (100% decision coverage under invariant TI-02).
> - **Overall Precision**: 100.0%.
> - **False-Positive Rate**: 0.00%—not a single legitimate claimant was wrongly accused.
>
> Crucially, every number displayed in this UI is dynamically bound to the executed pipeline in session state (invariant TI-12). Zero metrics are hardcoded."

---

### [1:45 – 2:45] Phase 3: Claims Explorer & Agent Trace Deep-Dive

#### Visual Actions:
1. Click the **Claims Explorer** tab.
   - Point out the claim status badges: `APPROVE` (Green), `REFER` (Amber), `SUSPECTED_FRAUD` (Crimson).
   - Point out that **Ground-Truth labels are completely absent** from this table (confirming invariant TI-01).
2. Select claim **`CLM-877572`** (Clean Genuine Claim):
   - Show Risk Score: **`0.0 / 100`** -> Disposition: **`APPROVE (Fast-track)`**.
   - Show the 6 verification tool results: all clear (valid DVLA registration, no write-off history, no prior damage, normal repair markup).
3. Now filter or select claim **`CLM-781453`** (Organised Fraud Claim):
   - Show Risk Score: **`65.0 / 100`** -> Disposition: **`SUSPECTED_FRAUD`**.
4. Click the **Investigation Traces** tab for `CLM-781453`:
   - Expand the **Planner Agent Trace**: Show the ordered investigation plan and adjuster rationale.
   - Expand the **Research Agent Dossier**: Show synthetic evidence retrieved from DVLA and MIAFTR.
   - Expand the **Investigator Agent Tool Trace**:
     * `RULE_STAGED_COLLISION_INDICATORS`: **+35.0 pts** (Short policy tenure of 2 days post-inception, 2 linked parties, flagged repair provider `PRV-FLAG-903`).
     * `RULE_REPAIR_BENCHMARK`: **+30.0 pts** (Claimed repair £2,840.51 exceeds benchmark £2,057.86 by 38.0% vs max 35.0% threshold).
     * Total Score: **65.0 pts**.

#### Operator Script:
> "Let's inspect how individual claims travel the multi-agent pipeline.
>
> First, look at **CLM-877572**: it is a genuine collision claim. The tools verify that the DVLA tax is active, the vehicle has never been written off, and the garage quote matches industry benchmarks. Its risk score is 0.0, enabling straight-through fast-track approval without manual adjuster friction.
>
> Next, look at **CLM-781453**: the Planner Agent detected a high-value claim filed just 2 days after policy inception. It planned targeted checks. The Investigator executed deterministic tools, uncovering an unapproved 38% markup from a flagged provider network. The additive risk score reaches 65.0, triggering an automatic referral for suspected fraud.
>
> Notice that every single decision includes statutory, itemized findings. Invariant TI-02 forbids empty findings."

---

### [2:45 – 3:45] Phase 4: Critic Report, Confusion Matrix & Trust Boundary

#### Visual Actions:
1. Click the **Critic Evaluation** tab.
2. Direct attention to the **Interactive Confusion Matrix Heatmap**:
   - Round 1: TP=9, FP=0, TN=21, FN=0.
   - Round 2: TP=9, FP=0, TN=21, FN=0.
   - Round 3: TP=9, FP=0, TN=21, FN=0.
3. Review the **Per-Typology Detection Breakdown**:
   - Staged Collision: 100.0%
   - Exaggerated Repair: 100.0%
   - Ghost Vehicle: 100.0%
   - Pre-Existing Damage: 100.0%
   - Opportunistic Inflation: 100.0%
4. Highlight the **Redacted Diagnostic Summary**:
   - Point out that only aggregate miss counts are passed back; no individual claim IDs or raw ground-truth records ever cross the boundary.

#### Operator Script:
> "Now we enter the **Evaluation Zone**. This is where architectural rigor separates SHARD from naive agent systems.
>
> Under invariant TI-01, the Planner, Researcher, and Investigator operate in complete isolation from the Ground-Truth Vault. Only the **Critic Agent** is authorized to read the vault. This isolation is structurally verified by our automated AST test suite: if any investigation module attempts to import `vault.py`, the build immediately fails.
>
> The Critic evaluates actual predictions against hidden labels, computing standard statistical metrics: Precision, Recall, F1, and False-Positive Rate.
>
> In our primary benchmark, baseline rules caught all 27 fraudulent claims across the 3 rounds with zero false alarms. But what happens when fraud schemes evolve and evade initial rules? That brings us to our adaptive feedback loop."

---

### [3:45 – 4:30] Phase 5: Adaptive Feedback Loop & Gate B Safety Check

#### Visual Actions:
1. Click the **Rulebook Evolution** tab.
2. Explain the **Multi-Gate Safety Validator**:
   - Point out **Gate A** (Schema & Parameter Bounds), **Gate B** (False-Positive Budget), **Gate C** (Golden Set Non-Regression), and **Gate D** (Diagnostic Justification).
3. Walk through the **Rulebook Diff View**:
   - In the standard baseline run, all fraud was caught, so 0 proposals were needed.
   - Explain the documented **Adaptive Calibration run** (with relaxed initial markup 1.60):
     * When the Fraudster attacks evasive line-item padding, recall drops (to 75.0% in Round 2).
     * The Critic emits an aggregate miss summary diagnosing 3 missed opportunistic inflation claims.
     * The **Rule-Writer Agent** proposes tightening `RULE_LINE_ITEM_ARITHMETIC.max_discrepancy_gbp` from 1.0 to 0.5.
     * **Gate B False-Positive Budget Check** re-scores this candidate rule against 40 fixed genuine validation claims (fraud_share=0.0, seed 9999). Because the candidate false-alarm rate is 0.0% (within the 10% budget), the proposal is **ACCEPTED**.
     * The Rulebook version advances from `v1.0.0` to `v1.1.0`.
     * In the subsequent round, detection on the new cohort recovers to 83.33% and F1 climbs to 0.909!
4. Highlight what happens if an over-aggressive rule is proposed:
   - "To demonstrate Gate B rejection, we point to our automated unit test fixture (`test_candidate_exceeding_fp_budget_rejected` in `tests/test_rule_writer.py`), where an over-aggressive rule that flags innocent claims is immediately rejected, preserving rulebook stability."

#### Operator Script:
> "In the Rulebook Evolution tab, you see SHARD's closed-loop adaptation.
>
> Crucially, SHARD does not permit unconstrained code generation or `eval()` execution. That would be catastrophic in a regulated insurance environment. Instead, adaptation is strictly parametric via pre-approved templates (invariant TI-04).
>
> Even more importantly, every proposal must pass **Gate B** (invariant TI-05): the candidate rule is re-scored against a fixed dataset of 40 genuine claims. If the false-positive surge exceeds the budget, the change is programmatically rejected. This provides an empirical safeguard ensuring aggressive fraud controls do not degrade customer experience on standard claims."

---

### [4:30 – 5:00] Phase 6: Structured Audit Trail & Viva Conclusion

#### Visual Actions:
1. Click the **Audit Trail** tab.
2. Show the structured JSON-lines event stream:
   - Event types: `CLAIM_DECISION`, `CRITIC_REPORT`, `RULE_PROPOSAL`, `RULE_VALIDATION`, `ROUND_SUMMARY`.
   - Point out that every event contains a UTC timestamp, monotonic sequence number, and sanitized payload (with hidden labels stripped under TI-01).
3. Show the **Export Audit Log** and **Download Canonical Summary** buttons.

#### Operator Script:
> "Finally, let's view the **Audit Trail** tab. Under invariant TI-09, every decision, tool execution, and rule proposal produces a structured, append-only JSON Lines event with monotonic sequence tracking. In our benchmark run, 100% of claims produced an audit record.
>
> In conclusion:
> - SHARD runs **100% locally and deterministically** (146 automated tests pass across 10 modules in ~34 seconds with zero paid APIs or remote keys).
> - It features **true multi-agent division of labor** with proven ground-truth isolation.
> - Its **adaptive adversarial loop** safely tightens controls without unconstrained code generation.
>
> Thank you, and I look forward to your questions."

---

## 3. Emergency Fallback & Troubleshooting Procedures

If encountering any unexpected environment or demonstration anomaly during the live viva:

### Scenario A: Port 8501 is Already Occupied
- **Action**: Run Streamlit on alternative port 8502:
  ```powershell
  streamlit run app/streamlit_app.py --server.port 8502
  ```

### Scenario B: Browser Session Desynchronized
- **Action**: Click the **`Reset Entire Experiment`** button in the Control Room sidebar, or refresh the browser with `Ctrl + F5`. The application will cleanly re-initialize state without corrupting the on-disk audit log.

### Scenario C: Offline CLI Headless Fallback
- **Action**: If the browser UI is unavailable, execute the canonical benchmark runner directly in PowerShell:
  ```powershell
  $env:PYTHONPATH='.'; python -c "from shard.orchestrator import Orchestrator, OrchestratorConfig; res = Orchestrator(OrchestratorConfig(base_seed=42, n_claims_per_round=30, max_rounds=3, early_stopping=False)).run(); print('Status:', res.final_stop_reason, '| Total Claims:', res.total_claims_processed, '| Final Version:', res.final_rulebook_version)"
  ```
  *Output*: Prints verified execution metrics in under 3 seconds.
