# SHARD — Comprehensive Viva Q&A & Technical Defense Guide

> **Document Type**: Academic & Technical Viva Examination Preparation  
> **Course**: CA-3 Foundations of Agentic AI  
> **System**: SHARD (Adversarial Multi-Agent Insurance Claims Auditor)  
> **Author**: Staff Software Engineer & Technical Lead  
> **Verification Status**: 100% Backed by Verified Code & Passing Automated Tests (146/146)

---

## Table of Contents

1. [Domain & Problem Formulation (Q1–Q3)](#1-domain--problem-formulation)
2. [Multi-Agent Architecture & Division of Labor (Q4–Q6)](#2-multi-agent-architecture--division-of-labor)
3. [Deterministic Verification Tools & Triage Logic (Q7–Q9)](#3-deterministic-verification-tools--triage-logic)
4. [Critic Agent, Evaluation & Metrics (Q10–Q12)](#4-critic-agent-evaluation--metrics)
5. [Adversarial Adaptation & Safe Rule Writing (Q13–Q15)](#5-adversarial-adaptation--safe-rule-writing)
6. [Multi-Gate Safety Validator & False-Positive Budgets (Q16–Q19)](#6-multi-gate-safety-validator--false-positive-budgets)
7. [Orchestration, Reproducibility & Audit Logging (Q20–Q22)](#7-orchestration-reproducibility--audit-logging)
8. [UI Data Binding, Limitations & Enterprise Roadmap (Q23–Q25)](#8-ui-data-binding-limitations--enterprise-roadmap)

---

## 1. Domain & Problem Formulation

### Q1: Why motor insurance fraud? Why is this domain suited to an agentic multi-agent architecture?
**Answer**:
Motor insurance fraud is an adversarial, multi-faceted domain characterised by structured, verifiable evidence (DVLA registrations, garage repair invoices, salvage markers, collision topologies) combined with intelligent, adaptive human fraudsters. According to the Association of British Insurers (ABI), detected motor insurance fraud exceeds £600 million annually in the UK alone.

The domain is exceptionally well suited to an agentic multi-agent architecture for three core reasons:
1. **Separation of Investigative Concerns**: Investigating a motor claim requires heterogeneous, specialized sub-tasks: inspecting claim features to construct an investigation plan, querying third-party vehicle databases for factual evidence, executing deterministic mathematical arithmetic on invoice items, and assigning triage dispositions. Distributing these to specialized agents mimics real-world Special Investigation Units (SIUs).
2. **Adversarial Dynamics**: Fraud is not a static classification target; it is a game-theoretic cat-and-mouse dynamic. When insurers deploy a rule to catch a specific pattern (e.g., claims over £2,000), organised syndicates adapt their claiming behavior (e.g., claiming £1,950). A closed-loop multi-agent system models both sides of this game: an adversarial Fraudster Agent that probes weaknesses and an adaptive Rule-Writer Agent that tightens controls.
3. **Rigorous Asymmetric Information**: In real investigations, ground-truth fraud is unknown to the adjuster. An agentic architecture enables formal trust boundaries: investigation agents process only observable claimant data, while an independent Critic Agent queries sealed truth to evaluate system performance without contaminating the investigation.

### Q2: Why is a single LLM prompt or static pipeline insufficient for this problem?
**Answer**:
A single LLM prompt or static linear pipeline suffers from fundamental structural failure modes when applied to insurance claims auditing:
- **Hallucination & Legal Non-Defensibility**: LLMs generate probabilistic text. In insurance claims triage, denying a claim or accusing a policyholder of fraud based on hallucinated evidence violates statutory fair-treatment regulations (e.g., UK Financial Conduct Authority FG21/1 guidelines). The system must produce deterministic, inspectable evidence.
- **Inability to Self-Calibrate Safely**: A single prompt cannot evaluate its own false-positive surge before changing its decision criteria. If an LLM prompt is asked to "be stricter on repair costs", it may unconstrainedly flag legitimate claimants, causing an operational disaster.
- **Evaluation Contamination (Data Leakage)**: A single prompt that generates synthetic test cases and evaluates them inevitably leaks ground-truth labels into the decision context.
- **Static Pipeline Rigidity (Structural Rule Lag)**: A traditional static script cannot adapt to new evasion vectors without manual human re-programming. By the time human analysts notice new fraud patterns, millions of pounds in fraudulent claims have already been paid out.

### Q3: How is ground-truth label isolation enforced, and how is it tested?
**Answer**:
Ground-truth isolation is enforced architecturally through Technical Invariant **TI-01**:
1. **Architectural Isolation**: At data generation time (`shard/data/generator.py`), the generator splits output into two completely distinct objects:
   - Observable `Claim` objects: Contain only observable customer data (`claim_id`, `incident_type`, `claimed_amount`, `vehicle_reg`, `provider_id`, etc.). The boolean `is_fraud` field does not exist on the `Claim` class.
   - `GroundTruthRecord` objects: Stored solely in `GroundTruthVault` (`shard/data/vault.py`).
2. **Strict Access Whitelist**: Only the `CriticAgent` (for evaluation) and `Orchestrator` (for stopping condition checks) are permitted to import or call `vault.py`. The `Planner`, `Researcher`, `Investigator`, `Fraudster`, `RuleWriter`, and deterministic tools never receive a reference to the vault.
3. **Automated AST & Runtime Enforcement**: Isolation is verified by automated Abstract Syntax Tree (AST) lint tests in `tests/test_gt_isolation.py` and `tests/test_critic.py`. The test suite parses the Python AST of every agent and rule module; if any unauthorized module imports `shard.data.vault` or inspects `is_fraud`, the test suite fails immediately.

---

## 2. Multi-Agent Architecture & Division of Labor

### Q4: What are the 7 agents, what is each agent's single responsibility, and what are their inputs and outputs?
**Answer**:
The SHARD system coordinates 7 distinct, specialized agents:

| Agent | Module | Single Responsibility | Inputs | Outputs |
|---|---|---|---|---|
| **Fraudster** | `shard/agents/fraudster.py` | Adversarially adjusts typology sampling weights across rounds based on previous detection gaps. | Redacted `CriticFeedbackSummary` (aggregate miss counts only). | Updated `GeneratorConfig` (typology sampling distribution). |
| **Generator** | `shard/data/generator.py` | Produces seeded batches of realistic synthetic motor claims and writes true labels to the sealed vault. | `GeneratorConfig`, random seed. | Tuple of observable `Claim` objects, populates `GroundTruthVault`. |
| **Planner** | `shard/agents/planner.py` | Inspects observable claim features and formulates an ordered list of verification checks with rationale. | Single observable `Claim`. | `InvestigationPlan` (ordered list of checks with rationale). |
| **Researcher** | `shard/agents/researcher.py` | Non-scoring information retrieval from synthetic DVLA, MIAFTR, and repair registries. | Observable `Claim`, `InvestigationPlan`. | `EvidenceBundle` (typed registry records & documents). |
| **Investigator** | `shard/agents/investigator.py` | Executes deterministic verification tools against evidence and applies versioned Rulebook thresholds. | Observable `Claim`, `EvidenceBundle`, `Rulebook`. | `ClaimDecision` (risk score, disposition, recorded findings, trace). |
| **Critic** | `shard/agents/critic.py` | Authorised evaluation against sealed Ground-Truth Vault; calculates metrics and audits internal consistency. | List of `ClaimDecision` objects, `GroundTruthVault`. | `CriticReport`, redacted `CriticFeedbackSummary`. |
| **Rule-Writer** | `shard/agents/rule_writer.py` | Proposes bounded parameter updates from predefined templates to target diagnosed evasion patterns. | `CriticFeedbackSummary`, current `Rulebook`. | `RuleProposal` submitted to Multi-Gate Safety Validator. |

### Q5: How does the Planner Agent decide which verification steps to run?
**Answer**:
The Planner Agent (`shard/agents/planner.py`) inspects the observable attributes of a `Claim` and maps risk triggers to a catalogue of 6 verification steps:
- **`VERIFY_REPAIR_COST` & `CHECK_LINE_ITEMS`**: Triggered if claimed amount exceeds £1,500 or if the provider ID matches an independent garage pattern.
- **`VERIFY_VEHICLE` & `CHECK_WRITE_OFF`**: Triggered for all collision, theft, or fire claims to establish vehicle existence and check MIAFTR salvage markers.
- **`CHECK_PRIOR_DAMAGE`**: Triggered if vehicle age is > 3 years or claimed amount > £2,500.
- **`FLAG_NETWORK_CLUSTER`**: Triggered if multiple parties are involved in a collision or if policy tenure is under 30 days.

For each selected step, the Planner generates a structured `PlanStep` containing the step identifier, priority, and an explicit human-readable rationale (e.g., *"Policy tenure is under 30 days; verify collision network graph for collusive staged accidents"*). This rationale provides full transparency to human claims adjusters.

### Q6: What is the role of the Research Agent, and why doesn't it make decisions?
**Answer**:
The Research Agent (`shard/agents/researcher.py`) acts strictly as an evidentiary data gatherer. Given a `Claim` and an `InvestigationPlan`, it queries synthetic registry backends to assemble an `EvidenceBundle`:
1. **DVLA Vehicle Registry**: Verifies registration string format, vehicle age, tax status, and MOT validity.
2. **MIAFTR Total Loss Register**: Queries whether the vehicle chassis has prior Category A (scrap only), Category B (break for parts), or Category S/N salvage flags.
3. **Prior Damage Registry**: Retrieves historical damage logs to check for overlapping body panels.
4. **Repair Invoice Registry**: Assembles itemized quotes and line items from the repair garage.

**Why it doesn't make decisions**:
In real-world compliance and forensics, information retrieval must be decoupled from adjudication. If the research layer performed scoring or filtering, it could introduce silent confirmation bias or opaque heuristics. By restricting the Researcher to compiling an objective, non-scoring `EvidenceBundle`, the downstream Investigator can apply explicit, version-controlled rules that can be audited by regulators.

---

## 3. Deterministic Verification Tools & Triage Logic

### Q7: Explain the 6 deterministic verification tools. What does each check?
**Answer**:
Implemented in `shard/rules/tools.py`, all 6 tools are deterministic Python functions taking a `Claim` and `EvidenceBundle` and returning a typed `ToolResult`:
1. **`verify_repair_benchmark`**: Calculates the ratio of the claimed repair quote to an independent make/model benchmark repair cost. Flags an anomaly if `markup_ratio > max_markup_ratio` (default 1.35x).
2. **`check_line_items`**: Performs arithmetic verification over all invoice items (labor hours × rate + parts prices + paint). Flags discrepancies where sum ≠ total by > £1.00, or flags unrelated items (e.g., alloy detailing on a front bumper repair).
3. **`verify_vehicle_registration`**: Cross-references vehicle registration against DVLA synthetic records. Flags unregistered vehicles, year mismatches, or expired road tax.
4. **`check_write_off_salvage`**: Queries MIAFTR database for previous total-loss write-off markers. Flags vehicles with Category A or B salvage markers that should legally never be back on the road.
5. **`check_prior_damage_reported`**: Checks whether claimant reported prior structural damage or if registry logs indicate pre-existing damage on the exact same panels currently claimed.
6. **`flag_network_cluster`**: Analyzes the claimant-provider-garage network graph. Flags instances where policy tenure is < 10 days, multiple claimants share an address, or a garage has a high collision clustering density.

### Q8: How does the Investigator Agent compute risk scores, and what are the disposition thresholds?
**Answer**:
The Investigator Agent (`shard/agents/investigator.py`) applies the active `Rulebook` against the results of the executed verification tools.

**Risk Scoring Formula**:
Scoring is bounded between `0.0` and `100.0`. Each triggered rule contributes an additive point value defined in `rulebook_v1.json`:
$$\text{Risk Score} = \min\left(100.0, \sum_{r \in \text{Triggered Rules}} \text{Score Contribution}(r)\right)$$

*Example Score Contributions*:
- `RULE_WRITE_OFF_STATUS`: +40.0 pts (Critical)
- `RULE_VEHICLE_REGISTRATION`: +35.0 pts (High)
- `RULE_STAGED_COLLISION_INDICATORS`: +35.0 pts (High)
- `RULE_REPAIR_BENCHMARK`: +30.0 pts (High)
- `RULE_PRIOR_DAMAGE`: +25.0 pts (Medium)
- `RULE_LINE_ITEM_ARITHMETIC`: +20.0 pts (Medium)

**Disposition Thresholds**:
- **`< 25.0`**: **`APPROVE`** — Low risk. Fast-track straight-through claims settlement without human adjuster intervention.
- **`25.0 – 59.9`**: **`REFER`** — Intermediate risk. Routed to a human claims adjuster for manual document inspection (satisfies Technical Invariant TI-08).
- **`>= 60.0`**: **`SUSPECTED_FRAUD`** — High risk. Escalated to the Special Investigation Unit (SIU) for formal fraud inquiry.

### Q9: Why are triage dispositions (APPROVE/REFER/SUSPECTED_FRAUD) decoupled from ground truth (TI-11)?
**Answer**:
Technical Invariant **TI-11** explicitly requires that the system's output is an **investigation recommendation**, never a judicial verdict.
In actual insurance operations, an automated software agent cannot declare a customer a "confirmed fraudster". Labeling a claim as "Fraud" would expose the insurer to severe legal defamation liability and breach the Insurance Act 2015.
Instead, SHARD outputs operational recommendations:
- The Investigator assigns an operational triage recommendation (`APPROVE`, `REFER`, `SUSPECTED_FRAUD`) based strictly on observable evidence.
- The ground-truth label (`is_fraud: True/False`) resides exclusively in the evaluation vault.
Decoupling these concepts ensures that the system serves as an auditable decision-support tool, not an unaccountable autonomous judge.

---

## 4. Critic Agent, Evaluation & Metrics

### Q10: What is the role of the Critic Agent, and how does it prevent evaluation leakage?
**Answer**:
The Critic Agent (`shard/agents/critic.py`) is the independent evaluation and quality assurance component of the architecture. It receives the batch of `ClaimDecision` objects produced during a round, retrieves the true labels from `GroundTruthVault`, and performs comprehensive performance benchmarking.

**Preventing Evaluation Leakage**:
1. **Unidirectional Feedback**: The Critic never shares individual ground-truth records or claim IDs with the Fraudster or Rule-Writer.
2. **Redacted Aggregate Feedback**: The Critic outputs a `CriticFeedbackSummary` that contains strictly aggregate counts and detection rates (e.g., *"3 missed cases of OPPORTUNISTIC_INFLATION"*).
3. **Non-Destructive Invariant**: The Critic never modifies or mutates the Investigator's decisions or scores.
4. **Isolated Evaluation**: Downstream adaptive agents only see what typologies were missed in aggregate, ensuring that the Rule-Writer cannot "memorize" specific claim IDs.

### Q11: How are precision, recall, F1, and FPR calculated? What are the exact formulas?
**Answer**:
The Critic defines fraud detection as a binary classification task where `SUSPECTED_FRAUD` is the positive prediction class, and `is_fraud: True` is the positive ground-truth label:
- **True Positive (TP)**: Claim flagged as `SUSPECTED_FRAUD` and true label `is_fraud == True`.
- **False Positive (FP)**: Claim flagged as `SUSPECTED_FRAUD` and true label `is_fraud == False` (False Alarm).
- **True Negative (TN)**: Claim disposition `APPROVE` or `REFER` and true label `is_fraud == False`.
- **False Negative (FN)**: Claim disposition `APPROVE` or `REFER` and true label `is_fraud == True` (Missed Fraud).

**Exact Mathematical Formulas**:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
$$\text{False-Positive Rate (FPR)} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

All formulas handle edge cases (e.g., $\text{TP} + \text{FP} = 0$) gracefully by returning `0.0` or `1.0` without throwing division-by-zero exceptions.

### Q12: What are internal consistency violations, and how does the Critic detect them?
**Answer**:
Implemented in `shard/agents/critic.py::check_decision_consistency()`, the Critic audits every `ClaimDecision` for internal logical coherence:
1. **Empty Findings Violation**: Any claim marked `SUSPECTED_FRAUD` or `REFER` that possesses an empty list of evidentiary findings (`len(findings) == 0`). This violates invariant TI-02.
2. **Threshold Mismatch**: Any claim whose assigned disposition does not match its calculated risk score (e.g., score = 75.0 but disposition = `APPROVE`).
3. **Negative Score Contribution**: Any finding with a negative or NaN score contribution.
4. **Contradictory Evidence**: E.g., vehicle marked invalid by DVLA check, yet risk contribution is 0.

If any violation is detected, the Critic logs a structured `ConsistencyViolation` record in the `CriticReport`. In all 146 automated tests and our 90-claim benchmark run, **zero consistency violations were recorded**.

---

## 5. Adversarial Adaptation & Safe Rule Writing

### Q13: How does the Fraudster Agent adapt its attack strategy without seeing ground-truth labels?
**Answer**:
The Fraudster Agent (`shard/agents/fraudster.py`) models an organised crime syndicate probing insurer controls. Under invariant **TI-01**, the Fraudster is forbidden from accessing the `GroundTruthVault`.
Instead, it receives the `CriticFeedbackSummary` containing only aggregate miss counts per typology:
$$\text{missed\_typology\_counts} = \{\text{"OPPORTUNISTIC\_INFLATION"}: 3, \text{"EXAGGERATED\_REPAIR"}: 0, \dots\}$$
Because the Fraudster knows which typologies successfully evaded detection in aggregate, it increases the probability of generating those evasive typologies in subsequent rounds. This creates genuine adversarial pressure without leaking individual claim records.

### Q14: What mathematical mechanism does the Fraudster use to update typology weights?
**Answer**:
The Fraudster uses a **Dirichlet-smoothed Bayesian weight adjustment** algorithm with bounded normalization:
1. **Raw Weight Calculation**:
   For each typology $t$, the new unnormalized weight $w'_t$ is adjusted proportional to its empirical miss count:
   $$w'_t = w_t \times \left(1.0 + \alpha \cdot \frac{m_t + 1}{\sum_j (m_j + 1)}\right)$$
   where $w_t$ is the current weight, $\alpha$ is `adaptation_strength` (default 0.15), and $m_t$ is the count of missed fraud cases for typology $t$. The $+1$ term provides Laplace smoothing to prevent zero-probability lock-in.
2. **Clamping & Bounded Normalization**:
   To prevent single-typology monopolies, all weights are strictly clamped between `[0.05, 0.60]`:
   $$w''_t = \text{clamp}(w'_t, 0.05, 0.60)$$
   $$w^{\text{final}}_t = \frac{w''_t}{\sum_j w''_j}$$
This ensures all 5 fraud typologies remain represented in every synthetic batch while elevating evasive vectors.

### Q15: How does the Rule-Writer Agent propose rule changes, and why is eval()/exec() forbidden?
**Answer**:
The Rule-Writer Agent (`shard/agents/rule_writer.py`) inspects the Critic's diagnostic miss summary and proposes targeted parameter changes to tighten controls.

**Why `eval()` / `exec()` is Forbidden (TI-04)**:
Allowing an automated AI agent to generate dynamic Python code strings and execute them via `eval()` or `exec()` introduces catastrophic vulnerabilities:
- **Code Injection & Arbitrary Execution**: Risk of executing untrusted or malicious system commands.
- **Runaway Logic**: Dynamic code can introduce infinite loops or memory leaks.
- **Loss of Regulatory Auditability**: Regulators cannot inspect or verify opaque, dynamically generated syntax trees.

**The Parametric Solution**:
SHARD uses **parameterised template functions** (`shard/rules/templates.py`). The Rule-Writer cannot write code; it can only propose changes to numeric thresholds or enum sets on existing, pre-approved rules (e.g., lowering `max_markup_ratio` from 1.35 to 1.25 on `RULE_REPAIR_BENCHMARK`). Each proposal must stay within hard physical bounds defined in `RuleDefinition.param_bounds`.

---

## 6. Multi-Gate Safety Validator & False-Positive Budgets

### Q16: Explain the Multi-Gate Safety Validator and each of its 4 gates (Gate A through Gate D).
**Answer**:
Implemented in `shard/rules/validator.py`, every candidate `RuleProposal` must pass 4 sequential safety gates before being committed to the rulebook:
1. **Gate A: Schema & Parameter Bounds Check**: Verifies that the targeted rule ID exists, the parameter is valid, the proposed value is of correct type, and the value lies strictly within allowed bounds (e.g., `markup_ratio` between 1.10 and 2.00). Also verifies maximum step delta limits (e.g., step change $\le 0.15$).
2. **Gate B: False-Positive Budget Check (TI-05)**: Re-scores candidate rule against the frozen Round 1 genuine claims set. Rejects the proposal if $\Delta \text{FPR} > \text{fp\_budget\_delta}$.
3. **Gate C: Golden Set Non-Regression Check**: Re-scores candidate rule against a golden dataset of historical confirmed fraud claims. Rejects the proposal if any previously caught fraud claim is missed.
4. **Gate D: Measurable Diagnostic Justification Check**: Requires that the proposal includes a non-empty human-readable rationale explicitly linked to an evasive typology identified by the Critic.

### Q17: What is Gate B (False-Positive Budget Check), why is it critical, and how is it calculated?
**Answer**:
Gate B is the cornerstone safety invariant (**TI-05**) of the SHARD system. In insurance fraud controls, a rule that catches 5 more fraudulent claims but simultaneously falsely flags 50 innocent customers is an unacceptable failure.

**Mathematical Calculation**:
1. The validation engine evaluates candidate rules on a dedicated, fixed validation dataset of 40 synthetic genuine claims (`fraud_share=0.0`, seed 9999) or frozen genuine claims.
2. When a candidate rule update is proposed in subsequent rounds, the validation engine creates a temporary sandbox rulebook containing the proposed change.
3. All validation claims are re-scored through the pipeline using the candidate rulebook:
   $$\text{FPR}_{\text{candidate}} = \frac{\text{FP}_{\text{candidate}}}{\text{Total Validation Claims}}$$
4. The candidate is evaluated against the false-positive budget:
   $$\text{Passed} \iff \text{FPR}_{\text{candidate}} \le \text{fp\_budget} \quad (\text{default } 0.10 \text{ or } 10\%)$$
   *Important Scientific Caveat*: This measured 0.0% false-positive rate is an empirical engineering safeguard evaluated on this fixed synthetic genuine cohort, not an unconstrained mathematical guarantee across arbitrary or real-world distributions.

### Q18: What happens when a proposed rule change fails Gate B?
**Answer**:
If a proposed rule change fails Gate B (e.g., $\text{FPR} = 0.125 > 0.10$):
1. **Immediate Rejection**: The proposal is marked `status = "REJECTED"` and `gate_failed = "GATE_B_FP_BUDGET"`.
2. **Rulebook Preservation**: The active `Rulebook` is preserved completely untouched. Its version remains unchanged, and no parameters are mutated.
3. **Structured Audit Logging**: A structured `RuleAuditRecord` is emitted to the audit log detailing the proposed change, the calculated validation FP rate, and the exact rejection reason (e.g., *"Candidate rulebook false-positive rate (12.5%, 5/40) exceeds allowable budget (10.0%)"*).
4. **Safe Continuation**: The orchestrator logs the rejection and proceeds to the next round safely without halting the pipeline.

### Q19: How does rulebook versioning work, and how is historical immutability maintained?
**Answer**:
Rulebook versioning is managed by semantic version strings (`v1.0.0`, `v1.1.0`) and immutable history logs in `shard/rules/rulebook.py`:
- **Cloning on Mutation**: When a rule change passes all 4 validation gates, the active rulebook is cloned. The original rulebook instance is left completely immutable.
- **Version Increment**: The minor version number is bumped (e.g., `1.0.0` $\to$ `1.1.0`).
- **Audit Changelog**: The cloned rulebook appends an immutable `RuleChangeRecord` to its internal `history` list, recording the timestamp, round number, rule ID, old parameter value, new parameter value, Gate B validation FP rate, and diagnostic rationale.
- **Round Result Immutability**: Every `RoundResult` holds frozen references to `rulebook_version_before` and `rulebook_version_after`, allowing the Streamlit UI to render a side-by-side diff between any two historical versions.

---

## 7. Orchestration, Reproducibility & Audit Logging

### Q20: Explain the Orchestrator's round lifecycle and stopping conditions.
**Answer**:
Implemented in `shard/orchestrator.py`, the `Orchestrator` manages the end-to-end execution loop across rounds:
```
Round N Lifecycle:
1. Orchestrator requests new claim batch from Generator (using Fraudster's round seed).
2. For each claim:
   a. Planner constructs InvestigationPlan.
   b. Researcher gathers EvidenceBundle.
   c. Investigator runs tools & applies Rulebook -> ClaimDecision.
3. Critic evaluates all decisions against GroundTruthVault -> CriticReport.
4. Orchestrator checks Dual Stopping Conditions:
   - Target recall achieved (Recall >= target_recall AND FPR <= max_fp_rate) -> Stop cleanly.
   - Maximum rounds reached (Round == max_rounds) -> Stop cleanly.
5. If continuing:
   a. Rule-Writer proposes parameter update -> Multi-Gate Safety Validator.
      - If accepted: Rulebook version bumped.
      - If rejected: Active rulebook preserved.
   b. Fraudster adjusts typology weights based on Critic's aggregate miss summary.
6. Audit logger flushes all events for the round.
```

### Q21: How is bit-identical reproducibility achieved (TI-07), and how do you prove it?
**Answer**:
Reproducibility is enforced under Technical Invariant **TI-07**:
- **Isolated PRNG Instances**: Global random state (`random.random()`, `np.random.seed()`) is strictly forbidden. All stochastic operations use isolated instances: `random.Random(seed)` and `np.random.default_rng(seed)`.
- **Deterministic Seed Derivation**: For round $N$, the round seed is deterministically derived from base seed:
  $$\text{round\_seed} = \text{int}(\text{hashlib.sha256}(f"{base\_seed}\_round\_{N}".encode()).hexdigest()[:8], 16)$$
- **Canonical Serialization**: Dictionary outputs are sorted and timestamps are excluded for canonical comparison (`to_canonical_dict()`).

**Verification Proof**:
In `tests/test_orchestrator.py::test_orchestrator_reproducibility()`, the orchestrator is run twice with `base_seed=42`. The test asserts that the resulting canonical dictionaries of all decisions, scores, findings, and metrics are **100% bit-identical**.

### Q22: Why are empty findings forbidden in ClaimDecisions (TI-02)?
**Answer**:
Technical Invariant **TI-02** mandates that every `ClaimDecision` must include a non-empty list of evidentiary `Finding` records.
In commercial claims adjudication and insurance litigation, an automated triage system cannot assign a disposition without recorded justification. If an insurer denies a claim or refers a claimant for fraud investigation with zero evidence, the insurer violates Treating Customers Fairly (TCF) regulations and Ombudsman standards.
In SHARD:
- Even an approved clean claim must record findings verifying that checks passed (e.g., *"DVLA registration valid"*, *"No MIAFTR total loss records found"*).
- The `ClaimDecision` constructor raises a `ValueError` if `findings` is empty.
- In our 90-claim benchmark run, exactly 540 findings were recorded (an average of 6 per claim), achieving 100% findings coverage.

---

## 8. UI Data Binding, Limitations & Enterprise Roadmap

### Q23: How does the Streamlit UI bind to the underlying engine, and how does it ensure zero hardcoded metrics (TI-12)?
**Answer**:
Under Technical Invariant **TI-12**, the Streamlit application (`app/streamlit_app.py`) is a pure presentation layer:
1. **Dynamic Session State Binding**: When the user clicks **`Execute Adversarial Loop`**, the UI instantiates `Orchestrator` in-process and stores the returned `ExperimentResult` in `st.session_state["experiment_result"]`.
2. **Zero Hardcoded Figures**: Every metric tile, confusion matrix heatmap, claims table row, agent trace container, and rulebook diff view pulls directly from the active `ExperimentResult` object or the on-disk audit log.
3. **Reset Safety**: The **`Reset Entire Experiment`** button clears session state and memory structures without deleting the historical on-disk JSON Lines audit trail.

### Q24: What are the known limitations of this proof-of-concept?
**Answer**:
To maintain academic integrity and rigorous transparency, SHARD documents 5 clear scope boundaries:
1. **Synthetic Motor Claims**: Evaluated on realistic synthetic UK motor claims; does not ingest production policyholder Personally Identifiable Information (PII).
2. **Deterministic Mathematical Checks**: Relies on mathematical, relational, and database verification rather than multi-modal physical image inspection or computer vision on damaged vehicle photographs.
3. **Simulated Registry Adapters**: DVLA and MIAFTR databases are simulated in-memory using UK realistic distributions rather than live external government REST endpoints.
4. **Parametric Template Adaptation**: Rule adaptation tunes thresholds on pre-approved templates rather than synthesizing completely novel rule logic.
5. **Single Insurance Domain**: Optimized specifically for motor insurance fraud typologies rather than commercial property, liability, or life insurance.

### Q25: How would you transition this architecture from an academic PoC to a production-grade enterprise system?
**Answer**:
To transition SHARD into an enterprise production deployment, four evolutionary architectural upgrades would be implemented:
1. **Local LLM Explanation Layer (Offline Ollama)**: Deploy a local quantized LLM (e.g., Llama-3-8B or Mistral-7B) to synthesize the Investigator's structured `Finding` objects into natural-language statutory letters for claims handlers.
2. **Live Registry Microservices**: Replace synthetic in-memory databases with secure, read-only REST connectors to the DVLA Vehicle Enquiry Service (VES) API and the Insurance Fraud Bureau (IFB) MIAFTR portal.
3. **On-Device Computer Vision**: Integrate lightweight convolutional networks (e.g., YOLOv8) into the Research Agent to inspect uploaded accident photographs for pre-existing rust or damage inconsistency.
4. **Distributed Orchestration & SIEM Integration**: Transition from in-process execution to an asynchronous message-queue architecture (e.g., Celery/Redis or Temporal) and pipe the JSON Lines audit trail directly into enterprise Security Information and Event Management (SIEM) systems (Splunk / Datadog).
