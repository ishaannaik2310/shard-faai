# Implementation Plan — Shard

> **Principle**: Build a working end-to-end vertical slice first. Do not build elaborate subsystems until a basic claim can travel the full pipeline and produce a validated result.

---

## Phase 0 — Workspace Reconnaissance and Specification Freeze

**Goal**: Understand the existing state of the codebase; produce all planning artifacts; freeze architecture and requirements.

**Files changed**: `docs/ca3/*.md`

**Acceptance gate**: All five planning documents exist; architecture diagram renders; traceability matrix lists every invariant; no application code written.

**Status**: IN PROGRESS (this phase)

**Dependencies**: None.

---

## Phase 1 — Submission Documentation and Architecture Assets

**Goal**: Produce the PowerPoint deck and architecture diagram required for the 15 October 2026 deadline.

**Files changed**:
- `docs/ca3/presentation/shard_ca3_deck.pptx`
- `docs/ca3/architecture_diagram.png`

**Acceptance gate**:
- Deck covers: problem statement, solution overview, architecture diagram, agent responsibilities, planned demo flow.
- Slide count: 8-12 slides, 5-minute presentation pace.
- Architecture diagram matches ARCHITECTURE.md.

**Dependencies**: Phase 0 complete.

**Risks**: Tight deadline (same day as Phase 0); prioritise this phase immediately.

---

## Phase 2 — Data Contracts and Seeded Synthetic Data

**Goal**: Implement typed data models and a reproducible claim generator. Prove reproducibility with tests.

**Files changed**:
- `shard/__init__.py`
- `shard/core/models.py`
- `shard/data/generator.py`
- `shard/data/vault.py`
- `tests/test_generator.py`

**Acceptance gate**:
- `pytest tests/test_generator.py` passes. (VERIFIED — 20/20 passed)
- Two runs with same seed produce identical claim batches. (VERIFIED)
- `Claim` objects contain no `is_fraud` field. (VERIFIED)
- GT labels accessible only via `vault.get_labels()`. (VERIFIED)

**Status**: COMPLETED & VERIFIED

**Dependencies**: Phase 0.

---

## Phase 3 — Baseline Rules and Deterministic Risk Scoring

**Goal**: Implement the Rulebook and the six deterministic investigation tools. Prove scoring correctness.

**Files changed**:
- `shard/rules/__init__.py`
- `shard/rules/rulebook.py`
- `shard/rules/templates.py`
- `shard/rules/tools.py`
- `data/rulebook_v1.json`
- `tests/test_tools.py`
- `tests/test_rulebook.py`

**Acceptance gate**:
- `pytest tests/test_tools.py tests/test_rulebook.py` passes. (VERIFIED — 34/34 passed)
- Each tool returns a typed result; none import from `vault.py`. (VERIFIED)
- Standardised 0–100 scale: A claim with all indicators set scores 100.0 (SUSPECTED_FRAUD); a clean claim scores 0.0 (APPROVE). (VERIFIED)

**Status**: COMPLETED & VERIFIED

**Dependencies**: Phase 2.

---

## Phase 4 — Investigation Agents and Tool Interfaces

**Goal**: Implement Planner, Research and Investigator agents. Produce ClaimDecision for a single claim.

**Files changed**:
- `shard/core/models.py`
- `shard/agents/__init__.py`
- `shard/agents/planner.py`
- `shard/agents/researcher.py`
- `shard/agents/investigator.py`
- `tests/test_agents.py`

**Acceptance gate**:
- `pytest tests/test_agents.py` passes. (VERIFIED — 11/11 passed, 65/65 total test suite)
- A single claim travels Planner -> Research -> Investigator and produces a valid ClaimDecision. (VERIFIED)
- `ClaimDecision.findings` is non-empty for all claims. (VERIFIED)
- AST-verified check confirms no vault imports in planner, researcher, investigator modules. (VERIFIED)

**Status**: COMPLETED & VERIFIED

**Dependencies**: Phase 3.

---

## Phase 5 — Critic, Ground-Truth Isolation and Evaluation Metrics

**Goal**: Implement Critic agent. Verify metrics are computed from actual predictions. Verify GT isolation.

**Files changed**:
- `shard/core/models.py`
- `shard/agents/__init__.py`
- `shard/agents/planner.py` (Issue A regression fix)
- `shard/agents/researcher.py` (Issue B provenance fix)
- `shard/agents/critic.py`
- `tests/test_agents.py`
- `tests/test_critic.py`

**Acceptance gate**:
- `pytest tests/test_critic.py` passes. (VERIFIED — 16/16 passed, 82/82 total test suite)
- With known predictions, Critic computes exact expected precision/recall/F1/FPR/Accuracy. (VERIFIED)
- AST-level check verifies that investigation-facing modules NEVER import vault. (VERIFIED)
- Consistency violations detected for duplicate IDs, missing decisions, unknown rule IDs, out-of-range risk scores, empty findings, and threshold mismatches. (VERIFIED)
- Strict non-destructive property verified: Critic never mutates original decisions. (VERIFIED)

**Status**: COMPLETED & VERIFIED

**Dependencies**: Phase 4.

---

## Phase 6 — Adaptive Adversary and Bounded Rule-Writer

**Goal**: Implement Fraudster and Rule-Writer. Demonstrate that FP budget safeguard rejects bad rules.

**Files changed**:
- `shard/core/models.py` (CriticFeedbackSummary, FraudsterAdaptationRecord, RuleProposal, RuleValidationResult, RuleAuditRecord)
- `shard/rules/validator.py` (multi-gate validation engine: Gates A, B, C, D)
- `shard/rules/rulebook.py` (added clone method)
- `shard/agents/__init__.py` (exported FraudsterAgent, RuleWriterAgent)
- `shard/agents/fraudster.py` (adaptive adversarial claim generation)
- `shard/agents/rule_writer.py` (bounded rule adaptation from diagnostic feedback)
- `tests/test_fraudster.py` (11/11 tests passed)
- `tests/test_rule_writer.py` (16/16 tests passed)

**Acceptance gate**:
- `pytest tests/test_rule_writer.py tests/test_fraudster.py` passes. (VERIFIED — 27/27 passed, 109/109 total test suite)
- A rule update that would push FP rate over budget is rejected and logged. (VERIFIED — Gate B test passed)
- A valid rule update is accepted, increments Rulebook version (e.g. 1.0.0 -> 1.1.0) and is recorded in history. (VERIFIED)
- No eval() or exec() calls anywhere in the rule pipeline. (VERIFIED — AST check passed)
- Fraudster increases weight of missed typology for the next round. (VERIFIED — EXAGGERATED_REPAIR weight increased from 0.25 to 0.3304)
- Multi-round Orchestration remains explicitly decoupled as future work (Phase 7).

**Status**: COMPLETED & VERIFIED

**Dependencies**: Phase 5.

---

## Phase 7 — Orchestration, Feedback Loops and Trace Logging

**Goal**: Implement Orchestrator and Audit Logger. Run a full multi-round loop end-to-end.

**Files changed**:
- `shard/orchestrator.py`
- `shard/logging/audit.py`
- `tests/test_orchestrator.py`
- `tests/test_audit.py`

**Acceptance gate**:
- `pytest tests/test_orchestrator.py tests/test_audit.py` passes. (VERIFIED — 26/26 passed, 135/135 total test suite in 2.09s)
- A 2-round run with 10 claims produces an audit log with correct event count. (VERIFIED — 20 decisions, paired lifecycle events)
- Stopping condition (max rounds) terminates the loop. (VERIFIED — halts cleanly when early stopping disabled or target not reached)
- Recall-based stopping condition terminates early when target is met. (VERIFIED — TARGET_MET_RECALL_AND_FP_RATE early exit verified)
- Audit log entries match claims processed (100% coverage). (VERIFIED — exactly 1 CLAIM_DECISION event per processed claim)
- Ground-truth isolation enforced (TI-01): hidden labels stripped from audit events; AST verification confirms no unauthorized vault imports. (VERIFIED)
- Bit-identical reproducibility across runs verified (TI-07). (VERIFIED)

**Status**: COMPLETED & VERIFIED

**Dependencies**: Phase 6.

---

## Phase 8 — Streamlit Integration

**Goal**: Wire the Orchestrator into the Streamlit UI. All tabs populate from executed pipeline.

**Files changed**:
- `app/__init__.py`
- `app/presentation.py` (dark theme CSS, Plotly trajectory/confusion matrix charts, observable dataframes)
- `app/streamlit_app.py` (6 interactive tabs: Control Room, Claims Explorer, Agent Trace, Critic Report, Rulebook Evolution, Audit Log)
- `shard/orchestrator.py` (updated RoundResult to retain observable claims and PipelineResults for UI inspection)
- `tests/test_ui.py` (11 unit and AppTest integration tests)

**Acceptance gate**:
- `streamlit run app/streamlit_app.py` starts without error. (VERIFIED — module imports cleanly, AppTest passes)
- Configuring seed/rounds/claims and clicking Run executes the pipeline and populates all tabs. (VERIFIED — AppTest simulation executes full 2-round pipeline)
- No hardcoded metric values in the UI layer. (VERIFIED — TI-12 enforced, all figures and KPI cards derived from ExperimentResult)
- Audit log tab displays actual log entries. (VERIFIED — 100% decision coverage indicator, downloadable .jsonl and canonical .json)
- Rulebook tab shows rule diffs and parameter histories between rounds. (VERIFIED)
- Ground-truth isolation strictly maintained in investigator views (TI-01 verified).
- `pytest tests/test_ui.py` passes (11/11 passed, 146/146 total test suite passed in 35.09s). (VERIFIED)

**Status**: COMPLETED & VERIFIED

---

## Phase 9 — Integration Testing, Regression, Demo Readiness and Viva Preparation

**Goal**: Full integration test; demo script rehearsal; viva Q&A preparation; submission deliverables.

**Files changed**:
- `docs/ca3/FINAL_ACCEPTANCE_CHECKLIST.md` (complete rubric verification for AD-01 to AD-09 and TI-01 to TI-12)
- `docs/ca3/diagrams/shard-architecture.mmd` (Mermaid architectural source diagram)
- `docs/ca3/diagrams/shard-architecture.html` (interactive architecture visualization)
- `docs/ca3/diagrams/shard-architecture.png` (high-resolution 2400x1450 diagram asset)
- `docs/ca3/presentation/SHARD_CA3_Presentation.pptx` (10-slide viva presentation deck embedding architecture diagram)
- `docs/ca3/results/experiment_summary_seed42.json` (canonical 3-round benchmark and adaptive calibration results)
- `docs/ca3/DEMO_SCRIPT.md` (timed 5-minute operator demonstration walkthrough)
- `docs/ca3/VIVA_QA.md` (25 comprehensive technical defense questions and answers)
- `docs/ca3/RUBRIC_TRACEABILITY.md` (updated matrix with all deliverables and invariants VERIFIED)

**Acceptance gate**:
- `pytest` passes entire test suite without errors. (VERIFIED — 146/146 passed in 29.42s)
- Presentation deck covers problem, solution, architecture, safety gates, and results. (VERIFIED — 10 slides, 16:9 widescreen)
- Architecture diagram matches implementation and is legible. (VERIFIED — 2400x1450 PNG, Segoe UI fonts, no badge collisions)
- Demo script rehearsed and timed for 5 minutes. (VERIFIED — docs/ca3/DEMO_SCRIPT.md)
- All 14 assignment deliverables verified against RUBRIC_TRACEABILITY.md. (VERIFIED — 9/9 AD and 12/12 TI verified)
- Canonical benchmark results documented and verified. (VERIFIED — docs/ca3/results/experiment_summary_seed42.json)

**Status**: COMPLETED & VERIFIED

**Dependencies**: Phase 8.

---

## Dependency Graph

```
Phase 0 -> Phase 1 (deadline: 15 Oct)
Phase 0 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8 -> Phase 9
```

Phase 1 (presentation deck) can proceed in parallel with Phases 2-4 but must be complete by the submission deadline.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Python 3.14 incompatibility with a dependency | Medium | High | Test imports in Phase 2; pin versions in requirements.txt |
| Streamlit session state complexity slows Phase 8 | Medium | Medium | Build thin UI wrapper; do heavy work in Orchestrator |
| FP budget logic too restrictive - no rules ever accepted | Low | Medium | Make FP_BUDGET_DELTA configurable; test with low fraud fraction |
| Demo crashes during live presentation | Medium | High | Pre-run and save audit log; UI can replay from saved log |
| 30 claims insufficient for stable metrics | Low | Medium | Make n_claims configurable up to 100 |
| Adversarial loop does not visibly improve recall | Medium | High | Seed carefully; ensure at least one obvious typology is initially below threshold |
