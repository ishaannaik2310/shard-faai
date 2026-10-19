# Requirements Traceability Matrix — Shard

> Legend: NOT STARTED | IN PROGRESS | VERIFIED | BLOCKED

---

## Part A — Assignment Mandatory Deliverables

| ID | Requirement | Planned Location | Verification Method | Acceptance Evidence | Status |
|---|---|---|---|---|---|
| AD-01 | Clear problem statement and domain justification | docs/ca3/PROJECT_BRIEF.md | Peer/examiner review | Document exists and answers: what problem, why agentic, why motor insurance | VERIFIED |
| AD-02 | Existing business challenge and proposed solution | docs/ca3/PROJECT_BRIEF.md Problem Statement | Review | Problem statement describes structural rule-lag, solution describes the loop | VERIFIED |
| AD-03 | Defined agent responsibilities and orchestration | docs/ca3/ARCHITECTURE.md | Review + code | Each agent has a named module, inputs, outputs and tool list | VERIFIED |
| AD-04 | Complete architecture diagram (agents, tools, data flow, decisions) | docs/ca3/ARCHITECTURE.md Mermaid block | Rendered diagram reviewed | Diagram matches implementation, all agents and trust boundaries visible | VERIFIED |
| AD-05 | Working Streamlit PoC using dummy/synthetic data | app/streamlit_app.py | Run streamlit run app/streamlit_app.py | Application starts, all tabs populate from executed workflow; pytest tests/test_ui.py passes | VERIFIED |
| AD-06 | Output validation, measured results, limitations, future scope | docs/ca3/PROJECT_BRIEF.md + Streamlit metrics tab | Metrics derived from executed run | Precision, recall, F1 from actual predictions vs. GT labels displayed in Critic tab & Control Room | VERIFIED |
| AD-07 | Presentation covering problem, solution, architecture, demonstration | docs/ca3/presentation/SHARD_CA3_Presentation.pptx | Slide deck verified (10 slides); demo rehearsed (docs/ca3/DEMO_SCRIPT.md) | 10-slide deck with embedded PNG architecture diagram; 5-min timed demo script; VIVA_QA.md completed | VERIFIED |
| AD-08 | Python and Streamlit technology stack | All source files | Code inspection | No non-Python runtimes; Streamlit drives the UI | VERIFIED |
| AD-09 | No paid API required | Entire codebase | grep for API key references; run with OPENAI_API_KEY unset | Application runs to completion without any external API call | VERIFIED |

---

## Part B — Technical Invariants

| ID | Invariant | Planned Location | Verification Method | Acceptance Evidence | Status |
|---|---|---|---|---|---|
| TI-01 | GT labels inaccessible to Planner, Research, Investigator | shard/data/vault.py (GT store); agent constructors receive no GT reference | AST import check: no vault/GT imports in planner, researcher, investigator | pytest tests/test_agents.py and tests/test_ui.py pass (Claims Explorer never exposes is_fraud) | VERIFIED |
| TI-02 | Every decision supported by recorded evidence | shard/core/models.py ClaimDecision requires non-empty findings list | Unit test: creating ClaimDecision with empty findings raises ValueError | pytest tests/test_rulebook.py::test_claim_decision_requires_non_empty_findings passes | VERIFIED |
| TI-03 | Metrics from actual predictions, not illustrative | shard/agents/critic.py computes from predictions vs gt_labels | Unit tests with known predictions check calculated values | pytest tests/test_critic.py (test_exact_metric_calculations, test_zero_denominator_handling) passes | VERIFIED |
| TI-04 | Rule execution & updates via templates, bounded parameters | shard/rules/validator.py + shard/rules/templates.py | Code inspection + AST test | No eval()/exec() in rule pipeline; pytest tests/test_rule_writer.py::test_no_eval_or_exec_in_rule_writer_pipeline passes | VERIFIED |
| TI-05 | Rule updates tested against genuine claims before acceptance | shard/rules/validator.py::RuleValidator (Gate B) | Unit test: candidate exceeding FP budget rejected | pytest tests/test_rule_writer.py::test_candidate_exceeding_fp_budget_rejected passes | VERIFIED |
| TI-06 | Critic evaluates independently (not just echoes Investigator) | shard/agents/critic.py receives ClaimDecision + GT label, not Investigator reasoning | Code inspection + AST test | pytest tests/test_critic.py (test_ground_truth_isolation_ast_check, test_label_mutation_alters_evaluation_not_investigation) passes | VERIFIED |
| TI-07 | Fixed-seed reproducibility | shard/data/generator.py accepts seed param | Run twice with same seed, diff output files | pytest tests/test_generator.py::test_reproducibility_identical_seeds passes | VERIFIED |
| TI-08 | Intermediate-risk claims go to human review, not auto-reject | shard/core/models.py DISPOSITION enum includes REFER | Unit test: Claims scoring in refer band [25, 59] get REFER disposition | pytest tests/test_rulebook.py::test_disposition_threshold_boundaries passes | VERIFIED |
| TI-09 | Full audit log for all decisions, rule changes, repair cycles | shard/logging/audit.py writes JSON-lines | Integration test checks log entry count matches claims processed | pytest tests/test_audit.py, tests/test_orchestrator.py, and tests/test_ui.py pass (100% coverage verified) | VERIFIED |
| TI-10 | Model availability not required | All agents use deterministic logic by default | Run with no ML/LLM dependencies installed; test suite passes | Entire test suite (146 tests) runs offline in ~35s without external models, network, or paid APIs | VERIFIED |
| TI-11 | Suspected != confirmed fraud | shard/core/models.py Disposition (triage recommendation) vs GroundTruthRecord.is_fraud | Code inspection + Critic classification convention | Triage recommendations (APPROVE/REFER/SUSPECTED_FRAUD) decoupled from evaluation truth; verified in UI badges and tables | VERIFIED |
| TI-12 | All UI results from executed workflow or labelled as illustrative | app/streamlit_app.py reads from session state populated by pipeline run | Code inspection + AppTest execution | pytest tests/test_ui.py confirms all UI elements populate dynamically from ExperimentResult with 0 hardcoded metrics | VERIFIED |

---

## Part C — Assumptions

| ID | Assumption | Risk if Wrong | Mitigation |
|---|---|---|---|
| AS-01 | Streamlit 1.66 supports all UI features we need | Low | Test UI before demo |
| AS-02 | Python 3.14 is compatible with all planned dependencies | Low for pure-Python; possible issues with C-extensions | Test imports at project setup |
| AS-03 | No LLM / Ollama integration required for MVP | Accepted scope decision | Documented in brief |
| AS-04 | 30 claims per round is sufficient to demonstrate metric improvement | Evaluation only | Configurable; increase if needed |
| AS-05 | scikit-learn (installed) can compute all required metrics | High confidence | Already in pip list |

---

## Part D — Out-of-scope Items (explicitly deferred)

| Item | Reason |
|---|---|
| Health / liability insurance | Complexity; out of stated MVP scope |
| OCR of physical documents | Requires external service or heavy ML |
| LLM / Ollama integration | Optional; not needed for deterministic pipeline |
| Production deployment | Not an assignment requirement |
| Real claimant data | Privacy; synthetic data is sufficient |
