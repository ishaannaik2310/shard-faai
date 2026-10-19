# SHARD — Adversarial Multi-Agent Insurance Claims Auditor

CA-3 Foundations of Agentic AI | Python 3.14 · Streamlit 1.66 · 100% Offline & Deterministic

---

## 1. Project Overview

**SHARD** is an adversarial multi-agent insurance claims auditor that simulates a realistic motor insurance fraud investigation department. Rather than relying on static rules or ungrounded generative models, SHARD couples:

1. **An Investigation Team** (Planner, Research, and Investigator agents) executing verifiable, deterministic fraud detection tools against observable claim files.
2. **An Evaluation Authority** (Critic Agent) operating behind a strict ground-truth boundary to compute exact classification metrics (precision, recall, F1, FPR, accuracy) and identify missed fraud patterns.
3. **An Adversarial Adaptation Loop** (Fraudster Agent shifting synthetic fraud attack distributions towards evading typologies; Rule-Writer proposing bounded rule adjustments checked against an explicit false-positive budget).
4. **An End-to-End Orchestrator & Audit Trail** capturing 100% of decisions in structured, append-only JSON-lines and driving an interactive Streamlit Control Room.

---

## 2. Implementation Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Workspace Reconnaissance & Architecture Specification | COMPLETED & VERIFIED |
| **Phase 1** | Presentation Deck & Architectural Assets | COMPLETED & VERIFIED |
| **Phase 2** | Data Contracts (`Claim`, `GroundTruthVault`) & Seeded Generator | COMPLETED & VERIFIED |
| **Phase 3** | Baseline Rulebook & Deterministic Investigation Tools | COMPLETED & VERIFIED |
| **Phase 4** | Investigation Pipeline (`Planner`, `Researcher`, `Investigator`) | COMPLETED & VERIFIED |
| **Phase 5** | `Critic` Agent, Metric Computations & Ground-Truth Boundaries | COMPLETED & VERIFIED |
| **Phase 6** | `Fraudster` Adversary & `RuleWriter` Multi-Gate Validator (Gates A–D) | COMPLETED & VERIFIED |
| **Phase 7** | Closed-Loop `Orchestrator` & Structured `AuditLogger` | COMPLETED & VERIFIED |
| **Phase 8** | Interactive Streamlit Control Room (`app/streamlit_app.py`) | COMPLETED & VERIFIED |
| **Phase 9** | Integration Testing, Demo Readiness & Viva Preparation | COMPLETED & VERIFIED |

**Test Suite**: 146 tests passing across 10 test modules with 100% offline local execution in ~34s (zero remote APIs or paid keys; required Python packages pinned in `requirements.txt`).

---

## 3. Project Structure

```
shard-ai/
├── app/
│   ├── __init__.py
│   ├── presentation.py           # Dark theme styling, dataframes, Plotly charts
│   └── streamlit_app.py          # Interactive Control Room UI (6 tabs)
├── shard/
│   ├── core/
│   │   └── models.py             # Typed data contracts (Claim, Plan, Decision, etc.)
│   ├── data/
│   │   ├── generator.py          # Seeded reproducible synthetic claim generator
│   │   └── vault.py              # Isolated Ground-Truth Vault
│   ├── agents/
│   │   ├── planner.py            # Feature-driven investigation planning
│   │   ├── researcher.py         # Evidence retrieval & registry lookup
│   │   ├── investigator.py       # Tool execution & score aggregation
│   │   ├── critic.py             # Ground-truth evaluation & anomaly audit
│   │   ├── fraudster.py          # Adaptive attack distribution generation
│   │   └── rule_writer.py        # Bounded rule adaptation
│   ├── rules/
│   │   ├── rulebook.py           # Versioned rulebook management
│   │   ├── templates.py          # Deterministic rule templates (no eval/exec)
│   │   ├── tools.py              # 6 deterministic verification tools
│   │   └── validator.py          # Gates A-D rule candidate validator
│   ├── logging/
│   │   └── audit.py              # Structured JSON-lines audit logging
│   └── orchestrator.py           # Multi-round closed-loop controller
├── data/
│   └── rulebook_v1.json          # Canonical baseline rulebook configuration
├── tests/
│   ├── test_agents.py            # Planner, Researcher, Investigator tests
│   ├── test_audit.py             # Audit logging and coverage tests
│   ├── test_critic.py            # Metric calculations & GT isolation tests
│   ├── test_fraudster.py         # Adversarial distribution shift tests
│   ├── test_generator.py         # Reproducibility & schema validation tests
│   ├── test_orchestrator.py      # Closed-loop execution & stopping conditions
│   ├── test_rule_writer.py       # Gate A-D validation & proposal tests
│   ├── test_rulebook.py          # Versioning, clone & boundary tests
│   ├── test_tools.py             # 6 deterministic tool verification tests
│   └── test_ui.py                # Streamlit AppTest & presentation unit tests
├── docs/ca3/
│   ├── PROJECT_BRIEF.md          # Formal problem statement & domain scope
│   ├── RUBRIC_TRACEABILITY.md    # Requirements traceability matrix
│   ├── ARCHITECTURE.md           # Architecture specification & Mermaid diagrams
│   ├── IMPLEMENTATION_PLAN.md    # Phase-by-phase engineering milestones
│   └── ASSUMPTIONS_AND_DECISIONS.md # Design decisions & technical invariants
├── requirements.txt
└── README.md
```

---

## 4. Quick Start

### Installation

```bash
git clone <repo_url>
cd shard-ai
pip install -r requirements.txt
```

### Running the Test Suite

Execute all 146 unit, integration, and UI tests:

```bash
python -m pytest -v
```

### Launching the Streamlit Control Room

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## 5. Streamlit Control Room Features

The UI provides six interactive tabs communicating directly with the running multi-agent orchestrator:

- **Tab A — Control Room**: Live KPI summary cards (Total Claims, Fraud Catch Rate, False Positive Rate, F1 Score), Plotly multi-round metric trajectory curve (Recall vs FPR), and round-by-round summary table.
- **Tab B — Claims Explorer**: Filterable observable claims table (filter by triage disposition, score, claim ID). Selecting a claim opens a structured deep-dive inspector with submission details, triage badge, audit findings, Planner task rationale, Research evidence dossier, and deterministic tool traces. Hidden ground-truth labels are strictly excluded.
- **Tab C — Agent Trace**: Chronological, filterable audit event stream displaying exact agent activity, task rationale, and JSON payloads across every round.
- **Tab D — Critic Report**: Authoritative ground-truth evaluation dashboard featuring a Plotly Confusion Matrix heatmap, global diagnostic metrics, per-typology detection rate breakdown, and consistency violation logs.
- **Tab E — Rulebook Evolution**: Active rule definitions, parameter tuning history, and validation gate audit logs (Gates A–D) proving whether proposed updates were accepted or rejected against the false-positive budget.
- **Tab F — Audit Log**: 100% decision coverage indicator, raw event stream table, and one-click downloads for sanitized `.jsonl` audit trails and canonical reproducibility `.json` experiment archives.

---

## 6. Key Technical Invariants

- **Strict Ground-Truth Isolation (TI-01)**: `GroundTruthVault` is strictly isolated. AST inspection verifies that Planner, Researcher, and Investigator never import the vault. Claims Explorer in the UI displays only observable claim data.
- **Evidence-Backed Decisions (TI-02)**: Every `ClaimDecision` requires non-empty findings; decisions without evidence raise validation errors.
- **Measured Metrics Only (TI-03, TI-12)**: All UI figures, KPI cards, and charts derive directly from actual predictions vs ground truth via `CriticAgent`. No hardcoded or illustrative numbers.
- **Deterministic & Bounded Rules (TI-04)**: No `eval()` or `exec()`. Rules execute through bounded parametric templates.
- **False-Positive Budget (TI-05)**: Proposed rule modifications are tested against a frozen genuine-claims validation set (Gate B) and rejected if false-positive growth exceeds budget.
- **Bit-Identical Reproducibility (TI-07)**: Dedicated seeded RNG instances ensure identical random seeds produce bit-identical claim batches and audit trails.
- **Decoupled Triage (TI-11)**: Dispositions (`APPROVE`, `REFER`, `SUSPECTED_FRAUD`) are presented as triage recommendations for human adjusters, not legally binding verdicts.
- **100% Offline & Zero Cost (AD-09, TI-10)**: Operates entirely locally with standard Python libraries. No external APIs, paid keys, or remote models required.
