# SHARD — Final Assignment Acceptance Checklist

**Project**: SHARD (Adversarial Multi-Agent Insurance Claims Auditor)  
**Assignment**: CA-3 Foundations of Agentic AI  
**Status Legend**: `VERIFIED` | `PARTIALLY VERIFIED` | `NOT VERIFIED` | `BLOCKED`  

---

## 1. Traceability Matrix of Mandatory Assignment Requirements

| # | Requirement | Implementation / Document Location | Verification Method | Available Evidence | Status |
|---|---|---|---|---|---|
| **1** | **Clear Problem Statement** | `docs/ca3/PROJECT_BRIEF.md` (§1), `docs/ca3/presentation/SHARD_CA3_Presentation.pptx` (Slide 2) | Document review against CA-3 brief | Articulates the structural rule-lag problem: static rules degrade as fraudsters learn detection thresholds; manual rule updates lag behind novel fraud; no pre-deployment false-positive testing. | **VERIFIED** |
| **2** | **Domain Selection & Justification** | `docs/ca3/PROJECT_BRIEF.md` (§2), `docs/ca3/VIVA_QA.md` (Q2) | Scope review & industry alignment | Motor insurance chosen for explicit, verifiable fraud typologies (inflated repair estimates, staged collisions, ghost vehicles, pre-existing damage, opportunistic padding) with objective verification tools. | **VERIFIED** |
| **3** | **Existing Business Workflow & Inefficiencies** | `docs/ca3/PROJECT_BRIEF.md` (§1, §2), `docs/ca3/presentation/SHARD_CA3_Presentation.pptx` (Slide 2) | Workflow analysis | Details the traditional linear adjuster pipeline: manual inspection of flagged claims, reactive rule creation months after losses occur, and lack of pre-deployment false-positive guardrails. | **VERIFIED** |
| **4** | **Proposed Solution & Expected Benefits** | `docs/ca3/PROJECT_BRIEF.md` (§1), `docs/ca3/ARCHITECTURE.md` (§1), `app/streamlit_app.py` | Architecture & UI review | Closed-loop multi-agent adversarial simulation: automated investigation team, independent critic evaluation, bounded rulebook adaptation, and false-positive budget safeguards. | **VERIFIED** |
| **5** | **Agent Roles & Responsibilities** | `shard/agents/` (`planner.py`, `researcher.py`, `investigator.py`, `critic.py`, `fraudster.py`, `rule_writer.py`), `shard/orchestrator.py` | AST code inspection & unit tests | 7 dedicated roles with distinct contracts: Fraudster (attack generation), Planner (task sequencing), Researcher (evidence lookup), Investigator (tool execution & scoring), Critic (ground-truth evaluation), Rule-Writer (bounded proposal), Orchestrator (closed-loop lifecycle). Verified in `tests/test_agents.py`, `tests/test_critic.py`, `tests/test_rule_writer.py`. | **VERIFIED** |
| **6** | **Actual Multi-Agent Orchestration** | `shard/orchestrator.py`, `tests/test_orchestrator.py` | Integration tests & programmatic execution | Executes complete multi-round loop: Generator -> Planner -> Researcher -> Investigator -> Critic -> Rule-Writer -> Validation Gate -> Fraudster -> Next Round. Verified in 16 orchestrator tests. | **VERIFIED** |
| **7** | **Architecture Diagram with Components, Flows & Boundaries** | `docs/ca3/diagrams/shard-architecture.mmd`, `docs/ca3/diagrams/shard-architecture.png`, `docs/ca3/diagrams/shard-architecture.html` | Visual inspection & rendering check | Complete architectural view detailing the 7 agents, deterministic tools, versioned Rulebook, Ground-Truth Vault, trust boundaries (Investigation Zone vs Evaluation Zone vs Adaptation Zone), and decision gates. | **VERIFIED** |
| **8** | **Working Streamlit UI** | `app/streamlit_app.py`, `app/presentation.py`, `tests/test_ui.py` | Streamlit AppTest suite & CLI execution | 6 interactive tabs (Control Room, Claims Explorer, Agent Trace, Critic Report, Rulebook Evolution, Audit Log). Zero mock metrics. Verified by 11 tests in `tests/test_ui.py` and local CLI launch. | **VERIFIED** |
| **9** | **Synthetic-Data Demonstration** | `shard/data/generator.py`, `shard/data/vault.py`, `tests/test_generator.py` | Data validation tests & seed diffs | Seeded, reproducible motor claim generator supporting 5 realistic fraud typologies. Produces rich observable claim attributes while strictly isolating ground-truth labels in `GroundTruthVault`. Verified in 20 generator tests. | **VERIFIED** |
| **10** | **Output Validation & Actual Measured Results** | `shard/agents/critic.py`, `docs/ca3/results/experiment_summary_seed42.json`, `app/presentation.py` | Exact metric tests & saved experiment artifact | Evaluates actual predictions against hidden ground truth to compute exact Precision, Recall, F1, FPR, and per-typology catch rates. Verified in 16 critic tests with known mixtures and edge cases. | **VERIFIED** |
| **11** | **Limitations & Future Scope** | `docs/ca3/PROJECT_BRIEF.md`, `README.md`, `docs/ca3/VIVA_QA.md` (Q22, Q25), `docs/ca3/presentation/SHARD_CA3_Presentation.pptx` (Slide 9) | Technical documentation review | Explicitly documents PoC boundaries: synthetic claims only, 5 discrete fraud typologies, simplified motor insurance domain, deterministic rule templates rather than neural policies, no live registry integrations. | **VERIFIED** |
| **12** | **PowerPoint Presentation** | `docs/ca3/presentation/SHARD_CA3_Presentation.pptx` | python-pptx structural validation & slide inspection | 10 academic slides structured strictly per assignment specifications: Problem, Solution, Agent Roles, Architecture, Agentic Characteristics, Live Demo, Measured Results, Limitations, Conclusion. | **VERIFIED** |
| **13** | **Repeatable 5-Minute Demonstration** | `docs/ca3/DEMO_SCRIPT.md` | Rehearsal script timing & step-by-step verification | Structured 5-minute timed walkthrough (0:00–5:00) with exact configuration inputs, button actions, tab sequence, specific claim case study, real rule evolution, and emergency fallback procedures. | **VERIFIED** |
| **14** | **Technical Understanding Sufficient for Viva** | `docs/ca3/VIVA_QA.md` | Technical assessment & question coverage | Comprehensive, mathematically and architecturally grounded answers to 25 core examiner questions covering orchestration, scoring, evaluation, safeguards, and causal limits. | **VERIFIED** |

---

## 2. Technical Invariants Audit (TI-01 to TI-12)

| ID | Invariant Summary | Primary Evidence | Status |
|---|---|---|---|
| **TI-01** | Ground-Truth Vault Isolation | AST tests in `tests/test_agents.py`, `tests/test_critic.py`, `tests/test_orchestrator.py`; dataframe validation in `tests/test_ui.py`. | **VERIFIED** |
| **TI-02** | Evidence-Backed Decisions | `ClaimDecision` enforces non-empty `findings`. Verified in `tests/test_rulebook.py`. | **VERIFIED** |
| **TI-03** | Real Measured Metrics | `CriticAgent` calculates exact formulas; zero hardcoded/mock metrics. Verified in `tests/test_critic.py`. | **VERIFIED** |
| **TI-04** | Bounded Rule Updates (No eval/exec) | AST check confirms 0 occurrences of `eval()` or `exec()`. Verified in `tests/test_rule_writer.py`. | **VERIFIED** |
| **TI-05** | False-Positive Budget Safeguard | Gate B in `RuleValidator` rejects proposals exceeding budget on frozen genuine claims. Verified in `tests/test_rule_writer.py`. | **VERIFIED** |
| **TI-06** | Independent Critic Evaluation | Critic takes structured decisions and labels, never investigator internal reasoning. Verified in `tests/test_critic.py`. | **VERIFIED** |
| **TI-07** | Fixed-Seed Reproducibility | Two runs with identical seeds produce bit-identical claims and decisions. Verified in `tests/test_generator.py` and `tests/test_orchestrator.py`. | **VERIFIED** |
| **TI-08** | Non-Binary Triage (REFER to Human) | Intermediate risk scores (25–59) map to `REFER`. Verified in `tests/test_rulebook.py`. | **VERIFIED** |
| **TI-09** | 100% Audit Coverage | Structured JSON-lines audit trail captures every lifecycle event and decision. Verified in `tests/test_audit.py` and `tests/test_orchestrator.py`. | **VERIFIED** |
| **TI-10** | Zero Remote API / Model Dependency | Operates 100% locally without external paid APIs, remote keys, or cloud models; requires standard Python packages pinned in `requirements.txt`. Verified offline in clean environment. | **VERIFIED** |
| **TI-11** | Decoupled Triage from Legal Truth | `SUSPECTED_FRAUD` is a triage recommendation for an adjuster, not a confirmed verdict. Verified in models, UI badges, and tests. | **VERIFIED** |
| **TI-12** | Pure Dynamic UI Data Binding | Streamlit app binds exclusively to session state `ExperimentResult`. Verified in `tests/test_ui.py`. | **VERIFIED** |
