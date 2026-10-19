# Assumptions, Decisions and Unresolved Questions — Shard

---

## 1. Explicit Assumptions

| ID | Assumption | Basis | Impact if Wrong |
|---|---|---|---|
| A-01 | The assignment does not require integration with a real insurance data source | Assignment brief states dummy or synthetic data | None — scope is already synthetic |
| A-02 | A deterministic Python pipeline (no LLM) satisfies the agentic AI requirement, provided it implements planning, tool execution, conditional decisions, validation and multi-agent coordination | Assignment brief permits optional LLM; the behaviours listed are what make a system agentic | If examiner requires LLM output: add Ollama template-rendering layer (no API cost) |
| A-03 | Python 3.14 is compatible with all required packages (Streamlit 1.66, scikit-learn, pandas, numpy, plotly) | All are listed in pip list; all are pure-Python or have pre-built wheels | Pin specific versions in requirements.txt; test on first import |
| A-04 | 30 claims per round is sufficient for a credible live demonstration | Configurable; can be raised to 100 without changing architecture | Raise if metrics are too noisy |
| A-05 | The Streamlit in-process execution model is acceptable (no separate backend server) | Streamlit supports this pattern; sufficient for PoC | If performance is poor: add a background thread or pre-run log |
| A-06 | The ground-truth label in this synthetic PoC is known only at evaluation time; the system's output is a recommendation, not a verdict | Explicitly required by TI-11 | UI and reports must clearly label outputs as recommendations |
| A-07 | No adversarial agent should have direct access to GT labels, even though GT is synthetically generated and known | Required by TI-01 | Enforced by vault isolation; test will catch violations |

---

## 2. Design Decisions Made

| ID | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| D-01 | Use deterministic Python rule engine, not an LLM | LLM via Ollama; LLM via OpenAI | Eliminates external dependencies; fully reproducible; no latency; examiners can inspect every rule; satisfies TI-10 |
| D-02 | Ground-truth vault is a separate module with a controlled import list | Embed GT in Claim; pass GT to Critic via Orchestrator | Separate module makes isolation testable; grep-based lint enforces it |
| D-03 | Rule templates are parameterised Python functions, not DSL strings | JSON-rule DSL; LLM-generated code | No eval/exec risk (TI-04); templates are inspectable and bounded |
| D-04 | Three-disposition system: APPROVE / REFER / SUSPECTED_FRAUD | Binary approve/reject | TI-08 requires intermediate risk to human review; three-way is simplest compliant design |
| D-05 | Validation set for FP budget = genuine claims from Round 1 (frozen) | Separate held-out set; all genuine claims across rounds | Simple; reproducible; available from round 1 without extra data generation |
| D-06 | Fraudster adjusts typology weights based on Critic's aggregate miss counts | Fraudster reads individual missed claim IDs | Aggregate counts only; does not expose individual GT labels through the Fraudster path |
| D-07 | Streamlit runs the Orchestrator in-process; session state holds results | Separate FastAPI backend | Simpler architecture for PoC; no deployment requirement |
| D-08 | Motor insurance only for MVP | Health + motor from the start | Reduces scope to a completable size; health adds new typologies and data structures without adding agentic complexity |
| D-09 | Fixed seed as primary reproducibility mechanism | Hash-based determinism; full snapshot | Simplest approach; well-supported by Python random.Random and NumPy default_rng |

---

## 3. Rejected Alternatives

| Alternative | Why Rejected |
|---|---|
| Use LangChain / LangGraph for agent orchestration | Adds LLM dependency; complicates reproducibility; no gain for deterministic pipeline |
| Use ChromaDB (installed) for claim similarity search | Not needed for deterministic rule-based pipeline; adds complexity without demonstrating agentic behaviour |
| Use sentence-transformers (installed) for semantic claim analysis | LLM-like dependency; reproducibility is harder; not needed for MVP |
| Generate illustrative metrics and present as measured | Explicitly forbidden by TI-03 and assignment brief |
| Build UI first, then implement backend | Violates vertical slice principle; UI would be non-functional shell |
| Multiple insurance domains from day one | Scope too large; likely to produce incomplete implementation |

---

## 4. Unresolved Decisions

| ID | Question | Options | Decision Needed By | Owner |
|---|---|---|---|---|
| U-01 | Should Ollama / local LLM be used for explanation generation (not for logic)? | (a) Pure template strings — simpler, offline; (b) Ollama for richer text — adds optional dependency | Before Phase 4 | Engineer + supervisor |
| U-02 | Should the Streamlit UI support replay from a saved audit log (demo safety net)? | (a) Yes — pre-run and save log for fallback; (b) No — always run live | Before Phase 8 | Engineer |
| U-03 | Is the presentation deck format fixed (PPTX)? | Assignment brief mentions PPT | Before Phase 1 | Check assignment brief |
| U-04 | Does the examiner expect to inspect individual agent prompts/outputs, or is the structured log sufficient? | Depends on examiner expectations | Before Phase 9 (viva prep) | Engineer |

---

## 5. Implementation Risks (Top 5)

1. **Python 3.14 is very new** — some packages may have undiscovered incompatibilities. Mitigate: test all imports at the start of Phase 2 and pin working versions.

2. **Adversarial loop may not produce visible metric improvement** — if the baseline rules are too weak or too strong, the feedback loop may not demonstrate a meaningful improvement arc. Mitigate: design initial rules to catch only the most obvious patterns; seed the data so that round 2 specifically stresses a newly-added pattern.

3. **FP budget validation may be too strict or too lenient** — if every proposed rule fails the budget check, the Rule-Writer produces nothing and the loop stalls. Mitigate: tune FP_BUDGET_DELTA during integration testing; configurable via UI.

4. **Live demo crash** — the most likely failure mode is a Streamlit state error during the live demonstration. Mitigate: implement audit-log replay so the demo can fall back to a pre-run result.

5. **Scope creep** — the installed packages (ChromaDB, sentence-transformers, torch) suggest prior work on a broader system. Resist adding features that are not required for the assignment until all core invariants are verified.

---

## 6. Notes on Existing Environment

The workspace (w:\shard-ai) is **completely empty** — no prior source files, no Git repository. The Python environment (3.14.6) has a rich set of packages pre-installed, suggesting the developer environment is shared with other projects. Key installed packages relevant to this project:

- streamlit 1.66.0 (confirmed)
- pandas 3.0.6 (confirmed)
- numpy 2.5.2 (confirmed)
- scikit-learn 1.9.0 (confirmed, for metrics)
- plotly 7.1.0 (confirmed, for charts)
- pytest 9.1.1 (confirmed)
- python-pptx 1.0.2 (confirmed, for presentation deck)
- pydantic 2.13.4 (confirmed, optional for model validation)

Packages installed but **not planned for use in MVP**:
- chromadb, sentence-transformers, torch, transformers — too heavy for a deterministic pipeline.
