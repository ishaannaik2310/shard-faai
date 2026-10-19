# Project Brief — Shard: Adversarial Multi-Agent Insurance Claims Auditor

## Problem Statement

Motor insurance fraud is a persistent and costly challenge. Insurers rely on rules-based fraud detection systems, but static rule sets degrade over time because organised fraudsters learn which patterns trigger investigation and deliberately avoid them. Manual re-calibration of rules is slow, expensive and reactive.

This project identifies a structural problem directly observable in how rules-based systems work:

- A rule written to catch pattern A cannot catch pattern B unless a human analyst writes a new rule.
- Analysts only write new rules once fraud has been detected and reported, creating an inevitable lag.
- The system has no mechanism to test whether a proposed rule change increases false positives before deploying it.

An agentic, adversarial feedback loop addresses exactly this lag by:
1. Simulating the fraudster's adaptive behaviour (generating synthetic claims that exploit current detection gaps).
2. Running an investigation pipeline that applies explicit, auditable rules.
3. Evaluating results against ground truth to measure what was missed.
4. Proposing bounded rule updates and stress-testing them against genuine claims before acceptance.
5. Repeating the loop, tightening controls progressively.

This cannot be achieved by a single model call or a static pipeline. It requires planning (which checks to run), tool execution (deterministic verification steps), validation (critic against hidden labels), conditional decisions (accept rule or reject it) and coordination across specialised agents.

## Domain

**Motor insurance claims** (UK context, synthetic data only).

Fraud typologies in scope for the MVP:
- Exaggerated repair cost claim (inflated quote from linked garage).
- Staged collision (two parties colluding; claim timing and network patterns are anomalous).
- Ghost vehicle (vehicle does not exist or was already written off).
- Pre-existing damage falsely reported as accident damage.
- Opportunistic claim inflation (genuine accident but padded with fictitious items).

Out of scope for MVP:
- Health / liability insurance.
- OCR of physical documents.
- Real claimant data.
- External API integrations.
- Production deployment.

## Users

| User | Need |
|---|---|
| Insurance analyst / investigator | Understand which claims are flagged and why; trust the audit trail. |
| Fraud team lead | See metrics, rule change history and improvement across rounds. |
| Assessor / viva examiner | Verify that a real agentic workflow executes, not a simulated one. |

## Scope

### In-scope MVP
- Synthetic motor insurance claim generator (seeded, reproducible).
- Adversarial fraudster that increases representation of previously undetected patterns.
- Planner, Research, Investigator, Critic, Rule-Writer agents.
- Deterministic rule engine with versioned rulebook.
- Orchestrator with configurable round count and stopping condition.
- Streamlit UI showing claims, decisions, traces, metrics and rule changes.
- Audit log (structured JSON lines).

### Not in scope (MVP)
- LLM integration (system works entirely on deterministic Python logic and templates).
- External APIs or paid services.
- Real insurance data.
- Deployment infrastructure.

## Measurable Goals

| Goal | Measure | Minimum acceptable |
|---|---|---|
| End-to-end pipeline executes | All claims in a round produce a decision record | 100% |
| Fraud detection improves | Recall on fraudulent claims increases across rounds | >= 1 additional fraud caught per round (on average) |
| False-positive safeguard works | Rule update rejected if genuine-claim FP rate exceeds budget | Demonstrated at least once |
| Ground-truth isolation | Planner/Research/Investigator cannot access GT labels | Verified by code inspection |
| Reproducibility | Same seed produces identical decisions | Verified by running twice |
| Audit trail | Every decision has structured log entry | 100% coverage |

## Deadline Summary

| Deliverable | Due |
|---|---|
| PPT, problem statement, domain details, architecture diagram | 15 October 2026 |
| Live presentation / demonstration | 19-22 October 2026 |
