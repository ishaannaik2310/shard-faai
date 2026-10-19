"""
Shard — Multi-Round Orchestrator.

Coordinates the end-to-end adversarial closed-loop simulation across rounds:
1. Generation (FraudsterAgent) -> Claims + Vault
2. Investigation (Planner -> Researcher -> Investigator) -> ClaimDecisions
3. Evaluation (CriticAgent) -> CriticReport (metrics + consistency checks)
4. Stopping condition evaluation (Target recall or max rounds)
5. Adaptation:
   - Rule-Writer proposes bounded parameter updates checked by Validation Gate
   - Fraudster updates attack typology sampling weights based on aggregate miss feedback
6. Complete audit logging (AuditLogger) in structured JSON Lines format.

CRITICAL INVARIANTS:
1. Ground-Truth Boundary (TI-01): Investigation agents receive ONLY observable Claims.
   Only the Critic accesses the evaluation GroundTruthVault.
2. Independent Metrics (TI-03): Stopping conditions and reported metrics are derived
   strictly from actual predictions against ground truth.
3. Safe Updating (TI-04, TI-05): Parameter updates use validated templates only; candidate
   rules exceeding the FP budget are rejected.
4. Reproducibility (TI-07): Base seed + deterministic round derivation guarantees bit-identical
   runs for identical configurations.
5. 100% Audit Coverage (TI-09): Every completed decision produces an individual audit event.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from shard.agents.critic import CriticAgent
from shard.agents.fraudster import FraudsterAgent, FraudsterRoundOutput
from shard.agents.investigator import InvestigationPipeline, PipelineResult
from shard.agents.rule_writer import RuleWriterAgent, RuleWriterOutput
from shard.core.models import (
    Claim,
    ClaimDecision,
    CriticFeedbackSummary,
    CriticReport,
    FraudsterAdaptationRecord,
    RuleAuditRecord,
    RuleProposal,
)
from shard.logging.audit import AuditLogger
from shard.rules.rulebook import Rulebook, load_default_rulebook
from shard.rules.validator import RuleValidator


# ---------------------------------------------------------------------------
# Configuration & Result Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestratorConfig:
    """Experiment configuration contract."""
    base_seed: int = 42
    n_claims_per_round: int = 30
    fraud_share: float = 0.30
    max_rounds: int = 5
    target_recall: float = 0.85
    max_fp_rate: float = 0.10
    adaptation_strength: float = 0.15
    fp_budget: float = 0.10
    early_stopping: bool = True
    audit_log_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.n_claims_per_round < 1:
            raise ValueError(f"n_claims_per_round must be >= 1, got {self.n_claims_per_round}")
        if not (0.0 <= self.fraud_share <= 1.0):
            raise ValueError(f"fraud_share must be between 0.0 and 1.0, got {self.fraud_share}")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if not (0.0 <= self.target_recall <= 1.0):
            raise ValueError(f"target_recall must be between 0.0 and 1.0, got {self.target_recall}")
        if not (0.0 <= self.max_fp_rate <= 1.0):
            raise ValueError(f"max_fp_rate must be between 0.0 and 1.0, got {self.max_fp_rate}")
        if self.adaptation_strength < 0.0:
            raise ValueError(f"adaptation_strength must be >= 0.0, got {self.adaptation_strength}")
        if not (0.0 <= self.fp_budget <= 1.0):
            raise ValueError(f"fp_budget must be between 0.0 and 1.0, got {self.fp_budget}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "n_claims_per_round": self.n_claims_per_round,
            "fraud_share": self.fraud_share,
            "max_rounds": self.max_rounds,
            "target_recall": self.target_recall,
            "max_fp_rate": self.max_fp_rate,
            "adaptation_strength": self.adaptation_strength,
            "fp_budget": self.fp_budget,
            "early_stopping": self.early_stopping,
            "audit_log_path": self.audit_log_path,
        }


@dataclass(frozen=True)
class RoundResult:
    """Complete, immutable record of an individual round's execution."""
    round_number: int
    rulebook_version_before: str
    rulebook_version_after: str
    claims_count: int
    decisions: Tuple[ClaimDecision, ...]
    critic_report: CriticReport
    rule_proposals: Tuple[RuleProposal, ...]
    rule_audit_records: Tuple[RuleAuditRecord, ...]
    fraudster_record: FraudsterAdaptationRecord
    continuation_decision: str  # TARGET_REACHED, CONTINUE, MAX_ROUNDS_REACHED, ERROR_STOP
    stop_reason: Optional[str] = None
    claims: Tuple[Claim, ...] = field(default_factory=tuple)
    pipeline_results: Tuple[PipelineResult, ...] = field(default_factory=tuple)

    def to_dict(self, include_timestamps: bool = True) -> Dict[str, Any]:
        """Convert round result to a serializable dictionary."""
        d = {
            "round_number": self.round_number,
            "rulebook_version_before": self.rulebook_version_before,
            "rulebook_version_after": self.rulebook_version_after,
            "claims_count": self.claims_count,
            "critic_report": self.critic_report.to_dict(),
            "rule_proposals": [p.to_dict() for p in self.rule_proposals],
            "rule_audit_records": [r.to_dict() for r in self.rule_audit_records],
            "fraudster_record": self.fraudster_record.to_dict(),
            "continuation_decision": self.continuation_decision,
            "stop_reason": self.stop_reason,
        }
        if not include_timestamps:
            if "generated_at" in d["critic_report"]:
                d["critic_report"].pop("generated_at")
            for r in d["rule_audit_records"]:
                if "timestamp" in r:
                    r.pop("timestamp")
        return d


@dataclass(frozen=True)
class ExperimentResult:
    """Complete immutable collection of all round results from an executed experiment."""
    config: OrchestratorConfig
    rounds: Tuple[RoundResult, ...]
    total_claims_processed: int
    final_rulebook: Rulebook
    final_rulebook_version: str
    total_proposals: int
    accepted_proposals: int
    rejected_proposals: int
    target_reached: bool
    final_stop_reason: str
    cumulative_metrics: Dict[str, Any]
    audit_log_path: Optional[str] = None
    run_id: str = ""

    def to_dict(self, include_timestamps: bool = True) -> Dict[str, Any]:
        """Convert experiment result to serializable dictionary."""
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "total_rounds": len(self.rounds),
            "total_claims_processed": self.total_claims_processed,
            "final_rulebook_version": self.final_rulebook_version,
            "total_proposals": self.total_proposals,
            "accepted_proposals": self.accepted_proposals,
            "rejected_proposals": self.rejected_proposals,
            "target_reached": self.target_reached,
            "final_stop_reason": self.final_stop_reason,
            "cumulative_metrics": dict(self.cumulative_metrics),
            "audit_log_path": self.audit_log_path,
            "rounds": [r.to_dict(include_timestamps=include_timestamps) for r in self.rounds],
        }

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Canonical dictionary representation for strict reproducibility verification."""
        d = self.to_dict(include_timestamps=False)
        d.pop("run_id", None)
        return d


# ---------------------------------------------------------------------------
# Orchestrator Engine
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Coordinates multi-round adversarial simulation, investigation, evaluation,
    adaptation, and audit logging.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        initial_rulebook: Optional[Rulebook] = None,
        investigation_pipeline: Optional[InvestigationPipeline] = None,
        critic: Optional[CriticAgent] = None,
        fraudster: Optional[FraudsterAgent] = None,
        rule_writer: Optional[RuleWriterAgent] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.initial_rulebook = initial_rulebook or load_default_rulebook()
        self.investigation_pipeline = investigation_pipeline or InvestigationPipeline()
        self.critic = critic or CriticAgent()
        self.fraudster = fraudster or FraudsterAgent(
            base_seed=self.config.base_seed,
            adaptation_strength=self.config.adaptation_strength,
        )
        self.rule_writer = rule_writer or RuleWriterAgent(
            validator=RuleValidator(fp_budget=self.config.fp_budget),
        )
        self.audit_logger = audit_logger or AuditLogger(
            log_path=self.config.audit_log_path,
        )

    def run(self) -> ExperimentResult:
        """
        Execute the complete multi-round adversarial feedback loop.

        Returns an immutable ExperimentResult containing all round records,
        cumulative metrics, final rulebook, and audit details.
        """
        active_rulebook = self.initial_rulebook.clone()
        rounds: List[RoundResult] = []

        # Log experiment initialization
        self.audit_logger.log_experiment_init(self.config.to_dict())

        previous_feedback: Optional[CriticFeedbackSummary] = None
        final_stop_reason = "MAX_ROUNDS_REACHED"
        target_reached = False

        for round_num in range(1, self.config.max_rounds + 1):
            version_before = active_rulebook.version
            round_seed = self.fraudster.derive_round_seed(round_num)

            # 1. Log Round Start
            self.audit_logger.log_round_start(
                round_number=round_num,
                rulebook_version=version_before,
                seed=round_seed,
            )

            # 2. Fraudster generates synthetic batch (claims + vault)
            fraudster_out: FraudsterRoundOutput = self.fraudster.generate_round(
                round_number=round_num,
                feedback=previous_feedback,
                n_claims=self.config.n_claims_per_round,
                fraud_share=self.config.fraud_share,
            )
            batch = fraudster_out.batch

            if fraudster_out.adaptation_record.adapted:
                self.audit_logger.log_fraudster_adaptation(
                    record=fraudster_out.adaptation_record,
                    round_number=round_num,
                )

            # 3. Investigation phase (observable claims only)
            # Update investigation pipeline rulebook with current active rulebook
            self.investigation_pipeline.rulebook = active_rulebook

            decisions: List[ClaimDecision] = []
            pipeline_results: List[PipelineResult] = []
            for claim in batch.claims:
                res: PipelineResult = self.investigation_pipeline.process(
                    claim=claim,
                    round_number=round_num,
                )
                decisions.append(res.decision)
                pipeline_results.append(res)
                # Audit log individual claim decision (100% audit coverage)
                self.audit_logger.log_claim_decision(
                    decision=res.decision,
                    round_number=round_num,
                )

            # 4. Evaluation phase (Critic accesses Vault)
            critic_report: CriticReport = self.critic.evaluate(
                decisions=decisions,
                vault=batch.vault,
                claims=batch.claims,
                rulebook=active_rulebook,
                round_number=round_num,
            )
            self.audit_logger.log_critic_report(
                report=critic_report,
                round_number=round_num,
            )

            # 5. Evaluate stopping condition on measured metrics
            target_met = (
                critic_report.recall >= self.config.target_recall
                and critic_report.fpr <= self.config.max_fp_rate
            )
            is_last_round = round_num >= self.config.max_rounds

            if self.config.early_stopping and target_met:
                continuation = "TARGET_REACHED"
                stop_reason = "TARGET_MET_RECALL_AND_FP_RATE"
                target_reached = True
                final_stop_reason = stop_reason
            elif is_last_round:
                continuation = "MAX_ROUNDS_REACHED"
                stop_reason = "MAX_ROUNDS_REACHED"
                final_stop_reason = stop_reason
            else:
                continuation = "CONTINUE"
                stop_reason = None

            # 6. Adaptation phase (if continuing to next round)
            round_proposals: Tuple[RuleProposal, ...] = ()
            round_audit_records: Tuple[RuleAuditRecord, ...] = ()

            if continuation == "CONTINUE":
                # Extract permitted aggregate feedback for next round
                previous_feedback = CriticFeedbackSummary.from_critic_report(critic_report)

                # Rule-Writer proposes and validates candidate modifications
                writer_out: RuleWriterOutput = self.rule_writer.propose_and_validate(
                    active_rulebook=active_rulebook,
                    feedback=previous_feedback,
                    round_number=round_num,
                )
                round_proposals = writer_out.proposals
                round_audit_records = writer_out.audit_records

                for p in round_proposals:
                    self.audit_logger.log_rule_proposal(p, round_number=round_num)
                for rec in round_audit_records:
                    self.audit_logger.log_rule_validation(rec, round_number=round_num)

                # Activate validated candidate rulebook for subsequent rounds
                if writer_out.accepted:
                    active_rulebook = writer_out.active_rulebook

            version_after = active_rulebook.version

            # 7. Log Round End
            self.audit_logger.log_round_end(
                round_number=round_num,
                claims_count=len(batch.claims),
                active_version=version_after,
                continuation_reason=continuation if stop_reason is None else stop_reason,
            )

            # Record round outcome
            round_record = RoundResult(
                round_number=round_num,
                rulebook_version_before=version_before,
                rulebook_version_after=version_after,
                claims_count=len(batch.claims),
                decisions=tuple(decisions),
                critic_report=critic_report,
                rule_proposals=round_proposals,
                rule_audit_records=round_audit_records,
                fraudster_record=fraudster_out.adaptation_record,
                continuation_decision=continuation,
                stop_reason=stop_reason,
                claims=tuple(batch.claims),
                pipeline_results=tuple(pipeline_results),
            )
            rounds.append(round_record)

            if stop_reason is not None:
                break

        # -------------------------------------------------------------------
        # 8. Post-Experiment Aggregation & Cumulative Metrics
        # -------------------------------------------------------------------
        total_claims = sum(r.claims_count for r in rounds)
        all_proposals = [p for r in rounds for p in r.rule_proposals]
        all_audit_records = [rec for r in rounds for rec in r.rule_audit_records]
        accepted_count = sum(1 for rec in all_audit_records if rec.status == "ACCEPTED")
        rejected_count = sum(1 for rec in all_audit_records if rec.status == "REJECTED")

        # Explicit mathematical aggregation: aggregate confusion matrix across all rounds
        cum_tp = sum(r.critic_report.tp for r in rounds)
        cum_fp = sum(r.critic_report.fp for r in rounds)
        cum_tn = sum(r.critic_report.tn for r in rounds)
        cum_fn = sum(r.critic_report.fn for r in rounds)
        cum_total = cum_tp + cum_fp + cum_tn + cum_fn

        cum_precision = round(cum_tp / (cum_tp + cum_fp), 4) if (cum_tp + cum_fp) > 0 else 0.0
        cum_recall = round(cum_tp / (cum_tp + cum_fn), 4) if (cum_tp + cum_fn) > 0 else 0.0
        cum_f1 = (
            round(2.0 * cum_precision * cum_recall / (cum_precision + cum_recall), 4)
            if (cum_precision + cum_recall) > 0.0
            else 0.0
        )
        cum_fpr = round(cum_fp / (cum_fp + cum_tn), 4) if (cum_fp + cum_tn) > 0 else 0.0
        cum_accuracy = round((cum_tp + cum_tn) / cum_total, 4) if cum_total > 0 else 0.0

        cumulative_metrics = {
            "total_evaluated": cum_total,
            "tp": cum_tp,
            "fp": cum_fp,
            "tn": cum_tn,
            "fn": cum_fn,
            "precision": cum_precision,
            "recall": cum_recall,
            "f1": cum_f1,
            "fpr": cum_fpr,
            "accuracy": cum_accuracy,
        }

        # Log Experiment End
        self.audit_logger.log_experiment_end(
            final_stop_reason=final_stop_reason,
            target_reached=target_reached,
            total_rounds=len(rounds),
            total_claims=total_claims,
            final_rulebook_version=active_rulebook.version,
            cumulative_metrics=cumulative_metrics,
        )

        return ExperimentResult(
            config=self.config,
            rounds=tuple(rounds),
            total_claims_processed=total_claims,
            final_rulebook=active_rulebook.clone(),
            final_rulebook_version=active_rulebook.version,
            total_proposals=len(all_proposals),
            accepted_proposals=accepted_count,
            rejected_proposals=rejected_count,
            target_reached=target_reached,
            final_stop_reason=final_stop_reason,
            cumulative_metrics=cumulative_metrics,
            audit_log_path=str(self.audit_logger.log_path) if self.audit_logger.log_path else None,
            run_id=self.audit_logger.run_id,
        )
