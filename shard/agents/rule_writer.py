"""
Shard — Bounded Rule-Writer Agent.

Translates structured evaluation diagnostics into safe, bounded candidate modifications
to the Rulebook and validates each candidate through an independent multi-stage validation gate.

ARCHITECTURAL TRUST BOUNDARY (TI-01, TI-04, TI-05, TI-06):
1. No Raw Ground-Truth Access: The Rule-Writer NEVER receives GroundTruthVault or individual
   claim labels. It consumes only redacted, aggregate evaluation summaries.
2. Template-Bounded Parameters Only: Proposes updates strictly through predefined parameter
   constraints. NO eval(), NO exec(), and NO dynamic Python code generation.
3. False-Positive Budget Protection: All proposed changes must pass the independent validation gate
   against a frozen genuine-claims set before activation.
4. Non-Destructive Invariant: Rejected proposals preserve the active rulebook 100% unchanged.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from shard.core.models import (
    CriticFeedbackSummary,
    CriticReport,
    RuleAuditRecord,
    RuleProposal,
    RuleValidationResult,
)
from shard.rules.rulebook import Rulebook
from shard.rules.validator import SUPPORTED_CONSTRAINTS, ParameterConstraint, RuleValidator


@dataclass(frozen=True)
class RuleWriterOutput:
    """Explicit return contract from RuleWriterAgent for a round's adaptation attempt."""
    active_rulebook: Rulebook
    proposals: Tuple[RuleProposal, ...]
    audit_records: Tuple[RuleAuditRecord, ...]
    accepted: bool
    summary: str


class RuleWriterAgent:
    """
    Evaluation-driven rule improvement agent.
    
    Identifies control weaknesses from diagnostic miss reports and proposes
    bounded parameter adjustments validated by the independent RuleValidator.
    """

    def __init__(
        self,
        validator: Optional[RuleValidator] = None,
        max_proposals_per_round: int = 1,
    ) -> None:
        if max_proposals_per_round < 1:
            raise ValueError(f"max_proposals_per_round must be >= 1, got {max_proposals_per_round}")
        self.validator = validator or RuleValidator()
        self.max_proposals_per_round = max_proposals_per_round

    def generate_proposals(
        self,
        active_rulebook: Rulebook,
        feedback: Optional[Union[CriticFeedbackSummary, CriticReport, Dict[str, Any]]],
        round_number: int = 1,
    ) -> List[RuleProposal]:
        """
        Analyze evaluation feedback and generate up to max_proposals_per_round
        justified, bounded candidate rule modifications.
        """
        if feedback is None:
            return []

        # Extract aggregate metrics safely
        if isinstance(feedback, CriticReport):
            summary = CriticFeedbackSummary.from_critic_report(feedback)
            missed_counts = summary.missed_typology_counts
        elif isinstance(feedback, CriticFeedbackSummary):
            missed_counts = feedback.missed_typology_counts
        elif isinstance(feedback, dict):
            missed_counts = feedback.get("missed_typology_counts", {})
        else:
            raise TypeError(f"Unsupported feedback type: {type(feedback).__name__}")

        if not missed_counts:
            return []

        # Sort missed typologies descending by miss count
        sorted_misses = sorted(missed_counts.items(), key=lambda kv: kv[1], reverse=True)

        proposals: List[RuleProposal] = []
        proposal_idx = 1

        for typology, count in sorted_misses:
            if len(proposals) >= self.max_proposals_per_round:
                break
            if count <= 0:
                continue

            # Identify candidate constraint for this typology
            cand_proposal = self._find_best_proposal(
                active_rulebook=active_rulebook,
                typology=typology,
                miss_count=count,
                round_number=round_number,
                proposal_idx=proposal_idx,
            )
            if cand_proposal is not None:
                proposals.append(cand_proposal)
                proposal_idx += 1

        return proposals

    def _find_best_proposal(
        self,
        active_rulebook: Rulebook,
        typology: str,
        miss_count: int,
        round_number: int,
        proposal_idx: int,
    ) -> Optional[RuleProposal]:
        """Find a feasible parameter adjustment directly addressing a missed typology."""
        # Mapping typologies to specific parameter tuning rules
        target_mappings: List[Tuple[str, str, str, str]] = [
            # (Typology, Rule ID, Parameter, Direction)
            (
                "EXAGGERATED_REPAIR",
                "RULE_REPAIR_BENCHMARK",
                "max_markup_ratio",
                "TIGHTEN",
            ),
            (
                "OPPORTUNISTIC_INFLATION",
                "RULE_LINE_ITEM_ARITHMETIC",
                "max_discrepancy_gbp",
                "TIGHTEN",
            ),
            (
                "STAGED_COLLISION",
                "RULE_STAGED_COLLISION_INDICATORS",
                "min_policy_age_days",
                "EXPAND",
            ),
        ]

        for target_typ, rule_id, param_name, direction in target_mappings:
            if typology != target_typ:
                continue

            constraint_key = (rule_id, param_name)
            if constraint_key not in SUPPORTED_CONSTRAINTS:
                continue

            constraint = SUPPORTED_CONSTRAINTS[constraint_key]

            try:
                rule_def = active_rulebook.get_rule(rule_id)
            except KeyError:
                continue

            if not rule_def.enabled:
                continue

            current_val = rule_def.parameters.get(param_name)
            if current_val is None:
                continue

            # Compute candidate value according to direction
            if direction == "TIGHTEN":
                cand_val = current_val - constraint.max_step_size
                if cand_val < constraint.min_value:
                    # Already at tightest allowed bound
                    continue
            elif direction == "EXPAND":
                cand_val = current_val + constraint.max_step_size
                if cand_val > constraint.max_value:
                    # Already at widest allowed bound
                    continue
            else:
                continue

            # Ensure correct numeric type
            final_cand_val = int(cand_val) if constraint.param_type is int else round(cand_val, 4)

            # Rationale explanation
            rationale = (
                f"Diagnosed {miss_count} missed {typology} claims in Round {round_number - 1}. "
                f"Proposing bounded {direction.lower()} of {rule_id}.{param_name} "
                f"from {current_val} to {final_cand_val} (allowed bounds: [{constraint.min_value}, {constraint.max_value}])."
            )

            expected_effect = (
                f"Increase sensitivity to {typology} patterns while verifying "
                f"false-positive rate remains within {self.validator.fp_budget:.1%} budget."
            )

            proposal_id = f"PROP-R{round_number:02d}-{proposal_idx:02d}"

            return RuleProposal(
                proposal_id=proposal_id,
                round_number=round_number,
                rule_id=rule_id,
                template_id=rule_def.template_id,
                parameter_name=param_name,
                old_value=current_val,
                candidate_value=final_cand_val,
                rationale=rationale,
                target_typology=typology,
                expected_effect=expected_effect,
            )

        return None

    def propose_and_validate(
        self,
        active_rulebook: Rulebook,
        feedback: Optional[Union[CriticFeedbackSummary, CriticReport, Dict[str, Any]]],
        round_number: int = 1,
    ) -> RuleWriterOutput:
        """
        Generate candidate rule proposals, submit to validation gate, and return
        the resulting active Rulebook and complete audit trail records.
        """
        proposals = self.generate_proposals(active_rulebook, feedback, round_number=round_number)

        if not proposals:
            return RuleWriterOutput(
                active_rulebook=active_rulebook.clone(),
                proposals=(),
                audit_records=(),
                accepted=False,
                summary=f"Round {round_number}: No justified parameter adjustments identified from feedback.",
            )

        audit_records: List[RuleAuditRecord] = []
        current_book = active_rulebook.clone()
        any_accepted = False

        for proposal in proposals:
            accepted, val_result, updated_book = self.validator.validate(current_book, proposal)

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if accepted and updated_book is not None:
                record = RuleAuditRecord(
                    proposal_id=proposal.proposal_id,
                    round_number=round_number,
                    rule_id=proposal.rule_id,
                    parameter_name=proposal.parameter_name,
                    old_value=proposal.old_value,
                    candidate_value=proposal.candidate_value,
                    rationale=proposal.rationale,
                    status="ACCEPTED",
                    gate_failed=None,
                    rejection_reason="",
                    validation_fp_rate=val_result.validation_fp_rate,
                    rulebook_version_before=current_book.version,
                    rulebook_version_after=updated_book.version,
                    timestamp=timestamp,
                )
                current_book = updated_book
                any_accepted = True
            else:
                record = RuleAuditRecord(
                    proposal_id=proposal.proposal_id,
                    round_number=round_number,
                    rule_id=proposal.rule_id,
                    parameter_name=proposal.parameter_name,
                    old_value=proposal.old_value,
                    candidate_value=proposal.candidate_value,
                    rationale=proposal.rationale,
                    status="REJECTED",
                    gate_failed=val_result.gate_failed,
                    rejection_reason=val_result.rejection_reason,
                    validation_fp_rate=val_result.validation_fp_rate,
                    rulebook_version_before=current_book.version,
                    rulebook_version_after=current_book.version,
                    timestamp=timestamp,
                )

            audit_records.append(record)

        summary = (
            f"Round {round_number}: {len(audit_records)} proposal(s) evaluated; "
            f"{'1 accepted, rulebook updated to v' + current_book.version if any_accepted else 'all rejected, rulebook maintained at v' + current_book.version}."
        )

        return RuleWriterOutput(
            active_rulebook=current_book,
            proposals=tuple(proposals),
            audit_records=tuple(audit_records),
            accepted=any_accepted,
            summary=summary,
        )
