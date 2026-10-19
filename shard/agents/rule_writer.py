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
    discovery_ranking: Optional[Dict[str, Any]] = None


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
        discovery_feedback: Optional[Any] = None,
    ) -> RuleWriterOutput:
        """
        Generate candidate rule proposals, submit to validation gate, and return
        the resulting active Rulebook and complete audit trail records.
        """
        proposals = self.generate_proposals(active_rulebook, feedback, round_number=round_number)

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

        # If baseline templates did not yield an accepted update and discovery feedback is available, run generic discovery
        discovery_ranking_payload: Optional[Dict[str, Any]] = None
        if (
            not any_accepted
            and discovery_feedback is not None
            and getattr(discovery_feedback, "has_sufficient_evidence", False)
        ):
            from shard.discovery.ranking import rank_features
            ranked_features, rank_status = rank_features(discovery_feedback)
            if rank_status == "OK" and ranked_features:
                discovery_ranking_payload = {
                    "ranked_features": [rf.to_dict() for rf in ranked_features],
                    "chosen_feature": ranked_features[0].feature_name,
                    "n_missed": discovery_feedback.n_missed,
                    "n_reference": discovery_feedback.n_reference,
                    "round_number": round_number,
                }
                top = ranked_features[0]
                missed_stats, ref_stats = discovery_feedback.features[top.feature_name]

                # Derive candidates from reference distribution ordered tightest to loosest
                if top.direction == ">":
                    raw_candidates = [
                        ("p95", ref_stats.p95),
                        ("p97", ref_stats.p97),
                        ("p99", ref_stats.p99),
                        ("max", ref_stats.max),
                    ]
                else:
                    raw_candidates = [
                        ("p5", ref_stats.p5),
                        ("min", ref_stats.min),
                    ]

                seen_thresh_vals = set()
                ordered_candidates: List[Tuple[str, float]] = []
                for q_name, q_val in raw_candidates:
                    val = round(q_val, 2) if top.feature_name == "markup_ratio" else float(round(q_val))
                    if val not in seen_thresh_vals:
                        seen_thresh_vals.add(val)
                        ordered_candidates.append((q_name, val))

                existing_disc_count = sum(
                    1 for r in current_book.rules if r.rule_id.startswith("RULE_DISCOVERED_")
                )
                rule_id = f"RULE_DISCOVERED_{existing_disc_count + 1:03d}"

                discovery_proposals: List[RuleProposal] = []
                for q_name, thresh_val in ordered_candidates:
                    proposal_id = f"PROP-DISC-{round_number:02d}-{top.feature_name}-{q_name}"
                    rationale = (
                        f"Discovered rule candidate from {top.feature_name} (separation score {top.separation_score:.4f}). "
                        f"Evaluating threshold {thresh_val} from reference {q_name}."
                    )
                    disc_proposal = RuleProposal(
                        proposal_id=proposal_id,
                        round_number=round_number,
                        rule_id=rule_id,
                        template_id="tpl_feature_threshold",
                        parameter_name="threshold",
                        old_value=None,
                        candidate_value=thresh_val,
                        rationale=rationale,
                        target_typology="DISCOVERED_PATTERN",
                        expected_effect=f"Flag claims with {top.feature_name} {top.direction} {thresh_val}",
                        rule_parameters={
                            "feature": top.feature_name,
                            "direction": top.direction,
                            "threshold": thresh_val,
                            "score_contribution": 25.0,
                            "severity": "HIGH",
                        },
                    )
                    discovery_proposals.append(disc_proposal)

                    accepted, val_result, updated_book = self.validator.validate(current_book, disc_proposal)
                    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

                    if accepted and updated_book is not None:
                        rec = RuleAuditRecord(
                            proposal_id=disc_proposal.proposal_id,
                            round_number=round_number,
                            rule_id=disc_proposal.rule_id,
                            parameter_name=disc_proposal.parameter_name,
                            old_value=disc_proposal.old_value,
                            candidate_value=disc_proposal.candidate_value,
                            rationale=(
                                f"Discovered rule for observable feature '{top.feature_name}' (separation score: {top.separation_score:.4f}). "
                                f"Chosen threshold {thresh_val} from reference {q_name} (Gate B false-positive rate: {val_result.validation_fp_rate:.1%})."
                            ),
                            status="ACCEPTED",
                            gate_failed=None,
                            rejection_reason="",
                            validation_fp_rate=val_result.validation_fp_rate,
                            rulebook_version_before=current_book.version,
                            rulebook_version_after=updated_book.version,
                            timestamp=timestamp,
                        )
                        audit_records.append(rec)
                        current_book = updated_book
                        any_accepted = True
                        break  # Install first passing candidate and stop
                    else:
                        rec = RuleAuditRecord(
                            proposal_id=disc_proposal.proposal_id,
                            round_number=round_number,
                            rule_id=disc_proposal.rule_id,
                            parameter_name=disc_proposal.parameter_name,
                            old_value=disc_proposal.old_value,
                            candidate_value=disc_proposal.candidate_value,
                            rationale=disc_proposal.rationale,
                            status="REJECTED",
                            gate_failed=val_result.gate_failed,
                            rejection_reason=val_result.rejection_reason,
                            validation_fp_rate=val_result.validation_fp_rate,
                            rulebook_version_before=current_book.version,
                            rulebook_version_after=current_book.version,
                            timestamp=timestamp,
                        )
                        audit_records.append(rec)

                proposals.extend(discovery_proposals)

        if not audit_records:
            return RuleWriterOutput(
                active_rulebook=active_rulebook.clone(),
                proposals=(),
                audit_records=(),
                accepted=False,
                summary=f"Round {round_number}: No justified parameter adjustments identified from feedback.",
                discovery_ranking=discovery_ranking_payload,
            )

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
            discovery_ranking=discovery_ranking_payload,
        )
