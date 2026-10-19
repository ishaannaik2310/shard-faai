"""
Shard — Critic Agent and Evaluation Engine.

Evaluates investigation decisions against ground truth, performs independent
decision consistency validation, computes fraud detection metrics, and generates
diagnostic reports for the adaptation feedback loop.

ARCHITECTURAL TRUST BOUNDARY (TI-01, TI-06):
1. Authorised Ground-Truth Access: The Critic Agent is explicitly authorised to access
   the GroundTruthVault strictly on the post-investigation evaluation path.
2. Ground-Truth Isolation: Raw ground-truth records (is_fraud, fraud_typology, etc.)
   must NEVER be injected into ClaimDecision records, ordinary application logs,
   or returned to upstream investigation agents.
3. Independent Evaluation: The Critic independently evaluates decisions using structured
   labels rather than merely restating Investigator findings.
4. Non-Destructive: The Critic NEVER modifies, rewrites, or mutates ClaimDecision objects.
"""

from __future__ import annotations

from collections import defaultdict
import datetime
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from shard.core.discovery_models import DiscoveryFeedback, FeatureAggregateStats
from shard.core.features import FEATURE_WHITELIST, extract_feature
from shard.core.models import (
    Claim,
    ClaimDecision,
    ConsistencyViolation,
    CriticReport,
    Disposition,
    TypologyMetrics,
)
from shard.data.vault import GroundTruthVault
from shard.rules.rulebook import Rulebook, load_default_rulebook


def _quantile(sorted_vals: List[float], q: float) -> float:
    """Compute empirical quantile using linear interpolation."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (q / 100.0) * (n - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    weight = idx - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def _compute_feature_stats(
    values: Sequence[float],
    ref_p95: Optional[float] = None,
    ref_p5: Optional[float] = None,
    ref_mean: Optional[float] = None,
) -> FeatureAggregateStats:
    """Compute aggregate distribution statistics for a numeric observable feature."""
    if not values:
        return FeatureAggregateStats(
            count=0,
            mean=0.0,
            std=0.0,
            p5=0.0,
            p50=0.0,
            p95=0.0,
            p97=0.0,
            p99=0.0,
            min=0.0,
            max=0.0,
            tail_share_beyond_ref_p95=0.0,
        )

    vals = sorted(values)
    count = len(vals)
    mean = sum(vals) / count
    variance = sum((v - mean) ** 2 for v in vals) / (count - 1) if count > 1 else 0.0
    std = math.sqrt(max(0.0, variance))

    p5 = _quantile(vals, 5.0)
    p50 = _quantile(vals, 50.0)
    p95 = _quantile(vals, 95.0)
    p97 = _quantile(vals, 97.0)
    p99 = _quantile(vals, 99.0)
    min_val = vals[0]
    max_val = vals[-1]

    tail_share = 0.0
    if ref_p95 is not None and ref_p5 is not None and ref_mean is not None and count > 0:
        if mean >= ref_mean:
            tail_share = sum(1 for v in vals if v > ref_p95) / count
        else:
            tail_share = sum(1 for v in vals if v < ref_p5) / count

    return FeatureAggregateStats(
        count=count,
        mean=round(mean, 4),
        std=round(std, 4),
        p5=round(p5, 4),
        p50=round(p50, 4),
        p95=round(p95, 4),
        p97=round(p97, 4),
        p99=round(p99, 4),
        min=round(min_val, 4),
        max=round(max_val, 4),
        tail_share_beyond_ref_p95=round(tail_share, 4),
    )


class CriticAgent:
    """
    Independent evaluation agent with controlled access to GroundTruthVault.
    
    Validates decision consistency, calculates fraud detection metrics,
    diagnoses missed fraud patterns, and produces structured CriticReports.
    """

    def __init__(
        self,
        enable_discovery: bool = False,
        min_missed_samples: int = 8,
    ) -> None:
        self.enable_discovery = enable_discovery
        self.min_missed_samples = min_missed_samples
        self._accumulated_missed_claims: List[Claim] = []
        self._accumulated_reference_claims: List[Claim] = []
        self._seen_claim_ids: Set[str] = set()

    def reset_accumulated_evidence(self) -> None:
        """Reset accumulated missed and reference claims."""
        self._accumulated_missed_claims.clear()
        self._accumulated_reference_claims.clear()
        self._seen_claim_ids.clear()

    def evaluate(
        self,
        decisions: Sequence[ClaimDecision],
        vault: GroundTruthVault,
        claims: Optional[Sequence[Claim]] = None,
        rulebook: Optional[Rulebook] = None,
        round_number: int = 1,
        round_seed: Optional[int] = None,
    ) -> CriticReport:
        """
        Evaluate an ordered sequence of ClaimDecision records against the ground truth.

        Binary Classification Convention:
        - APPROVE = not flagged (negative prediction)
        - REFER = flagged for investigation (positive prediction)
        - SUSPECTED_FRAUD = flagged for investigation (positive prediction)

        Confusion Matrix:
        - TP: Fraudulent claim flagged (REFER or SUSPECTED_FRAUD)
        - FP: Genuine claim flagged (REFER or SUSPECTED_FRAUD)
        - TN: Genuine claim approved (APPROVE)
        - FN: Fraudulent claim approved (APPROVE)

        Zero-Denominator Convention:
        Returns 0.0 whenever any metric denominator is zero.
        """
        if not isinstance(vault, GroundTruthVault):
            raise TypeError(f"Expected GroundTruthVault instance, got {type(vault).__name__}")
        if not isinstance(decisions, (list, tuple)):
            raise TypeError(f"Expected sequence of ClaimDecision, got {type(decisions).__name__}")

        rb = rulebook or load_default_rulebook()
        if not isinstance(rb, Rulebook):
            raise TypeError(f"Expected Rulebook instance, got {type(rb).__name__}")

        claims_by_id: Dict[str, Claim] = {}
        if claims is not None:
            if not isinstance(claims, (list, tuple)):
                raise TypeError(f"Expected sequence of Claim, got {type(claims).__name__}")
            for c in claims:
                if not isinstance(c, Claim):
                    raise TypeError(f"All claims must be Claim instances, got {type(c).__name__}")
                claims_by_id[c.claim_id] = c

        # Determine target claim set
        target_ids: Set[str] = set(claims_by_id.keys()) if claims_by_id else set(vault.claim_ids())

        if self.enable_discovery and round_seed is not None:
            from shard.rules.validator import GATE_B_SEEDS
            if round_seed in GATE_B_SEEDS:
                raise AssertionError(
                    f"Round seed {round_seed} matches Gate B validation seeds {GATE_B_SEEDS}"
                )

        violations: List[ConsistencyViolation] = []
        seen_decision_ids: Set[str] = set()
        valid_decisions: List[ClaimDecision] = []

        # -------------------------------------------------------------------
        # 1. Independent Consistency Validation
        # -------------------------------------------------------------------
        for idx, decision in enumerate(decisions):
            if not isinstance(decision, ClaimDecision):
                violations.append(
                    ConsistencyViolation(
                        claim_id=getattr(decision, "claim_id", f"INDEX_{idx}"),
                        violation_type="STRUCTURAL_INVALIDITY",
                        explanation=f"Item at index {idx} is {type(decision).__name__}, expected ClaimDecision.",
                        severity="ERROR",
                    )
                )
                continue

            cid = decision.claim_id

            # Duplicate check
            if cid in seen_decision_ids:
                violations.append(
                    ConsistencyViolation(
                        claim_id=cid,
                        violation_type="DUPLICATE_DECISION",
                        explanation=f"Duplicate decision encountered for claim ID '{cid}'.",
                        severity="ERROR",
                    )
                )
                continue
            seen_decision_ids.add(cid)

            # Unknown claim ID check
            if not vault.contains(cid):
                violations.append(
                    ConsistencyViolation(
                        claim_id=cid,
                        violation_type="UNKNOWN_CLAIM_ID",
                        explanation=f"Decision references claim ID '{cid}' not present in GroundTruthVault.",
                        severity="ERROR",
                    )
                )
                continue

            if target_ids and cid not in target_ids:
                violations.append(
                    ConsistencyViolation(
                        claim_id=cid,
                        violation_type="UNKNOWN_CLAIM_ID",
                        explanation=f"Decision references claim ID '{cid}' not present in target evaluated claim set.",
                        severity="WARNING",
                    )
                )

            # Risk score range check (0.0 to 100.0)
            if not (0.0 <= decision.risk_score <= 100.0):
                violations.append(
                    ConsistencyViolation(
                        claim_id=cid,
                        violation_type="INVALID_RISK_SCORE",
                        explanation=f"Risk score {decision.risk_score} is outside permitted 0.0-100.0 scale.",
                        severity="ERROR",
                    )
                )

            # Findings non-empty check (TI-02)
            if not decision.findings:
                violations.append(
                    ConsistencyViolation(
                        claim_id=cid,
                        violation_type="EMPTY_FINDINGS",
                        explanation="ClaimDecision contains an empty findings list (TI-02 invariant violation).",
                        severity="ERROR",
                    )
                )

            # Valid rule IDs in findings
            valid_rule_ids = {r.rule_id for r in rb.rules}
            for finding in decision.findings:
                if finding.check_name not in valid_rule_ids:
                    violations.append(
                        ConsistencyViolation(
                            claim_id=cid,
                            violation_type="UNKNOWN_RULE_ID",
                            explanation=f"Finding check_name '{finding.check_name}' is not in Rulebook v{rb.version}.",
                            severity="ERROR",
                        )
                    )

            # Disposition threshold consistency check
            thresh = rb.disposition_thresholds
            if decision.risk_score < thresh.approve_below:
                expected_disp = Disposition.APPROVE
            elif decision.risk_score < thresh.suspected_fraud_at_or_above:
                expected_disp = Disposition.REFER
            else:
                expected_disp = Disposition.SUSPECTED_FRAUD

            if decision.disposition != expected_disp:
                violations.append(
                    ConsistencyViolation(
                        claim_id=cid,
                        violation_type="DISPOSITION_THRESHOLD_MISMATCH",
                        explanation=(
                            f"Risk score {decision.risk_score:.2f} maps to {expected_disp.value} under Rulebook "
                            f"(approve < {thresh.approve_below}, suspected >= {thresh.suspected_fraud_at_or_above}), "
                            f"but decision has {decision.disposition.value}."
                        ),
                        severity="ERROR",
                    )
                )

            # Tool trace structural check
            if not isinstance(decision.tool_trace, (list, tuple)):
                violations.append(
                    ConsistencyViolation(
                        claim_id=cid,
                        violation_type="MALFORMED_TOOL_TRACE",
                        explanation="Decision tool_trace must be a tuple or list of trace dictionaries.",
                        severity="ERROR",
                    )
                )
            else:
                for step_i, step in enumerate(decision.tool_trace):
                    if not isinstance(step, dict) or "rule_id" not in step:
                        violations.append(
                            ConsistencyViolation(
                                claim_id=cid,
                                violation_type="MALFORMED_TOOL_TRACE",
                                explanation=f"tool_trace entry at index {step_i} is malformed or missing 'rule_id'.",
                                severity="ERROR",
                            )
                        )
                        break

            valid_decisions.append(decision)

        # Missing decisions check
        for target_id in target_ids:
            if target_id not in seen_decision_ids:
                violations.append(
                    ConsistencyViolation(
                        claim_id=target_id,
                        violation_type="MISSING_DECISION",
                        explanation=f"Claim ID '{target_id}' was expected in evaluation but no ClaimDecision was provided.",
                        severity="ERROR",
                    )
                )

        # -------------------------------------------------------------------
        # 2. Metric Calculation (TI-03)
        # -------------------------------------------------------------------
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        disposition_counts: Dict[str, int] = {
            Disposition.APPROVE.value: 0,
            Disposition.REFER.value: 0,
            Disposition.SUSPECTED_FRAUD.value: 0,
        }

        flagged_fraud_ids: List[str] = []
        missed_fraud_ids: List[str] = []
        false_alarm_ids: List[str] = []
        approved_genuine_ids: List[str] = []

        typology_totals: Dict[str, int] = defaultdict(int)
        typology_caught: Dict[str, int] = defaultdict(int)

        flagged_fraud_gbp = 0.0
        missed_fraud_gbp = 0.0
        flagged_genuine_gbp = 0.0
        approved_genuine_gbp = 0.0

        for decision in valid_decisions:
            gt = vault.get_label(decision.claim_id)
            is_flagged = decision.disposition in (Disposition.REFER, Disposition.SUSPECTED_FRAUD)
            disp_key = decision.disposition.value
            disposition_counts[disp_key] = disposition_counts.get(disp_key, 0) + 1

            claimed_amt = claims_by_id[decision.claim_id].claimed_amount if decision.claim_id in claims_by_id else 0.0

            if gt.is_fraud:
                typ = gt.fraud_typology or "UNSPECIFIED_FRAUD"
                typology_totals[typ] += 1
                if is_flagged:
                    tp += 1
                    flagged_fraud_ids.append(decision.claim_id)
                    typology_caught[typ] += 1
                    flagged_fraud_gbp += claimed_amt
                else:
                    fn += 1
                    missed_fraud_ids.append(decision.claim_id)
                    missed_fraud_gbp += claimed_amt
            else:
                if is_flagged:
                    fp += 1
                    false_alarm_ids.append(decision.claim_id)
                    flagged_genuine_gbp += claimed_amt
                else:
                    tn += 1
                    approved_genuine_ids.append(decision.claim_id)
                    approved_genuine_gbp += claimed_amt

            if (
                self.enable_discovery
                and decision.claim_id not in self._seen_claim_ids
                and decision.claim_id in claims_by_id
            ):
                self._seen_claim_ids.add(decision.claim_id)
                claim_obj = claims_by_id[decision.claim_id]
                if gt.is_fraud and not is_flagged:
                    self._accumulated_missed_claims.append(claim_obj)
                elif not gt.is_fraud:
                    self._accumulated_reference_claims.append(claim_obj)

        total_evaluated = tp + fp + tn + fn

        # Safe metric divisions with documented 0.0 default for zero denominators
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = (
            round(2.0 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0.0
            else 0.0
        )
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
        accuracy = round((tp + tn) / total_evaluated, 4) if total_evaluated > 0 else 0.0

        # Typology breakdown
        typology_metrics: Dict[str, TypologyMetrics] = {}
        missed_typology_counts: Dict[str, int] = {}

        for typ, tot in typology_totals.items():
            caught = typology_caught[typ]
            missed = tot - caught
            rate = round(caught / tot, 4) if tot > 0 else 0.0
            typology_metrics[typ] = TypologyMetrics(
                typology=typ,
                total_cases=tot,
                caught_cases=caught,
                missed_cases=missed,
                detection_rate=rate,
            )
            if missed > 0:
                missed_typology_counts[typ] = missed

        # Financial exposure
        total_claimed_gbp = (
            flagged_fraud_gbp + missed_fraud_gbp + flagged_genuine_gbp + approved_genuine_gbp
        )
        financial_exposure = {
            "flagged_fraud_gbp": round(flagged_fraud_gbp, 2),
            "missed_fraud_gbp": round(missed_fraud_gbp, 2),
            "flagged_genuine_gbp": round(flagged_genuine_gbp, 2),
            "approved_genuine_gbp": round(approved_genuine_gbp, 2),
            "total_claimed_gbp": round(total_claimed_gbp, 2),
        }

        # Discovery feedback computation
        discovery_feedback: Optional[DiscoveryFeedback] = None
        if self.enable_discovery:
            if self._accumulated_reference_claims:
                from shard.rules.validator import get_default_validation_dataset, GATE_B_V2_SEED
                gate_b_claims = get_default_validation_dataset(seed=GATE_B_V2_SEED)
                gate_b_ids = {c.claim_id for c in gate_b_claims}
                ref_ids = {c.claim_id for c in self._accumulated_reference_claims}
                assert ref_ids.isdisjoint(gate_b_ids), (
                    f"Discovery reference claims overlap with Gate B validation claims: {ref_ids & gate_b_ids}"
                )

            n_missed = len(self._accumulated_missed_claims)
            n_reference = len(self._accumulated_reference_claims)
            has_sufficient = n_missed >= self.min_missed_samples

            features_dict: Dict[str, Tuple[FeatureAggregateStats, FeatureAggregateStats]] = {}
            for feat_name in sorted(FEATURE_WHITELIST.keys()):
                m_vals = [
                    extract_feature(c, feat_name) for c in self._accumulated_missed_claims
                ]
                m_vals_clean = [v for v in m_vals if v is not None and math.isfinite(v)]

                r_vals = [
                    extract_feature(c, feat_name) for c in self._accumulated_reference_claims
                ]
                r_vals_clean = [v for v in r_vals if v is not None and math.isfinite(v)]

                r_stats = _compute_feature_stats(r_vals_clean)
                m_stats = _compute_feature_stats(
                    m_vals_clean,
                    ref_p95=r_stats.p95,
                    ref_p5=r_stats.p5,
                    ref_mean=r_stats.mean,
                )
                features_dict[feat_name] = (m_stats, r_stats)

            discovery_feedback = DiscoveryFeedback(
                round_number=round_number,
                n_missed=n_missed,
                n_reference=n_reference,
                features=features_dict,
                min_missed_samples=self.min_missed_samples,
                has_sufficient_evidence=has_sufficient,
            )

        generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return CriticReport(
            total_evaluated=total_evaluated,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            accuracy=accuracy,
            disposition_counts=disposition_counts,
            typology_metrics=typology_metrics,
            missed_typology_counts=missed_typology_counts,
            consistency_violations=tuple(violations),
            financial_exposure=financial_exposure,
            flagged_fraud_claim_ids=tuple(flagged_fraud_ids),
            missed_fraud_claim_ids=tuple(missed_fraud_ids),
            false_alarm_claim_ids=tuple(false_alarm_ids),
            approved_genuine_claim_ids=tuple(approved_genuine_ids),
            round_number=round_number,
            generated_at=generated_at,
            discovery_feedback=discovery_feedback,
        )
