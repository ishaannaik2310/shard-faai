"""
Shard — Core Models and Data Contracts.

This module defines typed data structures for observable claims, ground-truth labels,
and investigation decisions.

CRITICAL INVARIANT (TI-01, TI-11):
- `Claim` represents observable evidence only and MUST NEVER contain ground-truth labels
  (`is_fraud`, `fraud_typology`, etc.).
- `GroundTruthRecord` is stored exclusively in `GroundTruthVault` and isolated from
  investigation agents (Planner, Research, Investigator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class IncidentType(str, Enum):
    """Categorisation of vehicle incidents."""
    COLLISION = "COLLISION"
    THEFT = "THEFT"
    WEATHER = "WEATHER"
    FIRE = "FIRE"


class VehicleRegistrationStatus(str, Enum):
    """Registration authority database status."""
    VALID = "VALID"
    UNREGISTERED = "UNREGISTERED"
    SUSPENDED = "SUSPENDED"


class VehicleWriteOffStatus(str, Enum):
    """UK insurance write-off category markers."""
    CLEAN = "CLEAN"      # No prior write-off record
    CAT_A = "CAT_A"      # Scrap only; entire vehicle must be crushed
    CAT_B = "CAT_B"      # Break for parts only; chassis/body must be crushed
    CAT_S = "CAT_S"      # Structurally damaged but repairable
    CAT_N = "CAT_N"      # Non-structurally damaged and repairable


class FraudTypology(str, Enum):
    """
    Agreed motor insurance fraud typologies within MVP scope.
    Used strictly in GroundTruthRecord and adaptation feedback.
    """
    EXAGGERATED_REPAIR = "EXAGGERATED_REPAIR"
    STAGED_COLLISION = "STAGED_COLLISION"
    GHOST_VEHICLE = "GHOST_VEHICLE"
    PRE_EXISTING_DAMAGE = "PRE_EXISTING_DAMAGE"
    OPPORTUNISTIC_INFLATION = "OPPORTUNISTIC_INFLATION"
    DELAYED_FILING = "DELAYED_FILING"


class Disposition(str, Enum):
    """Investigation recommendation dispositions."""
    APPROVE = "APPROVE"
    REFER = "REFER"                      # Intermediate risk -> mandatory human review (TI-08)
    SUSPECTED_FRAUD = "SUSPECTED_FRAUD"  # Recommendation for fraud escalation (TI-11)


@dataclass(frozen=True)
class RepairLineItem:
    """Itemised component or labour entry from a garage repair quote."""
    part_or_operation: str
    cost: float  # In GBP, non-negative, 2 decimal places
    category: str = "PARTS"  # PARTS, LABOUR, PAINT, MISC


@dataclass(frozen=True)
class Claim:
    """
    Observable motor insurance claim payload presented to investigation agents.

    CRITICAL INVARIANT:
    This model MUST NEVER contain hidden ground truth (is_fraud, fraud_typology, etc.).
    All fields represent observable data that an insurance claims department receives.
    """
    claim_id: str
    policy_id: str
    claimant_id: str
    policy_inception_date: str          # ISO 8601 YYYY-MM-DD
    incident_date: str                  # ISO 8601 YYYY-MM-DD
    filing_date: str                    # ISO 8601 YYYY-MM-DD
    incident_type: str                  # Value from IncidentType
    incident_description: str
    vehicle_registration: str           # UK format e.g. AB12 CDE
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    vehicle_registration_status: str   # Value from VehicleRegistrationStatus
    vehicle_write_off_status: str      # Value from VehicleWriteOffStatus
    prior_damage_reported: bool         # Self-reported prior damage indicator
    claimed_amount: float              # Claimed repair total in GBP, non-negative
    benchmark_repair_cost: float       # Independent benchmark repair cost in GBP, non-negative
    repair_provider_id: str            # Identifier of the servicing garage / bodyshop
    repair_line_items: Tuple[RepairLineItem, ...] = field(default_factory=tuple)
    linked_parties: Tuple[str, ...] = field(default_factory=tuple)  # Co-claimants / third parties


@dataclass(frozen=True)
class GroundTruthRecord:
    """
    Hidden evaluation label for a synthetic claim.

    Stored strictly in GroundTruthVault and never attached to observable Claim instances.
    """
    claim_id: str
    is_fraud: bool
    fraud_typology: Optional[str] = None  # FraudTypology value or None for genuine
    detail: str = ""                       # Explanation of the synthetic fraud indicator


@dataclass(frozen=True)
class Finding:
    """Individual finding produced by a deterministic verification tool."""
    check_name: str
    result: str
    risk_contribution: float  # Bounded score contribution points (0.0 to 100.0)
    detail: str


@dataclass(frozen=True)
class ClaimDecision:
    """
    Final investigation recommendation for a claim.

    CRITICAL INVARIANT (TI-02):
    findings must be non-empty for every evaluated claim.
    Score convention: 0.0 to 100.0.
    """
    claim_id: str
    risk_score: float                  # Computed risk score (0.0 to 100.0)
    disposition: Disposition           # APPROVE (<25), REFER (25-59), or SUSPECTED_FRAUD (>=60)
    findings: Tuple[Finding, ...]      # Non-empty list of audit findings
    tool_trace: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    round_number: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= self.risk_score <= 100.0):
            raise ValueError(f"risk_score must be between 0.0 and 100.0, got {self.risk_score}")
        if not self.findings:
            raise ValueError("ClaimDecision must contain at least one Finding (TI-02 invariant)")
        if not isinstance(self.disposition, Disposition):
            raise TypeError(f"disposition must be a Disposition enum, got {type(self.disposition).__name__}")


@dataclass(frozen=True)
class InvestigationPlan:
    """
    Structured plan produced by the Planner Agent for a specific claim.

    CRITICAL INVARIANT:
    Planner works purely on observable claim attributes. No ground-truth labels.
    """
    claim_id: str
    tasks: Tuple[str, ...]                   # Ordered verification rule IDs to execute
    rationale: str                           # Explanation of task selection and ordering strategy
    task_reasons: Dict[str, str] = field(default_factory=dict)  # Why each task was selected
    missing_evidence_requirements: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.claim_id or not isinstance(self.claim_id, str):
            raise ValueError("InvestigationPlan claim_id must be a non-empty string")
        if not self.tasks:
            raise ValueError("InvestigationPlan tasks must contain at least one task")


@dataclass(frozen=True)
class EvidenceItem:
    """An individual piece of synthetic evidence retrieved by the Research Agent."""
    key: str
    value: Any
    source: str                              # Provenance: e.g. CLAIM_SUBMISSION, ESTIMATICS_BENCHMARK, etc.
    is_available: bool = True
    notes: str = ""


@dataclass(frozen=True)
class EvidenceBundle:
    """
    Structured evidence dossier gathered by the Research Agent for an investigation plan.
    """
    claim_id: str
    items: Dict[str, EvidenceItem] = field(default_factory=dict)
    retrieved_at: str = ""
    is_complete: bool = True

    def __post_init__(self) -> None:
        if not self.claim_id or not isinstance(self.claim_id, str):
            raise ValueError("EvidenceBundle claim_id must be a non-empty string")

    def get(self, key: str) -> Optional[EvidenceItem]:
        return self.items.get(key)

    def get_value(self, key: str, default: Any = None) -> Any:
        item = self.items.get(key)
        if item is not None and item.is_available:
            return item.value
        return default


@dataclass(frozen=True)
class ConsistencyViolation:
    """Structured record of an audit or consistency anomaly detected in a ClaimDecision."""
    claim_id: str
    violation_type: str
    explanation: str
    severity: str = "ERROR"

    def to_dict(self) -> Dict[str, str]:
        return {
            "claim_id": self.claim_id,
            "violation_type": self.violation_type,
            "explanation": self.explanation,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class TypologyMetrics:
    """Evaluation breakdown for a specific synthetic fraud typology."""
    typology: str
    total_cases: int
    caught_cases: int
    missed_cases: int
    detection_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "typology": self.typology,
            "total_cases": self.total_cases,
            "caught_cases": self.caught_cases,
            "missed_cases": self.missed_cases,
            "detection_rate": self.detection_rate,
        }


@dataclass(frozen=True)
class CriticReport:
    """
    Structured, serialisable evaluation report produced by the Critic Agent.

    CRITICAL INVARIANTS:
    1. Independent Metrics (TI-03): Derived strictly from actual predictions vs ground truth.
    2. Data Isolation (TI-01, TI-06): Contains aggregate statistics and diagnostic claim IDs,
       never raw GroundTruthRecord objects.
    3. Non-Destructive: Critic never mutates original ClaimDecision records.
    """
    total_evaluated: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    accuracy: float
    disposition_counts: Dict[str, int]
    typology_metrics: Dict[str, TypologyMetrics]
    missed_typology_counts: Dict[str, int]
    consistency_violations: Tuple[ConsistencyViolation, ...]
    financial_exposure: Dict[str, float] = field(default_factory=dict)
    flagged_fraud_claim_ids: Tuple[str, ...] = field(default_factory=tuple)
    missed_fraud_claim_ids: Tuple[str, ...] = field(default_factory=tuple)
    false_alarm_claim_ids: Tuple[str, ...] = field(default_factory=tuple)
    approved_genuine_claim_ids: Tuple[str, ...] = field(default_factory=tuple)
    round_number: int = 1
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "total_evaluated": self.total_evaluated,
            "confusion_matrix": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
            },
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "fpr": self.fpr,
                "accuracy": self.accuracy,
            },
            "disposition_counts": dict(self.disposition_counts),
            "typology_metrics": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.typology_metrics.items()
            },
            "missed_typology_counts": dict(self.missed_typology_counts),
            "financial_exposure": dict(self.financial_exposure),
            "flagged_fraud_claim_ids": list(self.flagged_fraud_claim_ids),
            "missed_fraud_claim_ids": list(self.missed_fraud_claim_ids),
            "false_alarm_claim_ids": list(self.false_alarm_claim_ids),
            "approved_genuine_claim_ids": list(self.approved_genuine_claim_ids),
            "consistency_violations": [
                v.to_dict() if hasattr(v, "to_dict") else v
                for v in self.consistency_violations
            ],
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class CriticFeedbackSummary:
    """
    Redacted aggregate evaluation feedback permitted to cross the trust boundary
    to adaptive agents (Fraudster and Rule-Writer).
    
    CRITICAL INVARIANT (TI-01, TI-06):
    Contains aggregate metrics and typology miss counts only; NEVER raw GroundTruthRecord objects.
    """
    round_number: int
    missed_typology_counts: Dict[str, int]
    typology_detection_rates: Dict[str, float]
    total_evaluated: int
    overall_recall: float

    @classmethod
    def from_critic_report(cls, report: CriticReport) -> CriticFeedbackSummary:
        """Extract redacted aggregate summary from a CriticReport."""
        rates = {
            k: v.detection_rate for k, v in report.typology_metrics.items()
        }
        return cls(
            round_number=report.round_number,
            missed_typology_counts=dict(report.missed_typology_counts),
            typology_detection_rates=rates,
            total_evaluated=report.total_evaluated,
            overall_recall=report.recall,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "missed_typology_counts": dict(self.missed_typology_counts),
            "typology_detection_rates": dict(self.typology_detection_rates),
            "total_evaluated": self.total_evaluated,
            "overall_recall": self.overall_recall,
        }


@dataclass(frozen=True)
class FraudsterAdaptationRecord:
    """Auditable log of Fraudster sampling weight shifts across rounds."""
    round_number: int
    base_seed: int
    derived_seed: int
    prior_weights: Dict[str, float]
    updated_weights: Dict[str, float]
    feedback_summary: Dict[str, Any]
    adapted: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "base_seed": self.base_seed,
            "derived_seed": self.derived_seed,
            "prior_weights": dict(self.prior_weights),
            "updated_weights": dict(self.updated_weights),
            "feedback_summary": dict(self.feedback_summary),
            "adapted": self.adapted,
        }


@dataclass(frozen=True)
class RuleProposal:
    """A bounded candidate modification to a specific Rulebook parameter."""
    proposal_id: str
    round_number: int
    rule_id: str
    template_id: str
    parameter_name: str
    old_value: Any
    candidate_value: Any
    rationale: str
    target_typology: str
    expected_effect: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "round_number": self.round_number,
            "rule_id": self.rule_id,
            "template_id": self.template_id,
            "parameter_name": self.parameter_name,
            "old_value": self.old_value,
            "candidate_value": self.candidate_value,
            "rationale": self.rationale,
            "target_typology": self.target_typology,
            "expected_effect": self.expected_effect,
        }


@dataclass(frozen=True)
class RuleValidationResult:
    """Outcome of evaluating a candidate rule change against validation gates."""
    passed: bool
    gate_failed: Optional[str] = None  # GATE_A_CONFIG, GATE_B_FP_BUDGET, GATE_C_REGRESSION
    rejection_reason: str = ""
    validation_fp_rate: float = 0.0
    fp_budget: float = 0.10
    golden_set_passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gate_failed": self.gate_failed,
            "rejection_reason": self.rejection_reason,
            "validation_fp_rate": self.validation_fp_rate,
            "fp_budget": self.fp_budget,
            "golden_set_passed": self.golden_set_passed,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RuleAuditRecord:
    """Structured audit trail record for an accepted or rejected rule update."""
    proposal_id: str
    round_number: int
    rule_id: str
    parameter_name: str
    old_value: Any
    candidate_value: Any
    rationale: str
    status: str  # ACCEPTED | REJECTED
    gate_failed: Optional[str] = None
    rejection_reason: str = ""
    validation_fp_rate: float = 0.0
    rulebook_version_before: str = ""
    rulebook_version_after: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "round_number": self.round_number,
            "rule_id": self.rule_id,
            "parameter_name": self.parameter_name,
            "old_value": self.old_value,
            "candidate_value": self.candidate_value,
            "rationale": self.rationale,
            "status": self.status,
            "gate_failed": self.gate_failed,
            "rejection_reason": self.rejection_reason,
            "validation_fp_rate": self.validation_fp_rate,
            "rulebook_version_before": self.rulebook_version_before,
            "rulebook_version_after": self.rulebook_version_after,
            "timestamp": self.timestamp,
        }



