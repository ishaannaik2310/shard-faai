"""
Shard — Independent Rule Validation Gate.

Implements multi-stage validation for candidate rulebook updates:
- Gate A: Configuration validity and bounds checking.
- Gate B: False-positive budget enforcement on a dedicated genuine-only validation set.
- Gate C: Golden-set regression testing on deterministic benchmark claims.
- Gate D: Safe, bounded activation with version increment and audit logging.

CRITICAL INVARIANTS:
1. Ground-Truth Isolation (TI-01, TI-06): The validation gate evaluates candidate rules
   on observable Claim instances only; it NEVER accesses the evaluation GroundTruthVault.
2. Safe Updating (TI-04, TI-05): Parameter updates use bounded templates only; no dynamic
   code or eval/exec is ever allowed. Updates exceeding the FP budget are strictly rejected.
3. Immutability on Rejection: If any validation gate fails, the active rulebook is preserved
   100% untouched.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from shard.core.models import (
    Claim,
    Disposition,
    FraudTypology,
    IncidentType,
    RepairLineItem,
    RuleProposal,
    RuleValidationResult,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)
from shard.data.generator import ClaimGenerator, GeneratorConfig
from shard.rules.rulebook import Rulebook
from shard.rules.templates import TEMPLATE_REGISTRY


# ---------------------------------------------------------------------------
# Parameter Constraints Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterConstraint:
    """Safe, bounded operational limits for an individual rule parameter."""
    rule_id: str
    parameter_name: str
    param_type: type
    min_value: float
    max_value: float
    max_step_size: float
    target_typologies: Tuple[str, ...]


SUPPORTED_CONSTRAINTS: Dict[Tuple[str, str], ParameterConstraint] = {
    ("RULE_REPAIR_BENCHMARK", "max_markup_ratio"): ParameterConstraint(
        rule_id="RULE_REPAIR_BENCHMARK",
        parameter_name="max_markup_ratio",
        param_type=float,
        min_value=1.10,
        max_value=1.50,
        max_step_size=0.05,
        target_typologies=(
            FraudTypology.EXAGGERATED_REPAIR.value,
            FraudTypology.OPPORTUNISTIC_INFLATION.value,
        ),
    ),
    ("RULE_LINE_ITEM_ARITHMETIC", "max_discrepancy_gbp"): ParameterConstraint(
        rule_id="RULE_LINE_ITEM_ARITHMETIC",
        parameter_name="max_discrepancy_gbp",
        param_type=float,
        min_value=0.0,
        max_value=5.0,
        max_step_size=0.50,
        target_typologies=(
            FraudTypology.OPPORTUNISTIC_INFLATION.value,
        ),
    ),
    ("RULE_STAGED_COLLISION_INDICATORS", "min_policy_age_days"): ParameterConstraint(
        rule_id="RULE_STAGED_COLLISION_INDICATORS",
        parameter_name="min_policy_age_days",
        param_type=int,
        min_value=5,
        max_value=30,
        max_step_size=5.0,
        target_typologies=(
            FraudTypology.STAGED_COLLISION.value,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Golden-Set Regression Definitions
# ---------------------------------------------------------------------------

def build_golden_regression_set() -> Tuple[Tuple[Claim, str, str], ...]:
    """
    Construct a deterministic golden regression suite of benchmark claims.
    Each item is a tuple of: (Claim, expected_disposition_or_flag, required_rule_triggered).
    """
    # 1. Clean genuine claim (must approve, score 0.0)
    clean_claim = Claim(
        claim_id="GOLDEN-CLEAN-01",
        policy_id="POL-G-01",
        claimant_id="CLMT-G-01",
        policy_inception_date="2025-01-01",
        incident_date="2026-03-01",
        filing_date="2026-03-03",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Queue shunt at mini-roundabout.",
        vehicle_registration="GF19 XKR",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
        prior_damage_reported=False,
        claimed_amount=1000.0,
        benchmark_repair_cost=1000.0,
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(
            RepairLineItem("Bumper respray", 600.0, "PAINT"),
            RepairLineItem("Fitting labour", 400.0, "LABOUR"),
        ),
        linked_parties=(),
    )

    # 2. Obvious repair cost markup (£3000 claimed vs £1000 benchmark -> 3.0x markup)
    cost_inflated_claim = Claim(
        claim_id="GOLDEN-COST-01",
        policy_id="POL-G-02",
        claimant_id="CLMT-G-02",
        policy_inception_date="2025-01-01",
        incident_date="2026-03-01",
        filing_date="2026-03-03",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Minor cosmetic damage with inflated bodyshop quote.",
        vehicle_registration="GF19 XKR",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
        prior_damage_reported=False,
        claimed_amount=3000.0,
        benchmark_repair_cost=1000.0,
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(RepairLineItem("Overpriced panel repair", 3000.0, "PARTS"),),
        linked_parties=(),
    )

    # 3. Severe non-repairable write-off marker (CAT_A scrap)
    write_off_claim = Claim(
        claim_id="GOLDEN-WRITEOFF-01",
        policy_id="POL-G-03",
        claimant_id="CLMT-G-03",
        policy_inception_date="2025-01-01",
        incident_date="2026-03-01",
        filing_date="2026-03-03",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Claim against pre-existing crushed vehicle record.",
        vehicle_registration="GF19 XKR",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CAT_A.value,
        prior_damage_reported=False,
        claimed_amount=1200.0,
        benchmark_repair_cost=1100.0,
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(RepairLineItem("Bumper respray", 1200.0, "PAINT"),),
        linked_parties=(),
    )

    # 4. Rapid inception staged collision (tenure 3d, 2 linked parties)
    staged_claim = Claim(
        claim_id="GOLDEN-STAGED-01",
        policy_id="POL-G-04",
        claimant_id="CLMT-G-04",
        policy_inception_date="2026-03-01",
        incident_date="2026-03-04",  # 3 days post-inception
        filing_date="2026-03-05",
        incident_type=IncidentType.COLLISION.value,
        incident_description="Sudden braking collision with multiple occupants.",
        vehicle_registration="GF19 XKR",
        vehicle_make="Ford",
        vehicle_model="Focus",
        vehicle_year=2019,
        vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
        vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
        prior_damage_reported=False,
        claimed_amount=1200.0,
        benchmark_repair_cost=1100.0,
        repair_provider_id="PRV-NORM-101",
        repair_line_items=(RepairLineItem("Bumper respray", 1200.0, "PAINT"),),
        linked_parties=("CLMT-L1", "CLMT-L2"),
    )

    return (
        (clean_claim, "APPROVE", "NONE"),
        (cost_inflated_claim, "FLAGGED", "RULE_REPAIR_BENCHMARK"),
        (write_off_claim, "FLAGGED", "RULE_WRITE_OFF_STATUS"),
        (staged_claim, "FLAGGED", "RULE_STAGED_COLLISION_INDICATORS"),
    )


# ---------------------------------------------------------------------------
# Validation Dataset Generator
# ---------------------------------------------------------------------------

_CACHED_VALIDATION_CLAIMS: Optional[Tuple[Claim, ...]] = None


def get_default_validation_dataset(n_claims: int = 40, seed: int = 9999) -> Tuple[Claim, ...]:
    """
    Generate or return a cached, fixed validation dataset consisting exclusively
    of genuine synthetic claims (fraud_share=0.0).
    
    Guarantees the exact same validation claims are used across candidate comparisons.
    """
    global _CACHED_VALIDATION_CLAIMS
    if _CACHED_VALIDATION_CLAIMS is None or len(_CACHED_VALIDATION_CLAIMS) != n_claims:
        cfg = GeneratorConfig(seed=seed, n_claims=n_claims, fraud_share=0.0)
        batch = ClaimGenerator(cfg).generate()
        _CACHED_VALIDATION_CLAIMS = batch.claims
    return _CACHED_VALIDATION_CLAIMS


def increment_version(version_str: str) -> str:
    """Increment minor version string (e.g. '1.0.0' -> '1.1.0')."""
    parts = version_str.split(".")
    if len(parts) >= 2:
        try:
            major = int(parts[0])
            minor = int(parts[1]) + 1
            patch = int(parts[2]) if len(parts) > 2 else 0
            return f"{major}.{minor}.{patch}"
        except ValueError:
            pass
    return f"{version_str}.1"


# ---------------------------------------------------------------------------
# Independent Rule Validation Gate
# ---------------------------------------------------------------------------

class RuleValidator:
    """
    Multi-stage independent validation gate for candidate rule changes.
    Enforces configuration validity, FP budget protection, and regression invariants.
    """

    def __init__(
        self,
        fp_budget: float = 0.10,
        validation_claims: Optional[Sequence[Claim]] = None,
    ) -> None:
        if not (0.0 <= fp_budget <= 1.0):
            raise ValueError(f"fp_budget must be between 0.0 and 1.0, got {fp_budget}")
        self.fp_budget = fp_budget
        self.validation_claims = tuple(validation_claims) if validation_claims is not None else get_default_validation_dataset()

    def validate(
        self,
        active_rulebook: Rulebook,
        proposal: RuleProposal,
    ) -> Tuple[bool, RuleValidationResult, Optional[Rulebook]]:
        """
        Evaluate a candidate RuleProposal across Gates A, B, C, and D.
        
        Returns:
            Tuple of (accepted: bool, validation_result: RuleValidationResult, new_rulebook_or_None).
        """
        if not isinstance(active_rulebook, Rulebook):
            raise TypeError(f"Expected Rulebook instance, got {type(active_rulebook).__name__}")
        if not isinstance(proposal, RuleProposal):
            raise TypeError(f"Expected RuleProposal instance, got {type(proposal).__name__}")

        # -------------------------------------------------------------------
        # Gate A — Configuration Validity
        # -------------------------------------------------------------------
        constraint_key = (proposal.rule_id, proposal.parameter_name)
        if constraint_key not in SUPPORTED_CONSTRAINTS:
            return (
                False,
                RuleValidationResult(
                    passed=False,
                    gate_failed="GATE_A_CONFIG",
                    rejection_reason=(
                        f"Unsupported parameter or rule ID: ('{proposal.rule_id}', '{proposal.parameter_name}'). "
                        f"Only predefined numerical parameters may be adapted."
                    ),
                    fp_budget=self.fp_budget,
                ),
                None,
            )

        constraint = SUPPORTED_CONSTRAINTS[constraint_key]

        # Verify active rule exists
        try:
            active_rule = active_rulebook.get_rule(proposal.rule_id)
        except KeyError:
            return (
                False,
                RuleValidationResult(
                    passed=False,
                    gate_failed="GATE_A_CONFIG",
                    rejection_reason=f"Rule '{proposal.rule_id}' does not exist in Rulebook v{active_rulebook.version}.",
                    fp_budget=self.fp_budget,
                ),
                None,
            )

        # Check parameter type
        if not isinstance(proposal.candidate_value, (int, float)):
            return (
                False,
                RuleValidationResult(
                    passed=False,
                    gate_failed="GATE_A_CONFIG",
                    rejection_reason=(
                        f"Candidate value must be numeric ({constraint.param_type.__name__}), "
                        f"got {type(proposal.candidate_value).__name__}."
                    ),
                    fp_budget=self.fp_budget,
                ),
                None,
            )

        # Check bounds
        cand_val = float(proposal.candidate_value)
        if not (constraint.min_value <= cand_val <= constraint.max_value):
            return (
                False,
                RuleValidationResult(
                    passed=False,
                    gate_failed="GATE_A_CONFIG",
                    rejection_reason=(
                        f"Candidate value {cand_val} is outside allowed bounds "
                        f"[{constraint.min_value}, {constraint.max_value}] for '{proposal.parameter_name}'."
                    ),
                    fp_budget=self.fp_budget,
                ),
                None,
            )

        # Check step size
        current_val = float(active_rule.parameters.get(proposal.parameter_name, proposal.old_value))
        step_taken = abs(cand_val - current_val)
        if step_taken > (constraint.max_step_size + 1e-9):
            return (
                False,
                RuleValidationResult(
                    passed=False,
                    gate_failed="GATE_A_CONFIG",
                    rejection_reason=(
                        f"Proposed step size {step_taken:.3f} exceeds maximum permitted step "
                        f"{constraint.max_step_size:.3f} for '{proposal.parameter_name}'."
                    ),
                    fp_budget=self.fp_budget,
                ),
                None,
            )

        # Construct candidate rulebook safely without mutating active_rulebook
        candidate_rulebook = active_rulebook.clone()
        cand_rule = candidate_rulebook.get_rule(proposal.rule_id)

        # Format parameter value according to its constraint type (int vs float)
        final_cand_val = int(cand_val) if constraint.param_type is int else round(cand_val, 4)
        cand_rule.parameters[proposal.parameter_name] = final_cand_val

        # -------------------------------------------------------------------
        # Gate B — False-Positive Budget
        # -------------------------------------------------------------------
        flagged_count = 0
        for claim in self.validation_claims:
            decision = candidate_rulebook.evaluate_claim(claim)
            if decision.disposition in (Disposition.REFER, Disposition.SUSPECTED_FRAUD):
                flagged_count += 1

        total_val_claims = len(self.validation_claims)
        fp_rate = round(flagged_count / total_val_claims, 4) if total_val_claims > 0 else 0.0

        if fp_rate > self.fp_budget:
            return (
                False,
                RuleValidationResult(
                    passed=False,
                    gate_failed="GATE_B_FP_BUDGET",
                    rejection_reason=(
                        f"Candidate rulebook false-positive rate ({fp_rate:.1%}, {flagged_count}/{total_val_claims}) "
                        f"exceeds allowable budget ({self.fp_budget:.1%})."
                    ),
                    validation_fp_rate=fp_rate,
                    fp_budget=self.fp_budget,
                    details={"flagged_count": flagged_count, "total_validation_claims": total_val_claims},
                ),
                None,
            )

        # -------------------------------------------------------------------
        # Gate C — Golden-Set Regression
        # -------------------------------------------------------------------
        golden_suite = build_golden_regression_set()
        for g_claim, expected_outcome, required_rule in golden_suite:
            decision = candidate_rulebook.evaluate_claim(g_claim)

            # Check clean claim passes
            if expected_outcome == "APPROVE" and decision.disposition != Disposition.APPROVE:
                return (
                    False,
                    RuleValidationResult(
                        passed=False,
                        gate_failed="GATE_C_REGRESSION",
                        rejection_reason=(
                            f"Golden clean claim '{g_claim.claim_id}' was flagged with disposition "
                            f"{decision.disposition.value} (risk score {decision.risk_score})."
                        ),
                        validation_fp_rate=fp_rate,
                        fp_budget=self.fp_budget,
                        golden_set_passed=False,
                    ),
                    None,
                )

            # Check benchmark fraud triggers expected rule
            if expected_outcome == "FLAGGED":
                if decision.disposition not in (Disposition.REFER, Disposition.SUSPECTED_FRAUD):
                    return (
                        False,
                        RuleValidationResult(
                            passed=False,
                            gate_failed="GATE_C_REGRESSION",
                            rejection_reason=(
                                f"Golden fraud claim '{g_claim.claim_id}' was not flagged "
                                f"(received {decision.disposition.value}, expected investigation flag)."
                            ),
                            validation_fp_rate=fp_rate,
                            fp_budget=self.fp_budget,
                            golden_set_passed=False,
                        ),
                        None,
                    )

                # Check required rule was triggered in tool_trace
                if required_rule != "NONE":
                    triggered_rules = {
                        t["rule_id"] for t in decision.tool_trace if t.get("triggered", False)
                    }
                    if required_rule not in triggered_rules:
                        return (
                            False,
                            RuleValidationResult(
                                passed=False,
                                gate_failed="GATE_C_REGRESSION",
                                rejection_reason=(
                                    f"Golden fraud claim '{g_claim.claim_id}' failed to trigger "
                                    f"mandated rule '{required_rule}'."
                                ),
                                validation_fp_rate=fp_rate,
                                fp_budget=self.fp_budget,
                                golden_set_passed=False,
                            ),
                            None,
                        )

        # -------------------------------------------------------------------
        # Gate D — Bounded Activation
        # -------------------------------------------------------------------
        new_version = increment_version(active_rulebook.version)
        candidate_rulebook.version = new_version
        candidate_rulebook.history.append({
            "version": new_version,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "proposal_id": proposal.proposal_id,
            "rule_id": proposal.rule_id,
            "parameter_name": proposal.parameter_name,
            "old_value": proposal.old_value,
            "new_value": final_cand_val,
            "validation_fp_rate": fp_rate,
            "rationale": proposal.rationale,
            "status": "ACCEPTED",
        })

        return (
            True,
            RuleValidationResult(
                passed=True,
                gate_failed=None,
                rejection_reason="",
                validation_fp_rate=fp_rate,
                fp_budget=self.fp_budget,
                golden_set_passed=True,
                details={"version": new_version, "fp_rate": fp_rate},
            ),
            candidate_rulebook,
        )
