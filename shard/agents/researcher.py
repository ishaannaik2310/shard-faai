"""
Shard — Research Agent.

Gathers, structures, and validates observable synthetic evidence for each task
in an InvestigationPlan, returning a typed EvidenceBundle with clear provenance.

CRITICAL INVARIANTS:
1. Ground-Truth Isolation (TI-01): The Research Agent MUST NEVER import GroundTruthVault,
   request hidden labels, or use evaluation-only fraud metadata.
2. Determinism (TI-07): Identical claims and plans produce identical evidence bundles.
3. Provenance: Every evidence item records its synthetic data source and availability.
4. No Network / No Live APIs: All evidence is generated from local observable models.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

from shard.core.models import Claim, EvidenceBundle, EvidenceItem, InvestigationPlan


class ResearchAgent:
    """
    Deterministic research agent that extracts and cross-references observable evidence
    for each verification check mandated by the InvestigationPlan.
    """

    def gather_evidence(self, claim: Claim, plan: InvestigationPlan) -> EvidenceBundle:
        """
        Assemble a typed EvidenceBundle for the given claim and plan.
        
        Validates that claim and plan identifiers match.
        """
        if not isinstance(claim, Claim):
            raise TypeError(f"Expected Claim instance, got {type(claim).__name__}")
        if not isinstance(plan, InvestigationPlan):
            raise TypeError(f"Expected InvestigationPlan instance, got {type(plan).__name__}")
        if claim.claim_id != plan.claim_id:
            raise ValueError(
                f"Identifier mismatch: Claim ID '{claim.claim_id}' != Plan Claim ID '{plan.claim_id}'"
            )

        items: Dict[str, EvidenceItem] = {}

        # 1. Cost Benchmark Evidence (derived from observable claim fields)
        bench_available = claim.benchmark_repair_cost > 0.0
        ratio = (
            round(claim.claimed_amount / claim.benchmark_repair_cost, 4)
            if bench_available
            else None
        )
        items["repair_cost_benchmark"] = EvidenceItem(
            key="repair_cost_benchmark",
            value={
                "claimed_amount": claim.claimed_amount,
                "benchmark_repair_cost": claim.benchmark_repair_cost,
                "markup_ratio": ratio,
            },
            source="CLAIM_SUBMITTED_ESTIMATICS_BENCHMARK",
            is_available=bench_available,
            notes=(
                "Observable repair benchmark cost and claimed total extracted from claim record."
                if bench_available
                else "Benchmark estimate unavailable or non-positive in claim record."
            ),
        )

        # 2. Repair Line Items Evidence (derived from claim submission)
        items_count = len(claim.repair_line_items)
        items_sum = round(sum(i.cost for i in claim.repair_line_items), 2)
        has_items = items_count > 0 or claim.claimed_amount == 0.0
        items["repair_line_items"] = EvidenceItem(
            key="repair_line_items",
            value={
                "item_count": items_count,
                "sum_cost": items_sum,
                "claimed_amount": claim.claimed_amount,
                "items": [
                    {"description": i.part_or_operation, "cost": i.cost, "category": i.category}
                    for i in claim.repair_line_items
                ],
            },
            source="CLAIM_SUBMISSION_GARAGE_QUOTE",
            is_available=has_items,
            notes=(
                "Itemised garage quote extracted directly from claim submission."
                if has_items
                else "No itemised operations submitted with repair claim."
            ),
        )

        # 3. Vehicle Registration Evidence (derived from claim record)
        plate_present = bool(claim.vehicle_registration.strip())
        items["vehicle_registration"] = EvidenceItem(
            key="vehicle_registration",
            value={
                "plate": claim.vehicle_registration,
                "make": claim.vehicle_make,
                "model": claim.vehicle_model,
                "year": claim.vehicle_year,
                "database_status": claim.vehicle_registration_status,
            },
            source="CLAIM_SUBMITTED_VEHICLE_RECORD",
            is_available=plate_present,
            notes=(
                "Vehicle registration details and recorded status from observable claim record."
                if plate_present
                else "Vehicle registration identifier missing from submission."
            ),
        )

        # 4. Vehicle Write-Off Record Evidence (derived from claim record)
        items["vehicle_write_off_status"] = EvidenceItem(
            key="vehicle_write_off_status",
            value={
                "category": claim.vehicle_write_off_status,
                "is_scrap_or_parts_only": claim.vehicle_write_off_status.upper() in ("CAT_A", "CAT_B"),
            },
            source="CLAIM_SUBMITTED_WRITE_OFF_RECORD",
            is_available=True,
            notes="Vehicle write-off category marker extracted from observable claim record.",
        )

        # 5. Prior Damage History Evidence (derived from claim submission)
        items["prior_damage_history"] = EvidenceItem(
            key="prior_damage_history",
            value={
                "self_reported": claim.prior_damage_reported,
                "incident_notes": claim.incident_description,
            },
            source="CLAIM_SUBMISSION_PRIOR_DAMAGE",
            is_available=True,
            notes="Self-reported prior damage indicator and incident statement from claim submission.",
        )

        # 6. Policy Tenure & Network Linkage Evidence (derived from claim record)
        tenure_days = -1
        try:
            incept = datetime.date.fromisoformat(claim.policy_inception_date)
            incident = datetime.date.fromisoformat(claim.incident_date)
            tenure_days = (incident - incept).days
        except (ValueError, TypeError):
            pass

        items["policy_and_network_linkages"] = EvidenceItem(
            key="policy_and_network_linkages",
            value={
                "policy_inception_date": claim.policy_inception_date,
                "incident_date": claim.incident_date,
                "filing_date": claim.filing_date,
                "tenure_days": tenure_days,
                "linked_parties": list(claim.linked_parties),
                "repair_provider_id": claim.repair_provider_id,
            },
            source="CLAIM_SUBMITTED_POLICY_AND_NETWORK",
            is_available=tenure_days >= 0,
            notes="Policy inception and co-claimant linkage details extracted from claim record.",
        )

        # Assess bundle completeness
        is_complete = all(it.is_available for it in items.values())

        return EvidenceBundle(
            claim_id=claim.claim_id,
            items=items,
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            is_complete=is_complete,
        )
