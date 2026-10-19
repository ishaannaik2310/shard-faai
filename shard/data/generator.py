"""
Shard — Reproducible Synthetic Claim Generator.

Generates reproducible batches of observable motor insurance claims paired with
isolated ground-truth records stored in a GroundTruthVault.

CRITICAL INVARIANTS:
1. Ground-truth labels (is_fraud, fraud_typology) NEVER leak into Claim instances.
2. Identical inputs (seed, claim count, fraud share, weights) produce identical outputs.
3. Random state uses an isolated random.Random instance; no global state is mutated.
4. Generates both genuine and fraudulent examples when requested parameters allow.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Sequence, Tuple

from shard.core.models import (
    Claim,
    FraudTypology,
    GroundTruthRecord,
    IncidentType,
    RepairLineItem,
    VehicleRegistrationStatus,
    VehicleWriteOffStatus,
)
from shard.data.vault import GroundTruthVault

DEFAULT_TYPOLOGY_WEIGHTS: Dict[str, float] = {
    FraudTypology.EXAGGERATED_REPAIR.value: 0.25,
    FraudTypology.STAGED_COLLISION.value: 0.25,
    FraudTypology.GHOST_VEHICLE.value: 0.20,
    FraudTypology.PRE_EXISTING_DAMAGE.value: 0.15,
    FraudTypology.OPPORTUNISTIC_INFLATION.value: 0.15,
}

DEFAULT_V3_TYPOLOGY_WEIGHTS: Dict[str, float] = {
    FraudTypology.EXAGGERATED_REPAIR.value: 0.20,
    FraudTypology.STAGED_COLLISION.value: 0.20,
    FraudTypology.GHOST_VEHICLE.value: 0.15,
    FraudTypology.PRE_EXISTING_DAMAGE.value: 0.15,
    FraudTypology.OPPORTUNISTIC_INFLATION.value: 0.15,
    FraudTypology.DELAYED_FILING.value: 0.15,
}

VEHICLE_CATALOGUE: Sequence[Tuple[str, str]] = (
    ("Ford", "Focus"),
    ("Vauxhall", "Corsa"),
    ("Volkswagen", "Golf"),
    ("BMW", "3 Series"),
    ("Audi", "A4"),
    ("Nissan", "Qashqai"),
    ("Toyota", "Yaris"),
    ("Mercedes-Benz", "A-Class"),
    ("Hyundai", "Tucson"),
    ("Kia", "Sportage"),
)

STANDARD_PROVIDERS: Sequence[str] = (
    "PRV-NORM-101",
    "PRV-NORM-102",
    "PRV-NORM-103",
    "PRV-NORM-104",
    "PRV-NORM-105",
)

FLAGGED_PROVIDERS: Sequence[str] = (
    "PRV-FLAG-901",
    "PRV-FLAG-902",
    "PRV-FLAG-903",
)

UK_AREA_CODES: Sequence[str] = (
    "AB", "WM", "LF", "RJ", "CV", "GF", "YN", "KM", "PL", "NX"
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration contract for synthetic claim generation."""
    seed: int = 42
    n_claims: int = 30
    fraud_share: float = 0.3
    typology_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TYPOLOGY_WEIGHTS))
    include_v3_typologies: bool = False
    v3_typology_start_round: int = 2
    round_number: int = 1

    def __post_init__(self) -> None:
        if self.n_claims < 0:
            raise ValueError(f"n_claims must be non-negative, got {self.n_claims}")
        if not (0.0 <= self.fraud_share <= 1.0):
            raise ValueError(f"fraud_share must be between 0.0 and 1.0, got {self.fraud_share}")
        if self.v3_typology_start_round < 1:
            raise ValueError(f"v3_typology_start_round must be >= 1, got {self.v3_typology_start_round}")
        if self.round_number < 1:
            raise ValueError(f"round_number must be >= 1, got {self.round_number}")
        if not self.typology_weights:
            raise ValueError("typology_weights dictionary cannot be empty")
        for typology, weight in self.typology_weights.items():
            if weight < 0.0:
                raise ValueError(f"Weight for typology '{typology}' must be non-negative, got {weight}")
        if sum(self.typology_weights.values()) <= 0.0:
            raise ValueError("Sum of typology_weights must be strictly positive")
        if (
            self.include_v3_typologies
            and self.round_number >= self.v3_typology_start_round
            and self.typology_weights == DEFAULT_TYPOLOGY_WEIGHTS
        ):
            object.__setattr__(self, "typology_weights", dict(DEFAULT_V3_TYPOLOGY_WEIGHTS))
        if (
            not self.include_v3_typologies
            and FraudTypology.DELAYED_FILING.value in self.typology_weights
        ):
            raise ValueError(
                "DELAYED_FILING cannot be included in typology_weights when include_v3_typologies is False"
            )


@dataclass(frozen=True)
class GenerationBatch:
    """
    Explicit return contract separating observable claims from evaluation ground truth.
    
    Investigative agents receive only `claims`.
    The `vault` is retained for evaluation components (Critic, Orchestrator).
    """
    claims: Tuple[Claim, ...]
    vault: GroundTruthVault

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def n_records(self) -> int:
        return len(self.vault)


class ClaimGenerator:
    """Deterministic, seeded synthetic motor insurance claim generator."""

    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        self.config = config or GeneratorConfig()

    def generate(self, config: Optional[GeneratorConfig] = None) -> GenerationBatch:
        """
        Generate a batch of claims and corresponding ground-truth records.
        Uses an isolated random.Random instance for strict seed reproducibility.
        """
        cfg = config or self.config
        rng = random.Random(cfg.seed)

        if cfg.n_claims == 0:
            return GenerationBatch(claims=(), vault=GroundTruthVault())

        # Determine exact fraud and genuine claim counts
        n_fraud, n_genuine = self._compute_split(cfg.n_claims, cfg.fraud_share)

        # Generate unique claim IDs
        claim_ids: List[str] = []
        seen_ids = set()
        while len(claim_ids) < cfg.n_claims:
            cid = f"CLM-{rng.randint(100000, 999999)}"
            if cid not in seen_ids:
                seen_ids.add(cid)
                claim_ids.append(cid)

        # Normalise typology weights
        active_weights = dict(cfg.typology_weights)
        if not cfg.include_v3_typologies or cfg.round_number < cfg.v3_typology_start_round:
            if FraudTypology.DELAYED_FILING.value in active_weights:
                active_weights.pop(FraudTypology.DELAYED_FILING.value)

        typology_items = list(active_weights.items())
        typologies = [item[0] for item in typology_items]
        weights = [item[1] for item in typology_items]

        # Allocate typologies to fraudulent claims
        fraud_typologies: List[str] = []
        if n_fraud > 0:
            fraud_typologies = rng.choices(typologies, weights=weights, k=n_fraud)

        # Build genuine and fraudulent records
        pairs: List[Tuple[Claim, GroundTruthRecord]] = []

        # 1. Genuine claims
        for i in range(n_genuine):
            cid = claim_ids[i]
            claim, gt = self._build_genuine_claim(
                cid, rng, include_v3_typologies=cfg.include_v3_typologies
            )
            pairs.append((claim, gt))

        # 2. Fraudulent claims
        for j in range(n_fraud):
            cid = claim_ids[n_genuine + j]
            typology = fraud_typologies[j]
            claim, gt = self._build_fraud_claim(cid, typology, rng)
            pairs.append((claim, gt))

        # Deterministically shuffle to avoid having all genuine first
        rng.shuffle(pairs)

        claims = tuple(p[0] for p in pairs)
        vault = GroundTruthVault(p[1] for p in pairs)

        return GenerationBatch(claims=claims, vault=vault)

    @staticmethod
    def _compute_split(n_claims: int, fraud_share: float) -> Tuple[int, int]:
        """
        Compute integer counts for fraud and genuine claims.
        Guarantees mixed classes when batch size and share permit (n_claims >= 2 and 0 < fraud_share < 1).
        """
        if n_claims == 0:
            return 0, 0
        if fraud_share == 0.0:
            return 0, n_claims
        if fraud_share == 1.0:
            return n_claims, 0
        if n_claims == 1:
            n_fraud = 1 if fraud_share >= 0.5 else 0
            return n_fraud, n_claims - n_fraud

        # For n_claims >= 2 and 0 < fraud_share < 1, guarantee at least 1 genuine and 1 fraud
        nominal = int(round(n_claims * fraud_share))
        n_fraud = max(1, min(n_claims - 1, nominal))
        n_genuine = n_claims - n_fraud
        return n_fraud, n_genuine

    def _generate_plate(self, rng: random.Random) -> str:
        area = rng.choice(UK_AREA_CODES)
        year = rng.choice(["17", "18", "19", "20", "21", "22", "23"])
        letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
        return f"{area}{year} {letters}"

    def _generate_dates(
        self,
        rng: random.Random,
        min_policy_age_days: int = 60,
        max_policy_age_days: int = 400,
        filing_delay_days: int = 3,
    ) -> Tuple[str, str, str]:
        """Returns (policy_inception_date, incident_date, filing_date)."""
        base_date = datetime.date(2026, 3, 1)
        offset_days = rng.randint(0, 150)
        incident_dt = base_date + datetime.timedelta(days=offset_days)

        low = min(min_policy_age_days, max_policy_age_days)
        high = max(min_policy_age_days, max_policy_age_days)
        policy_age = rng.randint(low, high)
        inception_dt = incident_dt - datetime.timedelta(days=policy_age)
        filing_dt = incident_dt + datetime.timedelta(days=filing_delay_days)

        return (
            inception_dt.isoformat(),
            incident_dt.isoformat(),
            filing_dt.isoformat(),
        )

    def _build_line_items(
        self,
        total_amount: float,
        rng: random.Random,
        unrelated_item: Optional[str] = None,
    ) -> Tuple[RepairLineItem, ...]:
        """Construct itemised repair operations summing exactly to total_amount."""
        items: List[RepairLineItem] = []
        operations = [
            ("Front bumper cover & reinforcement", "PARTS"),
            ("Panel beating and structural alignment", "LABOUR"),
            ("Oven bake multi-coat paint finish", "PAINT"),
            ("Headlight cluster assembly", "PARTS"),
            ("ADAS radar calibration", "LABOUR"),
        ]
        rng.shuffle(operations)

        n_items = rng.randint(3, 4)
        if unrelated_item:
            n_items = min(n_items, 3)

        # Distribute fractions
        fractions = [rng.uniform(1.0, 3.0) for _ in range(n_items)]
        total_weight = sum(fractions)
        allocated = 0.0

        for i in range(n_items):
            op_name, cat = operations[i]
            if i == n_items - 1 and not unrelated_item:
                cost = round(total_amount - allocated, 2)
            else:
                cost = round(total_amount * (fractions[i] / total_weight), 2)
            allocated += cost
            items.append(RepairLineItem(part_or_operation=op_name, cost=cost, category=cat))

        if unrelated_item:
            rem = round(total_amount - allocated, 2)
            items.append(RepairLineItem(part_or_operation=unrelated_item, cost=rem, category="MISC"))

        return tuple(items)

    def _build_genuine_claim(
        self,
        claim_id: str,
        rng: random.Random,
        include_v3_typologies: bool = False,
    ) -> Tuple[Claim, GroundTruthRecord]:
        """Build an authentic motor insurance claim with standard parameters."""
        make, model = rng.choice(VEHICLE_CATALOGUE)
        vehicle_year = rng.randint(2016, 2023)
        plate = self._generate_plate(rng)

        if include_v3_typologies:
            if rng.random() < 0.05:
                filing_delay = rng.randint(6, 8)
            else:
                filing_delay = rng.randint(1, 5)
        else:
            filing_delay = rng.randint(1, 5)

        incept, incident, filing = self._generate_dates(
            rng,
            min_policy_age_days=90,
            max_policy_age_days=400,
            filing_delay_days=filing_delay,
        )

        benchmark = round(rng.uniform(850.0, 3200.0), 2)
        # Genuine repair quote is typically between 92% and 112% of benchmark
        claimed = round(benchmark * rng.uniform(0.92, 1.12), 2)

        line_items = self._build_line_items(claimed, rng)

        claim = Claim(
            claim_id=claim_id,
            policy_id=f"POL-{rng.randint(100000, 999999)}",
            claimant_id=f"CLMT-{rng.randint(10000, 99999)}",
            policy_inception_date=incept,
            incident_date=incident,
            filing_date=filing,
            incident_type=IncidentType.COLLISION.value,
            incident_description="Standard low-speed rear-end collision at traffic signal.",
            vehicle_registration=plate,
            vehicle_make=make,
            vehicle_model=model,
            vehicle_year=vehicle_year,
            vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
            vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
            prior_damage_reported=False,
            claimed_amount=claimed,
            benchmark_repair_cost=benchmark,
            repair_provider_id=rng.choice(STANDARD_PROVIDERS),
            repair_line_items=line_items,
            linked_parties=(),
        )

        gt = GroundTruthRecord(
            claim_id=claim_id,
            is_fraud=False,
            fraud_typology=None,
            detail="Genuine claim with standard quote and clean vehicle status.",
        )
        return claim, gt

    def _build_fraud_claim(
        self, claim_id: str, typology: str, rng: random.Random
    ) -> Tuple[Claim, GroundTruthRecord]:
        """Build a synthetic claim exhibiting a specific fraud typology."""
        make, model = rng.choice(VEHICLE_CATALOGUE)
        vehicle_year = rng.randint(2015, 2022)
        plate = self._generate_plate(rng)

        if typology == FraudTypology.EXAGGERATED_REPAIR.value:
            incept, incident, filing = self._generate_dates(rng, min_policy_age_days=120)
            benchmark = round(rng.uniform(1000.0, 2500.0), 2)
            # Highly inflated: 2.8x to 3.8x benchmark
            claimed = round(benchmark * rng.uniform(2.8, 3.8), 2)
            line_items = self._build_line_items(claimed, rng)

            claim = Claim(
                claim_id=claim_id,
                policy_id=f"POL-{rng.randint(100000, 999999)}",
                claimant_id=f"CLMT-{rng.randint(10000, 99999)}",
                policy_inception_date=incept,
                incident_date=incident,
                filing_date=filing,
                incident_type=IncidentType.COLLISION.value,
                incident_description="Moderate collision with severe chassis distortion claimed.",
                vehicle_registration=plate,
                vehicle_make=make,
                vehicle_model=model,
                vehicle_year=vehicle_year,
                vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
                vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
                prior_damage_reported=False,
                claimed_amount=claimed,
                benchmark_repair_cost=benchmark,
                repair_provider_id=rng.choice(FLAGGED_PROVIDERS),
                repair_line_items=line_items,
                linked_parties=(),
            )
            gt = GroundTruthRecord(
                claim_id=claim_id,
                is_fraud=True,
                fraud_typology=typology,
                detail=f"Exaggerated repair cost: claimed £{claimed:.2f} vs benchmark £{benchmark:.2f}.",
            )

        elif typology == FraudTypology.STAGED_COLLISION.value:
            # Policy inception occurs suspiciously close to incident (2-6 days)
            incept, incident, filing = self._generate_dates(
                rng, min_policy_age_days=2, max_policy_age_days=6, filing_delay_days=1
            )
            benchmark = round(rng.uniform(1500.0, 3500.0), 2)
            claimed = round(benchmark * rng.uniform(1.2, 1.6), 2)
            line_items = self._build_line_items(claimed, rng)

            third_party_1 = f"CLMT-{rng.randint(10000, 99999)}"
            third_party_2 = f"CLMT-{rng.randint(10000, 99999)}"

            claim = Claim(
                claim_id=claim_id,
                policy_id=f"POL-{rng.randint(100000, 999999)}",
                claimant_id=f"CLMT-{rng.randint(10000, 99999)}",
                policy_inception_date=incept,
                incident_date=incident,
                filing_date=filing,
                incident_type=IncidentType.COLLISION.value,
                incident_description="Abrupt braking at roundabout with multiple colluding third parties.",
                vehicle_registration=plate,
                vehicle_make=make,
                vehicle_model=model,
                vehicle_year=vehicle_year,
                vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
                vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
                prior_damage_reported=False,
                claimed_amount=claimed,
                benchmark_repair_cost=benchmark,
                repair_provider_id=rng.choice(FLAGGED_PROVIDERS),
                repair_line_items=line_items,
                linked_parties=(third_party_1, third_party_2),
            )
            gt = GroundTruthRecord(
                claim_id=claim_id,
                is_fraud=True,
                fraud_typology=typology,
                detail="Staged collision: incident within days of policy inception with linked parties.",
            )

        elif typology == FraudTypology.GHOST_VEHICLE.value:
            incept, incident, filing = self._generate_dates(rng, min_policy_age_days=150)
            benchmark = round(rng.uniform(1800.0, 4200.0), 2)
            claimed = round(benchmark * rng.uniform(1.1, 1.4), 2)
            line_items = self._build_line_items(claimed, rng)

            # Registration or write-off anomaly: vehicle was previously scrapped or unregistered
            is_unregistered = rng.choice([True, False])
            reg_status = (
                VehicleRegistrationStatus.UNREGISTERED.value
                if is_unregistered
                else VehicleRegistrationStatus.VALID.value
            )
            write_off = (
                VehicleWriteOffStatus.CLEAN.value
                if is_unregistered
                else rng.choice([VehicleWriteOffStatus.CAT_A.value, VehicleWriteOffStatus.CAT_B.value])
            )

            claim = Claim(
                claim_id=claim_id,
                policy_id=f"POL-{rng.randint(100000, 999999)}",
                claimant_id=f"CLMT-{rng.randint(10000, 99999)}",
                policy_inception_date=incept,
                incident_date=incident,
                filing_date=filing,
                incident_type=IncidentType.COLLISION.value,
                incident_description="Accident reported on vehicle with anomalous registration or write-off status.",
                vehicle_registration=plate,
                vehicle_make=make,
                vehicle_model=model,
                vehicle_year=vehicle_year,
                vehicle_registration_status=reg_status,
                vehicle_write_off_status=write_off,
                prior_damage_reported=False,
                claimed_amount=claimed,
                benchmark_repair_cost=benchmark,
                repair_provider_id=rng.choice(STANDARD_PROVIDERS),
                repair_line_items=line_items,
                linked_parties=(),
            )
            gt = GroundTruthRecord(
                claim_id=claim_id,
                is_fraud=True,
                fraud_typology=typology,
                detail=f"Ghost vehicle: reg_status={reg_status}, write_off_status={write_off}.",
            )

        elif typology == FraudTypology.PRE_EXISTING_DAMAGE.value:
            incept, incident, filing = self._generate_dates(rng, min_policy_age_days=180)
            benchmark = round(rng.uniform(1200.0, 3000.0), 2)
            claimed = round(benchmark * rng.uniform(1.3, 1.7), 2)
            line_items = self._build_line_items(claimed, rng)

            claim = Claim(
                claim_id=claim_id,
                policy_id=f"POL-{rng.randint(100000, 999999)}",
                claimant_id=f"CLMT-{rng.randint(10000, 99999)}",
                policy_inception_date=incept,
                incident_date=incident,
                filing_date=filing,
                incident_type=IncidentType.COLLISION.value,
                incident_description="Minor parking collision with prior unresolved quarter-panel damage claimed.",
                vehicle_registration=plate,
                vehicle_make=make,
                vehicle_model=model,
                vehicle_year=vehicle_year,
                vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
                vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
                prior_damage_reported=True,  # Observable prior damage record
                claimed_amount=claimed,
                benchmark_repair_cost=benchmark,
                repair_provider_id=rng.choice(STANDARD_PROVIDERS),
                repair_line_items=line_items,
                linked_parties=(),
            )
            gt = GroundTruthRecord(
                claim_id=claim_id,
                is_fraud=True,
                fraud_typology=typology,
                detail="Pre-existing damage falsely bundled into current incident claim.",
            )

        elif typology == FraudTypology.OPPORTUNISTIC_INFLATION.value:
            incept, incident, filing = self._generate_dates(rng, min_policy_age_days=200)
            benchmark = round(rng.uniform(1000.0, 2400.0), 2)
            # Inflation of 1.4x to 1.7x with an unrelated item added
            claimed = round(benchmark * rng.uniform(1.4, 1.7), 2)
            line_items = self._build_line_items(
                claimed, rng, unrelated_item="Full diamond-cut alloy refurbishment (pre-existing)"
            )

            claim = Claim(
                claim_id=claim_id,
                policy_id=f"POL-{rng.randint(100000, 999999)}",
                claimant_id=f"CLMT-{rng.randint(10000, 99999)}",
                policy_inception_date=incept,
                incident_date=incident,
                filing_date=filing,
                incident_type=IncidentType.COLLISION.value,
                incident_description="Genuine minor shunt with opportunistic addition of unrelated repairs.",
                vehicle_registration=plate,
                vehicle_make=make,
                vehicle_model=model,
                vehicle_year=vehicle_year,
                vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
                vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
                prior_damage_reported=False,
                claimed_amount=claimed,
                benchmark_repair_cost=benchmark,
                repair_provider_id=rng.choice(STANDARD_PROVIDERS),
                repair_line_items=line_items,
                linked_parties=(),
            )
            gt = GroundTruthRecord(
                claim_id=claim_id,
                is_fraud=True,
                fraud_typology=typology,
                detail="Opportunistic claim padding: unrelated alloy wheel refurbishment added to quote.",
            )

        elif typology == FraudTypology.DELAYED_FILING.value:
            incept, incident, filing = self._generate_dates(
                rng,
                min_policy_age_days=90,
                max_policy_age_days=400,
                filing_delay_days=rng.randint(4, 10),
            )
            benchmark = round(rng.uniform(850.0, 3200.0), 2)
            claimed = round(benchmark * rng.uniform(0.92, 1.12), 2)
            line_items = self._build_line_items(claimed, rng)

            claim = Claim(
                claim_id=claim_id,
                policy_id=f"POL-{rng.randint(100000, 999999)}",
                claimant_id=f"CLMT-{rng.randint(10000, 99999)}",
                policy_inception_date=incept,
                incident_date=incident,
                filing_date=filing,
                incident_type=IncidentType.COLLISION.value,
                incident_description="Standard low-speed rear-end collision at traffic signal.",
                vehicle_registration=plate,
                vehicle_make=make,
                vehicle_model=model,
                vehicle_year=vehicle_year,
                vehicle_registration_status=VehicleRegistrationStatus.VALID.value,
                vehicle_write_off_status=VehicleWriteOffStatus.CLEAN.value,
                prior_damage_reported=False,
                claimed_amount=claimed,
                benchmark_repair_cost=benchmark,
                repair_provider_id=rng.choice(STANDARD_PROVIDERS),
                repair_line_items=line_items,
                linked_parties=(),
            )
            gt = GroundTruthRecord(
                claim_id=claim_id,
                is_fraud=True,
                fraud_typology=typology,
                detail="Delayed filing fraud: claim submission delayed past normal notification window.",
            )

        else:
            raise ValueError(f"Unrecognised fraud typology: '{typology}'")

        return claim, gt


def generate_claims(config: Optional[GeneratorConfig] = None) -> GenerationBatch:
    """Convenience function to generate a batch of claims with default or custom config."""
    generator = ClaimGenerator(config)
    return generator.generate()
