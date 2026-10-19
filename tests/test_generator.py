"""
Tests for Shard Synthetic Claim Generator, Data Contracts and Ground-Truth Vault.

Covers:
1. Output sizing and contracts.
2. Uniqueness of claim identifiers within batches.
3. 1-to-1 correspondence between claims and vault records.
4. Ground-truth label isolation from Claim models and serialization (TI-01).
5. Seed reproducibility.
6. Seed divergence across different configurations.
7. Validity of dates, ordering, and monetary amounts.
8. Small batches, boundaries (0, 1, fraud_share 0.0, 1.0), and invalid parameters.
9. Presence of both genuine and fraudulent claims when configured.
10. Vault operations, error handling, and protection against mutation.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from typing import Set

import pytest

from shard.core.models import (
    Claim,
    FraudTypology,
    GroundTruthRecord,
    RepairLineItem,
)
from shard.data.generator import (
    ClaimGenerator,
    DEFAULT_TYPOLOGY_WEIGHTS,
    GenerationBatch,
    GeneratorConfig,
    generate_claims,
)
from shard.data.vault import GroundTruthVault


# ---------------------------------------------------------------------------
# 1. Output Size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested_count", [5, 10, 30])
def test_output_size(requested_count: int) -> None:
    """Requesting N claims produces exactly N observable claims and N vault records."""
    cfg = GeneratorConfig(seed=123, n_claims=requested_count, fraud_share=0.3)
    batch = generate_claims(cfg)

    assert isinstance(batch, GenerationBatch)
    assert len(batch.claims) == requested_count
    assert batch.n_claims == requested_count
    assert len(batch.vault) == requested_count
    assert batch.n_records == requested_count


# ---------------------------------------------------------------------------
# 2. Unique Claim Identifiers
# ---------------------------------------------------------------------------

def test_unique_claim_identifiers() -> None:
    """Generated claim identifiers must be unique within every batch."""
    cfg = GeneratorConfig(seed=42, n_claims=60, fraud_share=0.4)
    batch = generate_claims(cfg)

    claim_ids = [c.claim_id for c in batch.claims]
    assert len(claim_ids) == 60
    assert len(set(claim_ids)) == 60, "Duplicate claim IDs detected within batch"


# ---------------------------------------------------------------------------
# 3. One-to-One Correspondence
# ---------------------------------------------------------------------------

def test_one_to_one_correspondence() -> None:
    """Every observable claim must have exactly one corresponding ground-truth record in the vault."""
    cfg = GeneratorConfig(seed=77, n_claims=25, fraud_share=0.3)
    batch = generate_claims(cfg)

    vault_ids = set(batch.vault.claim_ids())
    claim_ids = {c.claim_id for c in batch.claims}

    # Set equality ensures bijection
    assert claim_ids == vault_ids
    assert len(vault_ids) == 25

    # Each claim ID can be individually resolved in the vault
    for claim in batch.claims:
        label = batch.vault.get_label(claim.claim_id)
        assert label.claim_id == claim.claim_id
        assert isinstance(label.is_fraud, bool)


# ---------------------------------------------------------------------------
# 4. Ground-Truth Isolation (TI-01, TI-11)
# ---------------------------------------------------------------------------

def test_no_hidden_labels_in_claim_model() -> None:
    """
    CRITICAL INVARIANT:
    Claim dataclass and its serialized forms must never contain ground-truth labels.
    """
    cfg = GeneratorConfig(seed=999, n_claims=15, fraud_share=0.5)
    batch = generate_claims(cfg)

    prohibited_substrings = ("fraud", "typology", "ground_truth", "is_fraud", "is_fake")

    # 1. Inspect Claim dataclass field definitions
    claim_field_names = {f.name for f in dataclasses.fields(Claim)}
    for field_name in claim_field_names:
        for sub in prohibited_substrings:
            assert sub not in field_name.lower(), (
                f"Prohibited label substring '{sub}' leaked into Claim field '{field_name}'"
            )

    # 2. Inspect RepairLineItem field definitions
    item_field_names = {f.name for f in dataclasses.fields(RepairLineItem)}
    for field_name in item_field_names:
        for sub in prohibited_substrings:
            assert sub not in field_name.lower(), (
                f"Prohibited label substring '{sub}' leaked into RepairLineItem field '{field_name}'"
            )

    # 3. Inspect serialized dictionaries and JSON strings of every generated claim
    for claim in batch.claims:
        claim_dict = dataclasses.asdict(claim)

        # Top-level keys check
        for k in claim_dict.keys():
            for sub in prohibited_substrings:
                assert sub not in k.lower(), f"Key '{k}' contains prohibited term '{sub}'"

        # JSON serialization check
        serialized_json = json.dumps(claim_dict)
        for sub in ("is_fraud", "fraud_typology", "GroundTruth"):
            assert sub not in serialized_json, f"Serialized claim contains forbidden substring '{sub}'"


# ---------------------------------------------------------------------------
# 5. Seed Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility_identical_seeds() -> None:
    """Two generation runs with identical seeds must produce bit-identical claims and vault records."""
    cfg1 = GeneratorConfig(seed=4242, n_claims=20, fraud_share=0.35)
    cfg2 = GeneratorConfig(seed=4242, n_claims=20, fraud_share=0.35)

    batch1 = generate_claims(cfg1)
    batch2 = generate_claims(cfg2)

    assert len(batch1.claims) == len(batch2.claims)

    for c1, c2 in zip(batch1.claims, batch2.claims):
        assert c1 == c2, f"Claim mismatch: {c1.claim_id} != {c2.claim_id}"

    for c1 in batch1.claims:
        lbl1 = batch1.vault.get_label(c1.claim_id)
        lbl2 = batch2.vault.get_label(c1.claim_id)
        assert lbl1 == lbl2


# ---------------------------------------------------------------------------
# 6. Seed Divergence
# ---------------------------------------------------------------------------

def test_different_batches_different_seeds() -> None:
    """Different seeds must produce different batches."""
    cfg1 = GeneratorConfig(seed=101, n_claims=20, fraud_share=0.3)
    cfg2 = GeneratorConfig(seed=202, n_claims=20, fraud_share=0.3)

    batch1 = generate_claims(cfg1)
    batch2 = generate_claims(cfg2)

    ids1 = [c.claim_id for c in batch1.claims]
    ids2 = [c.claim_id for c in batch2.claims]

    assert ids1 != ids2
    # Detailed claim contents differ
    amounts1 = [c.claimed_amount for c in batch1.claims]
    amounts2 = [c.claimed_amount for c in batch2.claims]
    assert amounts1 != amounts2


# ---------------------------------------------------------------------------
# 7. Valid Dates and Monetary Values
# ---------------------------------------------------------------------------

def test_valid_dates_and_non_negative_amounts() -> None:
    """Claims must satisfy chronological order and non-negative currency amounts."""
    cfg = GeneratorConfig(seed=888, n_claims=30, fraud_share=0.4)
    batch = generate_claims(cfg)

    for c in batch.claims:
        # Currency amounts
        assert c.claimed_amount >= 0.0, f"Negative claimed amount on {c.claim_id}"
        assert c.benchmark_repair_cost >= 0.0, f"Negative benchmark cost on {c.claim_id}"

        for item in c.repair_line_items:
            assert item.cost >= 0.0, f"Negative line item cost on {c.claim_id}"
            assert isinstance(item.part_or_operation, str) and item.part_or_operation

        # Dates validity and ordering
        inception = datetime.date.fromisoformat(c.policy_inception_date)
        incident = datetime.date.fromisoformat(c.incident_date)
        filing = datetime.date.fromisoformat(c.filing_date)

        assert inception <= incident, (
            f"Policy inception {inception} occurs after incident {incident} on {c.claim_id}"
        )
        assert incident <= filing, (
            f"Incident date {incident} occurs after filing date {filing} on {c.claim_id}"
        )


# ---------------------------------------------------------------------------
# 8. Boundary Conditions and Invalid Parameters
# ---------------------------------------------------------------------------

def test_boundary_zero_claims() -> None:
    """Zero requested claims returns empty batch and empty vault without error."""
    cfg = GeneratorConfig(seed=1, n_claims=0, fraud_share=0.5)
    batch = generate_claims(cfg)

    assert batch.n_claims == 0
    assert batch.n_records == 0
    assert len(batch.claims) == 0
    assert len(batch.vault) == 0


def test_boundary_single_claim() -> None:
    """Single claim batch generates exactly 1 claim and 1 vault record."""
    cfg_genuine = GeneratorConfig(seed=2, n_claims=1, fraud_share=0.2)
    batch_g = generate_claims(cfg_genuine)
    assert batch_g.n_claims == 1
    assert batch_g.n_records == 1
    assert batch_g.vault.get_label(batch_g.claims[0].claim_id).is_fraud is False

    cfg_fraud = GeneratorConfig(seed=3, n_claims=1, fraud_share=0.8)
    batch_f = generate_claims(cfg_fraud)
    assert batch_f.n_claims == 1
    assert batch_f.n_records == 1
    assert batch_f.vault.get_label(batch_f.claims[0].claim_id).is_fraud is True


def test_boundary_fraud_share_zero_and_one() -> None:
    """fraud_share=0.0 must yield 100% genuine; fraud_share=1.0 must yield 100% fraud."""
    batch_all_genuine = generate_claims(GeneratorConfig(seed=10, n_claims=15, fraud_share=0.0))
    for c in batch_all_genuine.claims:
        label = batch_all_genuine.vault.get_label(c.claim_id)
        assert label.is_fraud is False
        assert label.fraud_typology is None

    batch_all_fraud = generate_claims(GeneratorConfig(seed=20, n_claims=15, fraud_share=1.0))
    for c in batch_all_fraud.claims:
        label = batch_all_fraud.vault.get_label(c.claim_id)
        assert label.is_fraud is True
        assert label.fraud_typology is not None


def test_invalid_parameters_raise_value_error() -> None:
    """Invalid configurations must raise ValueError on instantiation."""
    with pytest.raises(ValueError, match="n_claims must be non-negative"):
        GeneratorConfig(n_claims=-5)

    with pytest.raises(ValueError, match="fraud_share must be between 0.0 and 1.0"):
        GeneratorConfig(fraud_share=-0.1)

    with pytest.raises(ValueError, match="fraud_share must be between 0.0 and 1.0"):
        GeneratorConfig(fraud_share=1.5)

    with pytest.raises(ValueError, match="typology_weights dictionary cannot be empty"):
        GeneratorConfig(typology_weights={})

    with pytest.raises(ValueError, match="must be non-negative"):
        GeneratorConfig(typology_weights={"EXAGGERATED_REPAIR": -1.0})

    with pytest.raises(ValueError, match="Sum of typology_weights must be strictly positive"):
        GeneratorConfig(typology_weights={"EXAGGERATED_REPAIR": 0.0})


# ---------------------------------------------------------------------------
# 9. Mixed Genuine and Fraudulent Claim Generation
# ---------------------------------------------------------------------------

def test_mixed_generation_when_permitted() -> None:
    """When n_claims >= 2 and 0 < fraud_share < 1, both genuine and fraudulent examples must be produced."""
    cfg = GeneratorConfig(seed=505, n_claims=30, fraud_share=0.3)
    batch = generate_claims(cfg)

    fraud_labels = [batch.vault.get_label(c.claim_id) for c in batch.claims]
    n_fraud = sum(1 for l in fraud_labels if l.is_fraud)
    n_genuine = sum(1 for l in fraud_labels if not l.is_fraud)

    assert n_fraud > 0, "No fraudulent claims generated despite fraud_share=0.3"
    assert n_genuine > 0, "No genuine claims generated despite fraud_share=0.3"
    assert n_fraud + n_genuine == 30

    # Closer check: nominal count is round(30 * 0.3) = 9
    assert n_fraud == 9
    assert n_genuine == 21


def test_all_five_typologies_generated() -> None:
    """A larger run with default weights generates all 5 agreed motor insurance fraud typologies."""
    cfg = GeneratorConfig(seed=9999, n_claims=80, fraud_share=0.5)
    batch = generate_claims(cfg)

    typologies_seen: Set[str] = set()
    for c in batch.claims:
        label = batch.vault.get_label(c.claim_id)
        if label.is_fraud and label.fraud_typology:
            typologies_seen.add(label.fraud_typology)

    expected = {
        FraudTypology.EXAGGERATED_REPAIR.value,
        FraudTypology.STAGED_COLLISION.value,
        FraudTypology.GHOST_VEHICLE.value,
        FraudTypology.PRE_EXISTING_DAMAGE.value,
        FraudTypology.OPPORTUNISTIC_INFLATION.value,
    }
    assert typologies_seen == expected, f"Missing typologies: {expected - typologies_seen}"


# ---------------------------------------------------------------------------
# 10. Vault Operations and Mutation Protection
# ---------------------------------------------------------------------------

def test_vault_lookups_and_immutability() -> None:
    """Vault provides lookup validation, KeyError for missing records, and protects internal state."""
    rec1 = GroundTruthRecord(claim_id="CLM-1001", is_fraud=False, detail="Clean")
    rec2 = GroundTruthRecord(claim_id="CLM-1002", is_fraud=True, fraud_typology="GHOST_VEHICLE")

    vault = GroundTruthVault([rec1, rec2])

    assert len(vault) == 2
    assert "CLM-1001" in vault
    assert "CLM-9999" not in vault
    assert vault.contains("CLM-1002") is True

    # Single lookup
    assert vault.get_label("CLM-1001") == rec1

    # Bulk lookup
    bulk = vault.get_labels(["CLM-1001", "CLM-1002"])
    assert bulk == {"CLM-1001": rec1, "CLM-1002": rec2}

    # Unknown ID error handling
    with pytest.raises(KeyError, match="No ground-truth record found for claim ID: 'CLM-UNKNOWN'"):
        vault.get_label("CLM-UNKNOWN")

    with pytest.raises(KeyError, match="Vault missing ground-truth records"):
        vault.get_labels(["CLM-1001", "CLM-UNKNOWN"])

    # Type error handling
    with pytest.raises(TypeError):
        vault.get_label(123)  # type: ignore

    with pytest.raises(TypeError):
        vault.get_labels("not-a-sequence")  # type: ignore

    # Duplicate registration error
    with pytest.raises(ValueError, match="Ground-truth record already registered"):
        vault.register(rec1)

    # Immutability check 1: records returned are frozen dataclasses
    label = vault.get_label("CLM-1001")
    with pytest.raises(dataclasses.FrozenInstanceError):
        label.is_fraud = True  # type: ignore

    # Immutability check 2: modifying returned dict from get_labels does not mutate internal vault
    returned_dict = vault.get_labels(["CLM-1001"])
    returned_dict["CLM-1001"] = GroundTruthRecord(claim_id="CLM-1001", is_fraud=True)
    assert vault.get_label("CLM-1001").is_fraud is False, "Vault internal state was corrupted by caller mutation!"


def test_core_package_does_not_import_vault() -> None:
    """
    Least privilege verification:
    shard.core must never import from shard.data.vault.
    Note: Agent packages (Planner, Research, Investigator) do not yet exist;
    their import isolation checks belong in Phase 4 and Phase 5.
    """
    import shard.core.models as core_models
    import sys

    # Inspect module globals
    for key, val in vars(core_models).items():
        assert "vault" not in key.lower()
        if hasattr(val, "__module__"):
            assert val.__module__ != "shard.data.vault"


def test_claim_immutability() -> None:
    """Claim and RepairLineItem instances must be immutable (frozen dataclasses)."""
    item = RepairLineItem(part_or_operation="Bumper", cost=150.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.cost = 200.0  # type: ignore

    cfg = GeneratorConfig(seed=12, n_claims=1)
    claim = generate_claims(cfg).claims[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.claimed_amount = 9999.0  # type: ignore


def test_custom_typology_weights() -> None:
    """Configuring custom typology weights forces targeted generation of chosen typologies."""
    custom_weights = {
        FraudTypology.EXAGGERATED_REPAIR.value: 0.0,
        FraudTypology.STAGED_COLLISION.value: 1.0,
        FraudTypology.GHOST_VEHICLE.value: 0.0,
        FraudTypology.PRE_EXISTING_DAMAGE.value: 0.0,
        FraudTypology.OPPORTUNISTIC_INFLATION.value: 0.0,
    }
    cfg = GeneratorConfig(seed=55, n_claims=20, fraud_share=0.5, typology_weights=custom_weights)
    batch = generate_claims(cfg)

    fraud_records = [batch.vault.get_label(c.claim_id) for c in batch.claims if batch.vault.get_label(c.claim_id).is_fraud]
    assert len(fraud_records) > 0
    for r in fraud_records:
        assert r.fraud_typology == FraudTypology.STAGED_COLLISION.value


def test_vault_fraud_ratio_and_repr() -> None:
    """Test vault fraud_ratio calculation and string representation."""
    empty_vault = GroundTruthVault()
    assert empty_vault.fraud_ratio() == 0.0
    assert "records=0" in repr(empty_vault)

    rec1 = GroundTruthRecord(claim_id="C1", is_fraud=False)
    rec2 = GroundTruthRecord(claim_id="C2", is_fraud=True, fraud_typology="GHOST_VEHICLE")
    vault = GroundTruthVault([rec1, rec2])

    assert vault.fraud_ratio() == 0.5
    assert "records=2" in repr(vault)
    assert "fraud_ratio=0.50" in repr(vault)
    assert set(vault.claim_ids()) == {"C1", "C2"}

