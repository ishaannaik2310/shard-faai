"""
Shard — Whitelist of Observable Numeric Features.

Defines a fixed, observable feature catalog computed exclusively from Claim
attributes. These features are used symmetrically for:
1. Discovery feedback calculation (Critic).
2. Feature separation ranking (Rule-Writer).
3. Runtime evaluation of generic threshold rules (Investigator).

CRITICAL INVARIANTS:
1. Ground-Truth Isolation (TI-01): Features use observable claim fields only.
   Ground truth (is_fraud, fraud_typology) is strictly inaccessible.
2. Robust Extraction: Non-finite, missing, or malformed values safely yield None.
3. Reference Date Invariant: Uses a fixed REFERENCE_YEAR constant (2026) for
   reproducibility across evaluation runs.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from shard.core.models import Claim

REFERENCE_YEAR: int = 2026


@dataclass(frozen=True)
class FeatureDefinition:
    """Contract for a whitelisted numeric observable feature."""
    name: str
    description: str
    min_allowed_threshold: float
    max_allowed_threshold: float
    default_direction: str  # ">" or "<"
    extractor_fn: Callable[[Claim], Optional[float]]


def _extract_filing_delay(claim: Claim) -> Optional[float]:
    """Calculate filing delay in days between incident and submission."""
    try:
        inc = datetime.date.fromisoformat(claim.incident_date)
        fil = datetime.date.fromisoformat(claim.filing_date)
        delay = (fil - inc).days
        if delay < 0 or not math.isfinite(delay):
            return None
        return float(delay)
    except (ValueError, TypeError, AttributeError):
        return None


def _extract_policy_tenure(claim: Claim) -> Optional[float]:
    """Calculate policy tenure in days from inception to incident."""
    try:
        inc = datetime.date.fromisoformat(claim.incident_date)
        incept = datetime.date.fromisoformat(claim.policy_inception_date)
        tenure = (inc - incept).days
        if tenure < 0 or not math.isfinite(tenure):
            return None
        return float(tenure)
    except (ValueError, TypeError, AttributeError):
        return None


def _extract_vehicle_age(claim: Claim) -> Optional[float]:
    """Calculate vehicle age in years relative to REFERENCE_YEAR."""
    try:
        year = int(claim.vehicle_year)
        age = REFERENCE_YEAR - year
        if age < 0 or not math.isfinite(age):
            return None
        return float(age)
    except (ValueError, TypeError):
        return None


def _extract_markup_ratio(claim: Claim) -> Optional[float]:
    """Calculate ratio of claimed repair quote to independent benchmark."""
    try:
        claimed = float(claim.claimed_amount)
        benchmark = float(claim.benchmark_repair_cost)
        if benchmark <= 0.0 or not math.isfinite(claimed) or not math.isfinite(benchmark):
            return None
        ratio = round(claimed / benchmark, 4)
        if ratio < 0.0 or not math.isfinite(ratio):
            return None
        return ratio
    except (ValueError, TypeError):
        return None


def _extract_line_item_count(claim: Claim) -> Optional[float]:
    """Count number of itemised repair components/operations."""
    try:
        items = claim.repair_line_items
        count = len(items) if items is not None else 0
        return float(count)
    except (TypeError, AttributeError):
        return None


def _extract_linked_parties_count(claim: Claim) -> Optional[float]:
    """Count number of linked co-claimants or third parties."""
    try:
        parties = claim.linked_parties
        count = len(parties) if parties is not None else 0
        return float(count)
    except (TypeError, AttributeError):
        return None


FEATURE_WHITELIST: Dict[str, FeatureDefinition] = {
    "claim_filing_delay_days": FeatureDefinition(
        name="claim_filing_delay_days",
        description="Elapsed days between incident occurrence and claim filing submission.",
        min_allowed_threshold=2.0,
        max_allowed_threshold=30.0,
        default_direction=">",
        extractor_fn=_extract_filing_delay,
    ),
    "policy_tenure_days": FeatureDefinition(
        name="policy_tenure_days",
        description="Elapsed days between policy inception and incident occurrence.",
        min_allowed_threshold=1.0,
        max_allowed_threshold=730.0,
        default_direction="<",
        extractor_fn=_extract_policy_tenure,
    ),
    "vehicle_age_years": FeatureDefinition(
        name="vehicle_age_years",
        description="Age of vehicle in years relative to fixed reference year 2026.",
        min_allowed_threshold=0.0,
        max_allowed_threshold=30.0,
        default_direction=">",
        extractor_fn=_extract_vehicle_age,
    ),
    "markup_ratio": FeatureDefinition(
        name="markup_ratio",
        description="Ratio of claimed repair amount to independent estimatics benchmark.",
        min_allowed_threshold=1.0,
        max_allowed_threshold=5.0,
        default_direction=">",
        extractor_fn=_extract_markup_ratio,
    ),
    "repair_line_item_count": FeatureDefinition(
        name="repair_line_item_count",
        description="Total itemised operations or parts on garage repair invoice.",
        min_allowed_threshold=1.0,
        max_allowed_threshold=20.0,
        default_direction=">",
        extractor_fn=_extract_line_item_count,
    ),
    "linked_parties_count": FeatureDefinition(
        name="linked_parties_count",
        description="Number of linked third parties or co-claimants associated with incident.",
        min_allowed_threshold=1.0,
        max_allowed_threshold=10.0,
        default_direction=">",
        extractor_fn=_extract_linked_parties_count,
    ),
}

WHITELISTED_FEATURE_NAMES: Tuple[str, ...] = tuple(sorted(FEATURE_WHITELIST.keys()))


def extract_feature(claim: Claim, feature_name: str) -> Optional[float]:
    """
    Extract a single whitelisted numeric feature from a Claim.
    Returns None if feature is unknown or value cannot be safely computed.
    """
    fdef = FEATURE_WHITELIST.get(feature_name)
    if fdef is None:
        return None
    return fdef.extractor_fn(claim)


def extract_all_features(claim: Claim) -> Dict[str, Optional[float]]:
    """Extract all whitelisted features for a Claim."""
    return {name: fdef.extractor_fn(claim) for name, fdef in FEATURE_WHITELIST.items()}
