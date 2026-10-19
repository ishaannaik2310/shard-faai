"""
Shard — Discovery Feedback Models.

Data structures carrying aggregate statistical distributions of observable
claim features across missed fraud claims (false negatives) and genuine
reference claims across rounds.

CRITICAL INVARIANTS (TI-01, TI-06):
1. Data Isolation: Contains ONLY aggregate statistical summaries (counts,
   moments, percentiles).
2. Zero Leakage: NEVER contains individual claim identifiers, ground-truth labels,
   or fraud typology names.
3. Immutability: Frozen dataclasses prevent downstream mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class FeatureAggregateStats:
    """Aggregate distribution statistics for a numeric observable feature."""
    count: int
    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    p97: float
    p99: float
    min: float
    max: float
    tail_share_beyond_ref_p95: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "p5": self.p5,
            "p50": self.p50,
            "p95": self.p95,
            "p97": self.p97,
            "p99": self.p99,
            "min": self.min,
            "max": self.max,
            "tail_share_beyond_ref_p95": self.tail_share_beyond_ref_p95,
        }


@dataclass(frozen=True)
class DiscoveryFeedback:
    """
    Evaluation feedback containing aggregate distributions of whitelisted
    observable features for missed fraud claims vs genuine reference claims.

    Passed across the trust boundary from Critic to Rule-Writer.
    """
    round_number: int
    n_missed: int
    n_reference: int
    # Mapping: feature_name -> (missed_stats, reference_stats)
    features: Dict[str, Tuple[FeatureAggregateStats, FeatureAggregateStats]]
    min_missed_samples: int = 8
    has_sufficient_evidence: bool = True

    def __post_init__(self) -> None:
        if self.round_number < 1:
            raise ValueError(f"round_number must be >= 1, got {self.round_number}")
        if self.n_missed < 0 or self.n_reference < 0:
            raise ValueError("Sample counts must be non-negative")

    def get_feature_stats(
        self, feature_name: str
    ) -> Optional[Tuple[FeatureAggregateStats, FeatureAggregateStats]]:
        """Return (missed_stats, reference_stats) for a named feature."""
        return self.features.get(feature_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "n_missed": self.n_missed,
            "n_reference": self.n_reference,
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "min_missed_samples": self.min_missed_samples,
            "features": {
                name: {
                    "missed": pair[0].to_dict(),
                    "reference": pair[1].to_dict(),
                }
                for name, pair in sorted(self.features.items())
            },
        }
