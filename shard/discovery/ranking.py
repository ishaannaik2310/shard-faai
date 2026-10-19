"""
Shard — Discovery Feature Ranking Engine.

Pure mathematical ranking function that evaluates whitelisted observable
features against DiscoveryFeedback aggregate statistics.

CRITICAL INVARIANTS:
1. Pure Function: Deterministic, side-effect free computation over DiscoveryFeedback.
2. Generic Ranking: Completely feature-agnostic; no domain-specific typologies,
   claim IDs, or feature names are hardcoded.
3. Order-Independence: Sorts input feature keys to ensure identical output regardless
   of dict insertion order.
4. Insufficient Evidence Guard: If n_missed < min_missed_samples, immediately returns
   empty ranking with INSUFFICIENT_EVIDENCE status.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from shard.core.discovery_models import DiscoveryFeedback, FeatureAggregateStats


@dataclass(frozen=True)
class RankedFeature:
    """Evaluation result for a ranked feature candidate."""
    feature_name: str
    separation_score: float
    standardized_mean_diff: float
    tail_separation_share: float
    direction: str  # ">" or "<"
    missed_mean: float
    reference_mean: float
    reference_p95: float
    reference_p5: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "separation_score": round(self.separation_score, 4),
            "standardized_mean_diff": round(self.standardized_mean_diff, 4),
            "tail_separation_share": round(self.tail_separation_share, 4),
            "direction": self.direction,
            "missed_mean": round(self.missed_mean, 4),
            "reference_mean": round(self.reference_mean, 4),
            "reference_p95": round(self.reference_p95, 4),
            "reference_p5": round(self.reference_p5, 4),
        }


def _estimate_tail_share(
    missed: FeatureAggregateStats,
    threshold: float,
    direction: str,
) -> float:
    """
    Estimate proportion of missed claims beyond the reference quantile
    using empirical quantiles if tail_share_beyond_ref_p95 is not precomputed.
    """
    if hasattr(missed, "tail_share_beyond_ref_p95") and missed.tail_share_beyond_ref_p95 > 0.0:
        return float(missed.tail_share_beyond_ref_p95)

    if direction == ">":
        if missed.min > threshold:
            return 1.0
        if missed.p5 > threshold:
            return 0.95
        if missed.p50 > threshold:
            return 0.50
        if missed.p95 > threshold:
            return 0.05
        return 0.0
    else:  # direction == "<"
        if missed.max < threshold:
            return 1.0
        if missed.p95 < threshold:
            return 0.95
        if missed.p50 < threshold:
            return 0.50
        if missed.p5 < threshold:
            return 0.05
        return 0.0


def rank_features(
    feedback: DiscoveryFeedback,
) -> Tuple[List[RankedFeature], str]:
    """
    Rank all whitelisted observable features by statistical separation between
    missed fraud claims and genuine reference claims.

    Separation Score Formula:
    Score = StandardizedMeanDifference + 2.0 * TailSeparationShare

    Returns:
    - (ranked_list, "OK") if evidence is sufficient.
    - ([], "INSUFFICIENT_EVIDENCE") if n_missed < min_missed_samples.
    """
    if not feedback.has_sufficient_evidence or feedback.n_missed < feedback.min_missed_samples:
        return [], "INSUFFICIENT_EVIDENCE"

    if feedback.n_reference < 1:
        return [], "INSUFFICIENT_REFERENCE_SAMPLES"

    ranked: List[RankedFeature] = []

    # Sort feature keys for deterministic, order-independent evaluation
    sorted_feature_names = sorted(feedback.features.keys())

    for feat_name in sorted_feature_names:
        missed_stats, ref_stats = feedback.features[feat_name]

        # Guard against empty distributions
        if missed_stats.count < 1 or ref_stats.count < 1:
            continue

        n_m = missed_stats.count
        n_r = ref_stats.count
        m_mean = missed_stats.mean
        r_mean = ref_stats.mean
        m_var = missed_stats.std ** 2
        r_var = ref_stats.std ** 2

        # Pooled standard deviation
        df = max(1, n_m + n_r - 2)
        pooled_var = ((n_m - 1) * m_var + (n_r - 1) * r_var) / df
        pooled_std = math.sqrt(max(1e-8, pooled_var))

        mean_diff = abs(m_mean - r_mean)
        std_diff = mean_diff / (pooled_std + 1e-6)

        # Determine direction: ">" if missed claims have higher values, "<" if lower
        direction = ">" if m_mean >= r_mean else "<"

        # Tail separation share: proportion of missed claims beyond reference p95 / p5
        target_ref_quantile = ref_stats.p95 if direction == ">" else ref_stats.p5
        tail_share = _estimate_tail_share(missed_stats, target_ref_quantile, direction)

        # Monotonic combined separation score
        separation_score = round(std_diff + 2.0 * tail_share, 4)

        ranked.append(
            RankedFeature(
                feature_name=feat_name,
                separation_score=separation_score,
                standardized_mean_diff=round(std_diff, 4),
                tail_separation_share=round(tail_share, 4),
                direction=direction,
                missed_mean=round(m_mean, 4),
                reference_mean=round(r_mean, 4),
                reference_p95=round(ref_stats.p95, 4),
                reference_p5=round(ref_stats.p5, 4),
            )
        )

    # Sort descending by separation score, tie-breaking by standardized_mean_diff, then feature name
    ranked.sort(
        key=lambda rf: (-rf.separation_score, -rf.standardized_mean_diff, rf.feature_name)
    )

    return ranked, "OK"
