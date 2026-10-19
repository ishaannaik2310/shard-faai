"""
Shard — Fraudster Agent.

Generates reproducible synthetic motor insurance claims and adapts its attack
typology distribution across rounds based on redacted, aggregate Critic evaluation feedback.

ARCHITECTURAL TRUST BOUNDARY (TI-01, TI-06):
1. No Raw Ground-Truth Access: The Fraudster NEVER accesses GroundTruthVault, individual
   claim labels, or investigator records from previous rounds.
2. Redacted Aggregate Feedback Only: The Fraudster receives only aggregate miss counts
   and detection rates (CriticFeedbackSummary).
3. Seed Determinism (TI-07): Derived seeds guarantee 100% reproducible claim generation
   for identical base seeds and round parameters without mutating global random state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

from shard.core.models import (
    CriticFeedbackSummary,
    CriticReport,
    FraudTypology,
    FraudsterAdaptationRecord,
)
from shard.data.generator import (
    DEFAULT_TYPOLOGY_WEIGHTS,
    ClaimGenerator,
    GenerationBatch,
    GeneratorConfig,
)


@dataclass(frozen=True)
class FraudsterRoundOutput:
    """Explicit return contract from FraudsterAgent for an individual round."""
    batch: GenerationBatch
    adaptation_record: FraudsterAdaptationRecord
    config: GeneratorConfig


class FraudsterAgent:
    """
    Deterministic adaptive adversarial agent.
    
    Adjusts the distribution of synthetic fraud typologies to exploit observed
    investigation gaps reported by aggregate evaluation summaries.
    """

    def __init__(
        self,
        base_seed: int = 42,
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_strength: float = 0.15,
        max_weight: float = 0.60,
        min_weight: float = 0.05,
        include_v3_typologies: bool = False,
        v3_typology_start_round: int = 2,
    ) -> None:
        if adaptation_strength < 0.0:
            raise ValueError(f"adaptation_strength must be non-negative, got {adaptation_strength}")
        if not (0.0 < min_weight <= max_weight <= 1.0):
            raise ValueError(
                f"Invalid weight bounds: require 0 < min ({min_weight}) <= max ({max_weight}) <= 1.0"
            )

        self.base_seed = base_seed
        self.adaptation_strength = adaptation_strength
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.include_v3_typologies = include_v3_typologies
        self.v3_typology_start_round = v3_typology_start_round

        # Normalize and store baseline weights
        weights_source = initial_weights or dict(DEFAULT_TYPOLOGY_WEIGHTS)
        self.current_weights = self._normalize_weights(dict(weights_source))

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize positive weights dictionary to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0.0:
            # Fall back to uniform distribution across existing keys
            return {k: round(1.0 / len(weights), 4) for k in weights}
        return {k: round(v / total, 4) for k, v in weights.items()}

    def derive_round_seed(self, round_number: int) -> int:
        """Derive a reproducible random seed deterministically for round N."""
        if round_number < 1:
            raise ValueError(f"round_number must be >= 1, got {round_number}")
        # Deterministic linear offset formula
        return self.base_seed + (round_number - 1) * 1000

    def adapt_weights(
        self,
        feedback: Optional[Union[CriticFeedbackSummary, CriticReport, Dict[str, Any]]],
    ) -> Tuple[Dict[str, float], Dict[str, Any], bool]:
        """
        Calculate adapted typology sampling weights from aggregate evaluation feedback.

        Adaptation Policy:
        - If feedback is absent or reports 0 misses: weights remain unchanged.
        - Typologies with missed fraud increase proportionally to (1.0 - detection_rate).
        - Typologies with lower miss rates receive no unjustified boost and shrink in relative share.
        - Bounded by [min_weight, max_weight] and normalized.
        """
        prior_weights = dict(self.current_weights)

        if feedback is None:
            return prior_weights, {"status": "NO_PREVIOUS_FEEDBACK"}, False

        # Extract aggregate metrics safely
        if isinstance(feedback, CriticReport):
            summary = CriticFeedbackSummary.from_critic_report(feedback)
            missed_counts = summary.missed_typology_counts
            detection_rates = summary.typology_detection_rates
        elif isinstance(feedback, CriticFeedbackSummary):
            missed_counts = feedback.missed_typology_counts
            detection_rates = feedback.typology_detection_rates
        elif isinstance(feedback, dict):
            missed_counts = feedback.get("missed_typology_counts", {})
            detection_rates = feedback.get("typology_detection_rates", {})
        else:
            raise TypeError(f"Unsupported feedback type: {type(feedback).__name__}")

        total_misses = sum(missed_counts.values()) if missed_counts else 0
        if total_misses == 0:
            return prior_weights, {"status": "ZERO_MISSES_REPORTED"}, False

        # Apply bounded updates
        updated_raw: Dict[str, float] = {}
        for typology, current_w in prior_weights.items():
            miss_count = missed_counts.get(typology, 0)
            if miss_count > 0:
                det_rate = detection_rates.get(typology, 0.0)
                miss_rate = max(0.0, min(1.0, 1.0 - det_rate))
                boost = self.adaptation_strength * miss_rate
                new_w = current_w + boost
            else:
                # No misses -> no boost; will contract relatively after normalization
                new_w = current_w

            # Clamp to allowed bounds
            clamped_w = max(self.min_weight, min(self.max_weight, new_w))
            updated_raw[typology] = clamped_w

        normalized = self._normalize_weights(updated_raw)
        feedback_info = {
            "total_misses": total_misses,
            "missed_typologies": dict(missed_counts),
        }
        return normalized, feedback_info, True

    def generate_round(
        self,
        round_number: int = 1,
        feedback: Optional[Union[CriticFeedbackSummary, CriticReport, Dict[str, Any]]] = None,
        n_claims: int = 30,
        fraud_share: float = 0.3,
    ) -> FraudsterRoundOutput:
        """
        Execute round-level claim generation with feedback-driven adaptation.
        
        Outputs GenerationBatch (containing observable claims and restricted vault)
        and an auditable FraudsterAdaptationRecord.
        """
        prior_weights = dict(self.current_weights)

        # In v3 mode: introduce DELAYED_FILING from start_round onward
        if self.include_v3_typologies and round_number >= self.v3_typology_start_round:
            if FraudTypology.DELAYED_FILING.value not in self.current_weights:
                self.current_weights[FraudTypology.DELAYED_FILING.value] = 0.15
                self.current_weights = self._normalize_weights(self.current_weights)

        # Adapt weights if feedback is provided for subsequent rounds
        if round_number > 1 and feedback is not None:
            updated_weights, feedback_info, adapted = self.adapt_weights(feedback)
            self.current_weights = updated_weights
        else:
            updated_weights = dict(self.current_weights)
            feedback_info = {"status": "INITIAL_DISTRIBUTION"}
            adapted = False

        derived_seed = self.derive_round_seed(round_number)
        config = GeneratorConfig(
            seed=derived_seed,
            n_claims=n_claims,
            fraud_share=fraud_share,
            typology_weights=dict(self.current_weights),
            include_v3_typologies=self.include_v3_typologies,
            v3_typology_start_round=self.v3_typology_start_round,
            round_number=round_number,
        )

        batch = ClaimGenerator(config).generate()

        record = FraudsterAdaptationRecord(
            round_number=round_number,
            base_seed=self.base_seed,
            derived_seed=derived_seed,
            prior_weights=prior_weights,
            updated_weights=dict(self.current_weights),
            feedback_summary=feedback_info,
            adapted=adapted,
        )

        return FraudsterRoundOutput(
            batch=batch,
            adaptation_record=record,
            config=config,
        )
