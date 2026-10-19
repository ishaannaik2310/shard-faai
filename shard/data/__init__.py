"""Shard Data — Synthetic Claim Generation and Ground-Truth Storage."""

from shard.data.generator import (
    ClaimGenerator,
    DEFAULT_TYPOLOGY_WEIGHTS,
    GenerationBatch,
    GeneratorConfig,
    generate_claims,
)
from shard.data.vault import GroundTruthVault

__all__ = [
    "ClaimGenerator",
    "DEFAULT_TYPOLOGY_WEIGHTS",
    "GenerationBatch",
    "GeneratorConfig",
    "GroundTruthVault",
    "generate_claims",
]
