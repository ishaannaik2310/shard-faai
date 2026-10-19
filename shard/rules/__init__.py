"""
Shard Rules — Deterministic fraud verification tools, templates, and versioned rulebook.
"""

from shard.rules.rulebook import (
    DispositionThresholds,
    RuleDefinition,
    Rulebook,
    load_default_rulebook,
)
from shard.rules.templates import (
    TEMPLATE_REGISTRY,
    ToolCheckResult,
)
from shard.rules.tools import (
    ALL_VERIFICATION_TOOLS,
    check_prior_damage,
    check_repair_benchmark,
    check_repair_line_items,
    check_staged_collision_indicators,
    check_vehicle_registration,
    check_vehicle_write_off,
    execute_all_default_tools,
)

__all__ = [
    "ALL_VERIFICATION_TOOLS",
    "DispositionThresholds",
    "RuleDefinition",
    "Rulebook",
    "TEMPLATE_REGISTRY",
    "ToolCheckResult",
    "check_prior_damage",
    "check_repair_benchmark",
    "check_repair_line_items",
    "check_staged_collision_indicators",
    "check_vehicle_registration",
    "check_vehicle_write_off",
    "execute_all_default_tools",
    "load_default_rulebook",
]
