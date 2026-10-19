"""
Shard — Versioned Rulebook Engine.

Loads, validates, and executes versioned rule configurations against observable claims.
Enforces deterministic scoring, centralised thresholds, and traceable findings.

CRITICAL INVARIANTS:
1. Configuration-only: Loaded from JSON; never executes arbitrary code.
2. Centralised thresholds: Dispositions (APPROVE, REFER, SUSPECTED_FRAUD) are computed
   strictly from centralised rulebook thresholds.
3. Ground-truth isolation: Rulebook operates purely on observable Claim instances;
   it never imports, calls, or accesses GroundTruthVault.
4. Bounded scoring: Risk scores strictly adhere to the 0.0 to 100.0 scale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from shard.core.models import Claim, ClaimDecision, Disposition, Finding
from shard.rules.templates import TEMPLATE_REGISTRY, ToolCheckResult


@dataclass(frozen=True)
class DispositionThresholds:
    """Triage boundary thresholds on the 0.0 to 100.0 scale."""
    approve_below: float = 25.0
    refer_below: float = 60.0
    suspected_fraud_at_or_above: float = 60.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.approve_below <= self.refer_below <= 100.0):
            raise ValueError(
                f"Invalid disposition thresholds: require 0 <= approve ({self.approve_below}) "
                f"<= refer ({self.refer_below}) <= 100.0"
            )
        if self.suspected_fraud_at_or_above != self.refer_below:
            raise ValueError(
                f"suspected_fraud_at_or_above ({self.suspected_fraud_at_or_above}) must equal "
                f"refer_below ({self.refer_below}) to guarantee complete partition."
            )

    def determine_disposition(self, risk_score: float) -> Disposition:
        """Map a risk score (0.0 to 100.0) to a triage Disposition recommendation."""
        if risk_score < self.approve_below:
            return Disposition.APPROVE
        elif risk_score < self.refer_below:
            return Disposition.REFER
        else:
            return Disposition.SUSPECTED_FRAUD


@dataclass(frozen=True)
class RuleDefinition:
    """Explicit definition of a single configurable verification rule."""
    rule_id: str
    template_id: str
    description: str
    enabled: bool = True
    severity: str = "MEDIUM"
    score_contribution: float = 20.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id or not isinstance(self.rule_id, str):
            raise ValueError("RuleDefinition rule_id must be a non-empty string")
        if self.template_id not in TEMPLATE_REGISTRY:
            raise ValueError(
                f"Unknown template_id '{self.template_id}'. Must be one of: {list(TEMPLATE_REGISTRY.keys())}"
            )
        if not (0.0 <= self.score_contribution <= 100.0):
            raise ValueError(
                f"Rule '{self.rule_id}' score_contribution must be between 0.0 and 100.0, got {self.score_contribution}"
            )
        if not isinstance(self.parameters, dict):
            raise TypeError(f"Rule '{self.rule_id}' parameters must be a dictionary")


@dataclass
class Rulebook:
    """
    Versioned repository of fraud verification rules and centralised disposition thresholds.
    """
    version: str
    name: str
    description: str
    disposition_thresholds: DispositionThresholds
    rules: List[RuleDefinition]
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate rulebook integrity and enforce uniqueness of rule identifiers."""
        if not self.version or not isinstance(self.version, str):
            raise ValueError("Rulebook version must be a non-empty string")
        if not self.rules:
            raise ValueError("Rulebook must contain at least one RuleDefinition")

        seen_ids = set()
        for r in self.rules:
            if not isinstance(r, RuleDefinition):
                raise TypeError(f"Expected RuleDefinition, got {type(r).__name__}")
            if r.rule_id in seen_ids:
                raise ValueError(f"Duplicate rule_id '{r.rule_id}' in rulebook")
            seen_ids.add(r.rule_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Rulebook:
        """Construct a validated Rulebook instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("Rulebook configuration data must be a dictionary")

        thresh_raw = data.get("disposition_thresholds", {})
        thresholds = DispositionThresholds(
            approve_below=float(thresh_raw.get("approve_below", 25.0)),
            refer_below=float(thresh_raw.get("refer_below", 60.0)),
            suspected_fraud_at_or_above=float(thresh_raw.get("suspected_fraud_at_or_above", 60.0)),
        )

        rules_raw = data.get("rules", [])
        if not isinstance(rules_raw, list):
            raise TypeError("rules in rulebook configuration must be a list")

        rule_defs: List[RuleDefinition] = []
        for r in rules_raw:
            rule_defs.append(
                RuleDefinition(
                    rule_id=str(r.get("rule_id", "")),
                    template_id=str(r.get("template_id", "")),
                    description=str(r.get("description", "")),
                    enabled=bool(r.get("enabled", True)),
                    severity=str(r.get("severity", "MEDIUM")),
                    score_contribution=float(r.get("score_contribution", 20.0)),
                    parameters=dict(r.get("parameters", {})),
                )
            )

        return cls(
            version=str(data.get("version", "1.0.0")),
            name=str(data.get("name", "Untitled Rulebook")),
            description=str(data.get("description", "")),
            disposition_thresholds=thresholds,
            rules=rule_defs,
            history=list(data.get("history", [])),
        )

    @classmethod
    def load(cls, path: Union[Path, str]) -> Rulebook:
        """Load and validate a Rulebook from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Rulebook configuration file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in rulebook file '{file_path}': {e}") from e

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Export rulebook configuration to a serializable dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "disposition_thresholds": {
                "approve_below": self.disposition_thresholds.approve_below,
                "refer_below": self.disposition_thresholds.refer_below,
                "suspected_fraud_at_or_above": self.disposition_thresholds.suspected_fraud_at_or_above,
            },
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "template_id": r.template_id,
                    "description": r.description,
                    "enabled": r.enabled,
                    "severity": r.severity,
                    "score_contribution": r.score_contribution,
                    "parameters": r.parameters,
                }
                for r in self.rules
            ],
            "history": self.history,
        }

    def save(self, path: Union[Path, str]) -> None:
        """Save rulebook configuration to JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def clone(self) -> Rulebook:
        """Create an independent, deep-copied clone of this Rulebook."""
        return Rulebook.from_dict(self.to_dict())

    def get_rule(self, rule_id: str) -> RuleDefinition:
        """Fetch a specific RuleDefinition by ID."""
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(f"Rule '{rule_id}' not found in rulebook")

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a specific rule is enabled."""
        return self.get_rule(rule_id).enabled

    def evaluate_claim(self, claim: Claim, round_number: int = 1) -> ClaimDecision:
        """
        Evaluate an observable claim against all enabled rules in the rulebook.
        
        Deterministic Risk Scoring:
        1. Executes each enabled rule via its corresponding parameterised template.
        2. Aggregates triggered score contributions without double-counting.
        3. Clamps final score to [0.0, 100.0].
        4. Maps score to disposition using centralised thresholds.
        5. Returns a ClaimDecision with complete, non-empty audit findings.
        """
        if not isinstance(claim, Claim):
            raise TypeError(f"Expected Claim instance, got {type(claim).__name__}")

        findings: List[Finding] = []
        tool_traces: List[Dict[str, Any]] = []
        raw_score = 0.0

        for rule in self.rules:
            template_fn = TEMPLATE_REGISTRY[rule.template_id]

            if not rule.enabled:
                # Disabled rule: record audit pass with 0 contribution
                finding = Finding(
                    check_name=rule.rule_id,
                    result="DISABLED",
                    risk_contribution=0.0,
                    detail=f"Rule '{rule.rule_id}' is disabled in rulebook v{self.version}.",
                )
                findings.append(finding)
                tool_traces.append({
                    "rule_id": rule.rule_id,
                    "enabled": False,
                    "triggered": False,
                    "points": 0.0,
                })
                continue

            # Execute template with rule's configured parameters
            result: ToolCheckResult = template_fn(
                claim=claim,
                score_contribution=rule.score_contribution,
                severity=rule.severity,
                rule_id=rule.rule_id,
                **rule.parameters,
            )

            if result.triggered:
                raw_score += result.score_contribution

            findings.append(result.to_finding())
            tool_traces.append({
                "rule_id": rule.rule_id,
                "template_id": rule.template_id,
                "enabled": True,
                "triggered": result.triggered,
                "points": result.score_contribution if result.triggered else 0.0,
                "severity": result.severity,
                "observed": result.observed_value,
                "detail": result.detail,
            })

        return self.build_decision(
            claim_id=claim.claim_id,
            findings=findings,
            tool_traces=tool_traces,
            round_number=round_number,
        )

    def score_findings(self, findings: Sequence[Finding]) -> Tuple[float, Disposition]:
        """
        Compute bounded risk score and centralized disposition from a sequence of Findings.
        Enforces 0.0 to 100.0 score bounds and centralised disposition triage.
        """
        raw_score = sum(f.risk_contribution for f in findings)
        final_score = min(100.0, max(0.0, round(raw_score, 1)))
        disposition = self.disposition_thresholds.determine_disposition(final_score)
        return final_score, disposition

    def build_decision(
        self,
        claim_id: str,
        findings: Sequence[Finding],
        tool_traces: Sequence[Dict[str, Any]],
        round_number: int = 1,
    ) -> ClaimDecision:
        """Construct a validated ClaimDecision from findings using the rulebook's centralized thresholds."""
        final_score, disposition = self.score_findings(findings)
        return ClaimDecision(
            claim_id=claim_id,
            risk_score=final_score,
            disposition=disposition,
            findings=tuple(findings),
            tool_trace=tuple(tool_traces),
            round_number=round_number,
        )


def load_default_rulebook() -> Rulebook:
    """Load the canonical initial rulebook from data/rulebook_v1.json."""
    default_path = Path(__file__).resolve().parent.parent.parent / "data" / "rulebook_v1.json"
    return Rulebook.load(default_path)
