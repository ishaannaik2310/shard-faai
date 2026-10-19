"""
Shard — Audit Logging Engine.

Produces structured, auditable JSON Lines (.jsonl) event trails for multi-agent
insurance fraud investigation cycles.

CRITICAL INVARIANTS:
1. Ground-Truth Isolation (TI-01, TI-09): The audit logger NEVER records raw ground-truth
   records, hidden fraud labels (is_fraud), or confidential vault payloads.
2. Complete Audit Coverage (TI-02, TI-09): Every completed ClaimDecision is recorded as an
   individual audit event; total decision events must exactly match processed claims.
3. Stable Event Ordering: Events have monotonic sequence numbers (1, 2, 3...) and stable
   event categorisation.
4. Reproducibility Support: Canonical event representations can be compared across runs
   independent of wall-clock timestamps.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from shard.core.models import (
    ClaimDecision,
    CriticReport,
    FraudsterAdaptationRecord,
    RuleAuditRecord,
    RuleProposal,
)


@dataclass(frozen=True)
class AuditEvent:
    """An individual immutable audit event entry."""
    event_type: str
    sequence: int
    run_id: str
    round_number: int
    payload: Dict[str, Any]
    timestamp: str = ""

    def to_dict(self, include_timestamp: bool = True) -> Dict[str, Any]:
        """Convert event to a dictionary for JSONL serialisation."""
        data = {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "round_number": self.round_number,
            "payload": self.payload,
        }
        if include_timestamp:
            data["timestamp"] = self.timestamp
        return data

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return a canonical dictionary without wall-clock timestamps for reproducibility diffs."""
        return self.to_dict(include_timestamp=False)

    def to_json(self, include_timestamp: bool = True) -> str:
        """Serialise event to a single JSON line."""
        return json.dumps(self.to_dict(include_timestamp=include_timestamp))


class AuditLogger:
    """
    Append-only structured logger writing JSON Lines to disk or in-memory streams.
    """

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self.run_id = run_id or f"RUN-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self._sequence = 0
        self._events: List[AuditEvent] = []

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Create or truncate log file
            with open(self.log_path, "w", encoding="utf-8") as f:
                pass

    @property
    def event_count(self) -> int:
        return len(self._events)

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce privacy and ground-truth boundary safeguards:
        Strip prohibited hidden labels (is_fraud, ground_truth, vault_record).
        """
        prohibited_keys = {"is_fraud", "ground_truth", "vault_record", "raw_gt"}
        sanitized = {}
        for k, v in payload.items():
            if k.lower() in prohibited_keys:
                continue
            if isinstance(v, dict):
                sanitized[k] = self._sanitize_payload(v)
            else:
                sanitized[k] = v
        return sanitized

    def log_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        round_number: int = 1,
    ) -> AuditEvent:
        """Record a structured event into the audit stream."""
        if not event_type or not isinstance(event_type, str):
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")

        self._sequence += 1
        clean_payload = self._sanitize_payload(payload)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        event = AuditEvent(
            event_type=event_type,
            sequence=self._sequence,
            run_id=self.run_id,
            round_number=round_number,
            payload=clean_payload,
            timestamp=now_iso,
        )
        self._events.append(event)

        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")

        return event

    def log_experiment_init(self, config_dict: Dict[str, Any]) -> AuditEvent:
        """Log experiment configuration and initialization parameters."""
        return self.log_event(
            event_type="EXPERIMENT_INIT",
            payload={"config": config_dict},
            round_number=0,
        )

    def log_round_start(
        self,
        round_number: int,
        rulebook_version: str,
        seed: int,
    ) -> AuditEvent:
        """Log round commencement."""
        return self.log_event(
            event_type="ROUND_START",
            payload={
                "rulebook_version": rulebook_version,
                "round_seed": seed,
            },
            round_number=round_number,
        )

    def log_claim_decision(
        self,
        decision: ClaimDecision,
        round_number: int,
    ) -> AuditEvent:
        """
        Log an individual claim decision with its triage disposition, score,
        and audit findings summary without exposing hidden ground truth.
        """
        payload = {
            "claim_id": decision.claim_id,
            "risk_score": decision.risk_score,
            "disposition": decision.disposition.value,
            "findings_count": len(decision.findings),
            "triggered_rules": [
                f.check_name for f in decision.findings if "FLAGGED" in f.result
            ],
            "tool_trace_steps": len(decision.tool_trace),
        }
        return self.log_event(
            event_type="CLAIM_DECISION",
            payload=payload,
            round_number=round_number,
        )

    def log_critic_report(
        self,
        report: CriticReport,
        round_number: int,
    ) -> AuditEvent:
        """Log aggregate Critic evaluation metrics and confusion counts."""
        payload = {
            "total_evaluated": report.total_evaluated,
            "tp": report.tp,
            "fp": report.fp,
            "tn": report.tn,
            "fn": report.fn,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "fpr": report.fpr,
            "accuracy": report.accuracy,
            "disposition_counts": dict(report.disposition_counts),
            "missed_typology_counts": dict(report.missed_typology_counts),
            "consistency_violations_count": len(report.consistency_violations),
        }
        return self.log_event(
            event_type="CRITIC_REPORT",
            payload=payload,
            round_number=round_number,
        )

    def log_rule_proposal(
        self,
        proposal: RuleProposal,
        round_number: int,
    ) -> AuditEvent:
        """Log a candidate rulebook modification proposed by the Rule-Writer."""
        return self.log_event(
            event_type="RULE_PROPOSAL",
            payload=proposal.to_dict(),
            round_number=round_number,
        )

    def log_rule_validation(
        self,
        record: RuleAuditRecord,
        round_number: int,
    ) -> AuditEvent:
        """Log validation gate outcome for a proposed rule change."""
        return self.log_event(
            event_type="RULE_VALIDATION",
            payload=record.to_dict(),
            round_number=round_number,
        )

    def log_fraudster_adaptation(
        self,
        record: FraudsterAdaptationRecord,
        round_number: int,
    ) -> AuditEvent:
        """Log Fraudster attack typology distribution shifts."""
        return self.log_event(
            event_type="FRAUDSTER_ADAPTATION",
            payload=record.to_dict(),
            round_number=round_number,
        )

    def log_discovery_ranking(
        self,
        ranking_payload: Dict[str, Any],
        round_number: int,
    ) -> AuditEvent:
        """Log feature separation ranking produced by generic discovery."""
        return self.log_event(
            event_type="DISCOVERY_RANKING",
            payload=ranking_payload,
            round_number=round_number,
        )

    def log_round_end(
        self,
        round_number: int,
        claims_count: int,
        active_version: str,
        continuation_reason: str,
    ) -> AuditEvent:
        """Log round completion and status."""
        return self.log_event(
            event_type="ROUND_END",
            payload={
                "claims_processed": claims_count,
                "active_rulebook_version": active_version,
                "continuation_reason": continuation_reason,
            },
            round_number=round_number,
        )

    def log_experiment_end(
        self,
        final_stop_reason: str,
        target_reached: bool,
        total_rounds: int,
        total_claims: int,
        final_rulebook_version: str,
        cumulative_metrics: Dict[str, Any],
    ) -> AuditEvent:
        """Log experiment conclusion and final cumulative outcome."""
        return self.log_event(
            event_type="EXPERIMENT_END",
            payload={
                "final_stop_reason": final_stop_reason,
                "target_reached": target_reached,
                "total_rounds": total_rounds,
                "total_claims": total_claims,
                "final_rulebook_version": final_rulebook_version,
                "cumulative_metrics": cumulative_metrics,
            },
            round_number=total_rounds,
        )

    def get_events(self) -> List[AuditEvent]:
        """Return all recorded audit events."""
        return list(self._events)

    def get_events_by_type(self, event_type: str) -> List[AuditEvent]:
        """Filter audit events by event type."""
        return [e for e in self._events if e.event_type == event_type]

    def decision_coverage(self) -> Dict[str, int]:
        """Return counts of processed claims and recorded decision events."""
        decision_events = self.get_events_by_type("CLAIM_DECISION")
        return {
            "decision_events_count": len(decision_events),
            "total_audit_events": len(self._events),
        }
