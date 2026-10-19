"""
Shard — Ground-Truth Vault.

This module provides a secure, evaluation-only repository for synthetic ground-truth
labels. It strictly isolates hidden fraud labels from observable claim payloads.

ARCHITECTURAL TRUST BOUNDARY (TI-01):
- Authorised callers: Critic Agent, Orchestrator (stopping-condition check only).
- Prohibited callers: Planner, Research Agent, Investigator Agent.
- Under Python's shared-process model, this boundary is enforced by convention,
  explicit software design, and static/lint tests in the test suite.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from shard.core.models import GroundTruthRecord


class GroundTruthVault:
    """
    Isolated vault storing ground-truth evaluation labels.
    
    Guarantees:
    1. Internal records dictionary is private and cannot be mutated by callers.
    2. Read operations return frozen dataclass instances.
    3. Lookups validate requested claim IDs and raise KeyError for unknown claims.
    """

    def __init__(self, records: Optional[Iterable[GroundTruthRecord]] = None) -> None:
        self._records: Dict[str, GroundTruthRecord] = {}
        if records is not None:
            for record in records:
                self.register(record)

    def register(self, record: GroundTruthRecord) -> None:
        """Register a single ground-truth record. Duplicate IDs are rejected."""
        if not isinstance(record, GroundTruthRecord):
            raise TypeError(f"Expected GroundTruthRecord, got {type(record).__name__}")
        if not isinstance(record.claim_id, str) or not record.claim_id.strip():
            raise ValueError("GroundTruthRecord must have a non-empty string claim_id")
        if record.claim_id in self._records:
            raise ValueError(f"Ground-truth record already registered for claim: '{record.claim_id}'")
        self._records[record.claim_id] = record

    def get_label(self, claim_id: str) -> GroundTruthRecord:
        """
        Retrieve ground-truth label for a specific claim.
        
        Raises KeyError if claim_id is not found in the vault.
        """
        if not isinstance(claim_id, str):
            raise TypeError(f"claim_id must be a string, got {type(claim_id).__name__}")
        if claim_id not in self._records:
            raise KeyError(f"No ground-truth record found for claim ID: '{claim_id}'")
        return self._records[claim_id]

    def get_labels(self, claim_ids: Sequence[str]) -> Dict[str, GroundTruthRecord]:
        """
        Retrieve ground-truth labels for an ordered sequence of claim IDs.
        
        Returns a newly allocated dictionary mapping claim_id -> GroundTruthRecord.
        Raises KeyError if any requested claim_id is unknown.
        """
        if not isinstance(claim_ids, (list, tuple, set)):
            raise TypeError(f"claim_ids must be a sequence of strings, got {type(claim_ids).__name__}")

        result: Dict[str, GroundTruthRecord] = {}
        missing: List[str] = []

        for cid in claim_ids:
            if not isinstance(cid, str):
                raise TypeError(f"All claim IDs must be strings, got {type(cid).__name__}")
            if cid in self._records:
                result[cid] = self._records[cid]
            else:
                missing.append(cid)

        if missing:
            raise KeyError(f"Vault missing ground-truth records for claim IDs: {missing}")

        return result

    def contains(self, claim_id: str) -> bool:
        """Check whether a claim ID exists in the vault."""
        return isinstance(claim_id, str) and claim_id in self._records

    def __contains__(self, claim_id: object) -> bool:
        return isinstance(claim_id, str) and claim_id in self._records

    def __len__(self) -> int:
        return len(self._records)

    def claim_ids(self) -> List[str]:
        """Return a copy of all registered claim IDs."""
        return list(self._records.keys())

    def fraud_ratio(self) -> float:
        """Compute the empirical fraud fraction within the vault."""
        if not self._records:
            return 0.0
        n_fraud = sum(1 for r in self._records.values() if r.is_fraud)
        return round(n_fraud / len(self._records), 4)

    def __repr__(self) -> str:
        return f"<GroundTruthVault records={len(self._records)} fraud_ratio={self.fraud_ratio():.2f}>"
