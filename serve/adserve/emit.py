"""Emit anomalies as JSON-lines that Wazuh can re-ingest.

Each anomaly is one JSON object appended to out_path. A <localfile> stanza in
ossec.conf tails that file so the anomaly appears in Wazuh as an ALERT for an
analyst. The service is ADVISORY ONLY: records carry no Windows account fields
(data.win.eventdata.*), so they can never drive Disable-ADAccount. Identity
attribution + approval would be a separate, reviewed workflow.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("adserve.emit")


class AnomalyEmitter:
    def __init__(self, out_path: Path, max_bytes: int = 50 << 20, backups: int = 5):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.backups = int(backups)

    def _rotate_if_needed(self) -> None:
        """Size-based rotation: ml_anomalies.json -> .1 -> .2 ... (bounded retention)."""
        try:
            if self.out_path.stat().st_size < self.max_bytes:
                return
        except FileNotFoundError:
            return
        for i in range(self.backups - 1, 0, -1):
            src = self.out_path.with_name(f"{self.out_path.name}.{i}")
            if src.exists():
                src.replace(self.out_path.with_name(f"{self.out_path.name}.{i + 1}"))
        self.out_path.replace(self.out_path.with_name(f"{self.out_path.name}.1"))

    def emit(self, *, host: str, score: float, threshold: float, model: str,
             window_tokens: list[str], detail: dict) -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integration": "adserve",
            "model": model,
            "host": host,
            "anomaly_score": round(float(score), 4),
            "threshold": round(float(threshold), 4),
            "severity": "high" if score >= threshold * 1.5 else "medium",
            "window": window_tokens,
            "detail": detail,
        }
        self._rotate_if_needed()
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.warning("ANOMALY host=%s score=%.3f thr=%.3f model=%s peak=%s",
                    host, score, threshold, model, detail.get("peak_token"))
        return record
