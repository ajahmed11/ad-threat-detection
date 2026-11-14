"""Emit anomalies as JSON-lines that Wazuh can re-ingest.

Each anomaly is one JSON object appended to out_path. A <localfile> stanza in
ossec.conf tails that file, a small decoder/rule pair lifts the score, and the
existing Active Response chain (Disable-ADAccount) can fire on it. The service
never contains an account itself: it raises a signal and lets the SIEM decide,
which keeps the ML engine advisory and auditable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("adserve.emit")


class AnomalyEmitter:
    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

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
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.warning("ANOMALY host=%s score=%.3f thr=%.3f model=%s peak=%s",
                    host, score, threshold, model, detail.get("peak_token"))
        return record
