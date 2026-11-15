"""Runtime configuration for the live scoring service.

Every value can be overridden by an environment variable (prefix ADSERVE_)
or a --flag on the CLI, so the same code runs on the Ubuntu SIEM node and on
a laptop in replay mode without edits. Defaults assume a standard Wazuh
single-node install.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default):
    return os.environ.get(f"ADSERVE_{name}", default)


@dataclass
class ServeConfig:
    # --- Model bundle (produced by export_bundle.py) -----------------------
    # Directory holding model.pt + meta.json + vocab.json.
    bundle_dir: Path = field(default_factory=lambda: Path(_env("BUNDLE_DIR", "serve_bundle")))

    # Trusted SHA-256 of model.pt, supplied OUT-OF-BAND (env/CLI), never from the bundle.
    model_sha256: str = field(default_factory=lambda: str(_env("MODEL_SHA256", "")))

    # --- Input: the Wazuh alert stream (JSON-lines) ------------------------
    # alerts.json is written one JSON object per line by the Wazuh manager.
    alerts_path: Path = field(default_factory=lambda: Path(_env("ALERTS_PATH", "/var/ossec/logs/alerts/alerts.json")))

    # --- Output: anomalies the manager re-ingests --------------------------
    # Written as JSON-lines; a <localfile> in ossec.conf tails this file so
    # the existing rule/active-response chain can act on it.
    out_path: Path = field(default_factory=lambda: Path(_env("OUT_PATH", "/var/ossec/logs/ml_anomalies.json")))

    # --- Log-key extraction -------------------------------------------------
    # Dotted path into the alert JSON that yields the discrete log key.
    # MUST match the event-key definition used by the training pipeline
    # (build_dataset.py). Default: Wazuh rule id. meta.json overrides this.
    log_key_field: str = str(_env("LOG_KEY_FIELD", "rule.id"))
    # Dotted path to the host identifier used to separate per-host windows.
    host_field: str = str(_env("HOST_FIELD", "agent.name"))

    # --- Scoring ------------------------------------------------------------
    # Window length and decision threshold normally come from meta.json
    # (frozen at train time). These are fallbacks only.
    window_size: int = int(_env("WINDOW_SIZE", "20"))
    threshold: float = float(_env("THRESHOLD", "nan"))  # NaN -> use meta.json
    device: str = str(_env("DEVICE", "cpu"))            # tiny windows: CPU is fine

    # --- Operational --------------------------------------------------------
    # Seconds a host window may sit idle before it is flushed from memory.
    idle_evict_s: float = float(_env("IDLE_EVICT_S", "900"))
    poll_interval_s: float = float(_env("POLL_INTERVAL_S", "0.5"))
    log_level: str = str(_env("LOG_LEVEL", "INFO"))
    # Resource ceilings (a hostile/garbled log must not exhaust memory).
    max_line_bytes: int = int(_env("MAX_LINE_BYTES", str(1 << 20)))   # 1 MiB per JSON line
    max_hosts: int = int(_env("MAX_HOSTS", "512"))
    out_max_bytes: int = int(_env("OUT_MAX_BYTES", str(50 << 20)))    # rotate anomalies file at 50 MiB
    out_backups: int = int(_env("OUT_BACKUPS", "5"))

    def resolved_threshold(self, meta: dict) -> float:
        """CLI/env threshold wins; otherwise the value frozen at train time."""
        import math
        if not math.isnan(self.threshold):
            return self.threshold
        t = meta.get("threshold")
        if t is None:
            raise ValueError(
                "No threshold set: pass --threshold or record one in meta.json "
                "(e.g. the 99th percentile of benign training scores)."
            )
        return float(t)
