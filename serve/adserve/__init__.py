"""adserve - real-time anomaly scoring service for AD threat detection.

Tails Wazuh alerts, maintains a per-host sliding window of log keys, scores
each window with the trained sequence model (max-surprisal), and emits
anomalies as JSON that Wazuh can re-ingest to drive Active Response.
"""
__version__ = "1.0.0"
