"""Entry point: wire the stream -> featurize -> window -> score -> emit loop.

    python -m adserve.service --bundle-dir serve_bundle \
        --alerts /var/ossec/logs/alerts/alerts.json

Replay a historical file for a demo or offline evaluation (no live attack):

    python -m adserve.service --bundle-dir serve_bundle \
        --replay data/alerts_2025-11-09.json --out /tmp/anoms.json
"""
from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path

from .config import ServeConfig
from .emit import AnomalyEmitter
from .featurize import Featurizer
from .scorer import ModelBundle
from .tailer import follow, replay
from .windows import WindowStore

log = logging.getLogger("adserve")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live AD anomaly scoring service.")
    p.add_argument("--bundle-dir", type=Path, help="dir with model.pt/meta.json/vocab.json")
    p.add_argument("--alerts", type=Path, help="Wazuh alerts.json to follow")
    p.add_argument("--out", type=Path, help="output anomalies JSON-lines")
    p.add_argument("--replay", type=Path, help="score a static file then exit")
    p.add_argument("--threshold", type=float, help="override decision threshold")
    p.add_argument("--window-size", type=int, help="override window length")
    p.add_argument("--device", help="cpu or cuda")
    p.add_argument("--log-key-field", help="dotted path to the log key")
    p.add_argument("--host-field", help="dotted path to the host id")
    p.add_argument("--from-start", action="store_true", help="follow from BOF, not EOF")
    p.add_argument("--log-level", default=None)
    return p


def cfg_from_args(args) -> ServeConfig:
    c = ServeConfig()
    if args.bundle_dir:    c.bundle_dir = args.bundle_dir
    if args.alerts:        c.alerts_path = args.alerts
    if args.out:           c.out_path = args.out
    if args.threshold is not None:   c.threshold = args.threshold
    if args.window_size:   c.window_size = args.window_size
    if args.device:        c.device = args.device
    if args.log_key_field: c.log_key_field = args.log_key_field
    if args.host_field:    c.host_field = args.host_field
    if args.log_level:     c.log_level = args.log_level
    return c


def run(cfg: ServeConfig, *, replay_path: Path | None = None, from_start: bool = False) -> int:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bundle = ModelBundle.load(cfg.bundle_dir, device=cfg.device)
    if not math.isnan(cfg.threshold):
        threshold = cfg.threshold
    else:
        threshold = cfg.resolved_threshold(bundle.meta)
    window_size = bundle.window_size or cfg.window_size
    model_name = bundle.meta.get("model", "model")

    # The featurizer's composite-key spec comes from the bundle, so the live
    # log key matches the training definition exactly.
    featurizer = Featurizer.from_meta(bundle.vocab, bundle.meta)
    store = WindowStore(window_size, cfg.idle_evict_s)
    emitter = AnomalyEmitter(cfg.out_path)

    log.info("loaded %s | vocab=%d window=%d threshold=%.3f key_fields=%s device=%s",
             model_name, len(bundle.vocab), window_size, threshold,
             "|".join(featurizer.key_fields), cfg.device)

    if replay_path is not None:
        source = replay(replay_path)
        log.info("REPLAY %s", replay_path)
    else:
        source = follow(cfg.alerts_path, cfg.poll_interval_s, from_start=from_start)
        log.info("FOLLOW %s", cfg.alerts_path)

    _install_signal_handlers()

    seen = scored = fired = 0
    last_evict = time.monotonic()
    for alert in source:
        if _STOP:
            break
        seen += 1
        feat = featurizer(alert)
        if feat is None:
            continue
        host, token_id = feat
        window = store.push(host, token_id)
        if window is None:
            continue
        scored += 1
        score = bundle.score(window)
        if score >= threshold:
            detail = bundle.explain(window)
            tokens = [bundle.vocab.id_to_token.get(t, "<unk>") for t in window]
            emitter.emit(host=host, score=score, threshold=threshold,
                         model=model_name, window_tokens=tokens, detail=detail)
            fired += 1

        now = time.monotonic()
        if now - last_evict > 60:
            store.evict_idle()
            last_evict = now

    log.info("done: seen=%d scored=%d anomalies=%d", seen, scored, fired)
    return 0


_STOP = False


def _install_signal_handlers():
    def _handler(signum, _frame):
        global _STOP
        _STOP = True
        log.info("signal %d: shutting down after current event", signum)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except ValueError:
            pass  # not in main thread (e.g. under test): ignore


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = cfg_from_args(args)
    return run(cfg, replay_path=args.replay, from_start=args.from_start)


if __name__ == "__main__":
    sys.exit(main())
