"""Robust JSON-lines tailer for Wazuh alerts.json.

Follows a growing file the way `tail -F` does: survives log rotation (inode
change) and truncation (size shrink), and buffers partial trailing lines until
the newline arrives. Yields one parsed JSON object per complete line.

In replay mode it reads a static file start-to-finish and stops, so the exact
same scoring path can be driven offline for evaluation or a viva demo.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterator, Optional


log = logging.getLogger("adserve.tailer")
DEFAULT_MAX_LINE = 1 << 20   # 1 MiB: far above any real Wazuh alert
_CHUNK = 64 * 1024


def _inode(path: Path) -> Optional[int]:
    try:
        return os.stat(path).st_ino
    except FileNotFoundError:
        return None


def iter_json_lines(line_iter, max_line_bytes: int = DEFAULT_MAX_LINE) -> Iterator[dict]:
    for line in line_iter:
        if len(line) > max_line_bytes:
            log.warning("dropping oversized line (%d chars > %d)", len(line), max_line_bytes)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            # A malformed/hostile line never stalls the stream; skip and continue.
            continue
        if isinstance(obj, dict):
            yield obj


def replay(path: Path, max_line_bytes: int = DEFAULT_MAX_LINE) -> Iterator[dict]:
    """Yield every JSON object in a static file, then stop.

    Reads with a bounded readline so one gigantic line cannot be loaded whole.
    """
    with open(path, "r", errors="replace") as fh:
        while True:
            line = fh.readline(max_line_bytes + 1)
            if not line:
                return
            if len(line) > max_line_bytes and not line.endswith("\n"):
                # oversized: drain the remainder of this line without keeping it
                while True:
                    rest = fh.readline(_CHUNK)
                    if not rest or rest.endswith("\n"):
                        break
                log.warning("dropping oversized line (> %d chars)", max_line_bytes)
                continue
            yield from iter_json_lines([line], max_line_bytes)


def follow(path: Path, poll_interval_s: float = 0.5, from_start: bool = False,
           max_line_bytes: int = DEFAULT_MAX_LINE) -> Iterator[dict]:
    """Yield JSON objects as they are appended to `path`, forever."""
    path = Path(path)
    buf = ""
    skipping = False          # inside an oversized line: discard until its newline
    fh = None
    cur_inode = None
    try:
        while True:
            if fh is None:
                if not path.exists():
                    time.sleep(poll_interval_s)
                    continue
                fh = open(path, "r", errors="replace")
                cur_inode = _inode(path)
                if not from_start:
                    fh.seek(0, os.SEEK_END)

            chunk = fh.read(_CHUNK)           # bounded read, never the whole backlog
            if chunk:
                if skipping:
                    nl = chunk.find("\n")
                    if nl < 0:
                        continue
                    chunk, skipping = chunk[nl + 1:], False
                buf += chunk
                *lines, buf = buf.split("\n")
                yield from iter_json_lines(lines, max_line_bytes)
                if len(buf) > max_line_bytes:   # no newline within the limit
                    log.warning("dropping oversized partial line (> %d chars)", max_line_bytes)
                    buf, skipping = "", True
                continue

            # No new data: check for rotation or truncation.
            time.sleep(poll_interval_s)
            disk_inode = _inode(path)
            if disk_inode is not None and disk_inode != cur_inode:
                # File was rotated: reopen the new one from its start.
                fh.close()
                fh = open(path, "r", errors="replace")
                cur_inode = disk_inode
                buf, skipping = "", False
            else:
                try:
                    if os.stat(path).st_size < fh.tell():
                        # Truncated in place: rewind to the new start.
                        fh.seek(0)
                        buf, skipping = "", False
                except FileNotFoundError:
                    fh.close()
                    fh = None
    finally:
        if fh is not None:
            fh.close()
