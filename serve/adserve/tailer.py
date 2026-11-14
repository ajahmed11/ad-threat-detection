"""Robust JSON-lines tailer for Wazuh alerts.json.

Follows a growing file the way `tail -F` does: survives log rotation (inode
change) and truncation (size shrink), and buffers partial trailing lines until
the newline arrives. Yields one parsed JSON object per complete line.

In replay mode it reads a static file start-to-finish and stops, so the exact
same scoring path can be driven offline for evaluation or a viva demo.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional


def _inode(path: Path) -> Optional[int]:
    try:
        return os.stat(path).st_ino
    except FileNotFoundError:
        return None


def iter_json_lines(line_iter) -> Iterator[dict]:
    for line in line_iter:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # A malformed line never stalls the stream; skip and continue.
            continue


def replay(path: Path) -> Iterator[dict]:
    """Yield every JSON object in a static file, then stop."""
    with open(path, "r", errors="replace") as fh:
        yield from iter_json_lines(fh)


def follow(path: Path, poll_interval_s: float = 0.5, from_start: bool = False) -> Iterator[dict]:
    """Yield JSON objects as they are appended to `path`, forever."""
    path = Path(path)
    buf = ""
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

            chunk = fh.read()
            if chunk:
                buf += chunk
                *lines, buf = buf.split("\n")
                yield from iter_json_lines(lines)
                continue

            # No new data: check for rotation or truncation.
            time.sleep(poll_interval_s)
            disk_inode = _inode(path)
            if disk_inode is not None and disk_inode != cur_inode:
                # File was rotated: reopen the new one from its start.
                fh.close()
                fh = open(path, "r", errors="replace")
                cur_inode = disk_inode
                buf = ""
            else:
                try:
                    if os.stat(path).st_size < fh.tell():
                        # Truncated in place: rewind to the new start.
                        fh.seek(0)
                        buf = ""
                except FileNotFoundError:
                    fh.close()
                    fh = None
    finally:
        if fh is not None:
            fh.close()
