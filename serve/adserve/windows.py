"""Per-host sliding windows over the live token stream.

Each host gets its own fixed-length deque. Every new event slides the window
forward by one; once a window is full it is eligible to be scored. Idle hosts
are evicted so memory stays bounded on a long-running daemon.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _HostState:
    buf: deque = field(default_factory=deque)
    last_seen: float = field(default_factory=time.monotonic)


class WindowStore:
    def __init__(self, window_size: int, idle_evict_s: float, max_hosts: int = 512):
        self.window_size = int(window_size)
        self.idle_evict_s = float(idle_evict_s)
        self.max_hosts = int(max_hosts)     # hard cap: hostile agent names cannot grow memory
        self._hosts: dict[str, _HostState] = {}

    def push(self, host: str, token_id: int) -> list[int] | None:
        """Add one token for `host`. Return the full window if one is ready."""
        st = self._hosts.get(host)
        if st is None:
            if len(self._hosts) >= self.max_hosts:      # evict least-recently-seen
                oldest = min(self._hosts, key=lambda h: self._hosts[h].last_seen)
                del self._hosts[oldest]
            st = _HostState(buf=deque(maxlen=self.window_size))
            self._hosts[host] = st
        st.buf.append(token_id)
        st.last_seen = time.monotonic()
        if len(st.buf) == self.window_size:
            return list(st.buf)
        return None

    def evict_idle(self) -> int:
        """Drop hosts unseen for longer than idle_evict_s. Returns count."""
        now = time.monotonic()
        stale = [h for h, s in self._hosts.items() if now - s.last_seen > self.idle_evict_s]
        for h in stale:
            del self._hosts[h]
        return len(stale)

    @property
    def active_hosts(self) -> int:
        return len(self._hosts)
