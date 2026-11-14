"""
parse_archives.py - flatten Wazuh archive files into one tidy event table.

Reads the archive files listed in config.ARCHIVE_FILES. Robust to the two things
that broke earlier attempts:
  * null bytes in the files (stripped per line),
  * gzip (.gz) archives (handled transparently),
  * one-JSON-object-per-line format read line by line, bad lines skipped.

Keeps only real Windows Security / Sysmon events (those carrying data.win) and pulls
the fields needed for the log key and for labelling. Writes data/processed/events.parquet.

Run:  python src/parse_archives.py
"""
from __future__ import annotations
import gzip, io, json
import pandas as pd
import config


def _open_any(path):
    p = str(path)
    if p.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8", errors="ignore")
    return open(p, encoding="utf-8", errors="ignore")


def _get(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def parse_one(path):
    rows = []
    with _open_any(path) as f:
        for line in f:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            win = _get(o, "data", "win")
            if not win:
                continue  # not a Windows event (syslog/SCA/etc.) -> no event_id
            sysd = win.get("system", {}) or {}
            ed = win.get("eventdata", {}) or {}
            eid = sysd.get("eventID")
            if eid is None:
                continue
            rows.append({
                "timestamp":    o.get("timestamp"),
                "agent":        _get(o, "agent", "name", default="?"),
                "event_id":     str(eid),
                "logon_type":   ed.get("logonType"),
                "ticket_enc":   ed.get("ticketEncryptionType"),
                "preauth":      ed.get("preAuthType"),
                "status":       ed.get("status"),
                "target_user":  ed.get("targetUserName"),
                "subject_user": ed.get("subjectUserName"),
                "service":      ed.get("serviceName"),
                "src_ip":       ed.get("ipAddress"),
            })
    return rows


def main():
    all_rows = []
    for path in config.ARCHIVE_FILES:
        if not path.exists():
            raise SystemExit(f"Missing archive file: {path}\n"
                             f"Check config.RAW_DIR and the archives_clean folder.")
        n0 = len(all_rows)
        all_rows.extend(parse_one(path))
        print(f"[parse] {path.name}: +{len(all_rows) - n0} windows events")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No Windows events parsed - check the archive files.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(config.EVENTS_PARQUET, index=False)
    print(f"[parse] wrote {len(df):,} events -> {config.EVENTS_PARQUET}")
    print(f"[parse] span {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print("[parse] top event IDs:\n", df["event_id"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
