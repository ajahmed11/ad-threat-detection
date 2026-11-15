"""
Central configuration for the AD identity-attack ML dataset pipeline.
Every path and experiment parameter lives here so the dissertation can cite exact values.

Set the raw-data location with the environment variable ADPIPE_RAW_DIR (no source
edits needed), e.g.   $env:ADPIPE_RAW_DIR = "D:\\Final_Year\\CSP\\raw"   (PowerShell)
or   export ADPIPE_RAW_DIR=/data/raw   (bash). The raw archives are NOT in git; see
docs/REPRODUCING.md for how to obtain them.
"""
import os
from pathlib import Path

RAW_DIR = Path(os.environ.get("ADPIPE_RAW_DIR", "./raw")).expanduser()

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Inputs (relative to RAW_DIR) ---------------------------------------
# Only the two days that carry the frozen clean attacks (Nov 9 = 89 runs, Oct 22 = 5 runs).
# Other archive days are deliberately excluded: they hold UNLABELLED attacks
# (Oct 12/17 originals, Nov 6-7 CVE) that would contaminate the benign class.
ARCHIVE_FILES = [
    RAW_DIR / "archives_clean" / "2026-Nov-ossec-archive-09.json",
    RAW_DIR / "archives_clean" / "2026-Oct-ossec-archive-22.json",
]
FROZEN_RUNS = RAW_DIR / "runs_frozen_20261114.jsonl"   # positive labels (verified-clean)
ALL_RUNS    = RAW_DIR / "runs_all.jsonl"               # mask: every run (clean + failed)

# --- Outputs ------------------------------------------------------------
EVENTS_PARQUET   = OUT_DIR / "events.parquet"
LABELLED_PARQUET = OUT_DIR / "labelled_events.parquet"
DATASET_NPZ      = OUT_DIR / "windows.npz"
VOCAB_JSON       = OUT_DIR / "vocab.json"
REPORT_JSON      = OUT_DIR / "dataset_report.json"

# --- Labelling ----------------------------------------------------------
PAD_SECONDS = 5           # padding on each side of a run window (matches rule scoring)

# --- Windowing ----------------------------------------------------------
WINDOW_LEN = 20           # events per sequence window
WINDOW_STRIDE = 10        # step between consecutive windows
MAX_GAP_SECONDS = 300     # do not build a window across a gap longer than this (per host)

# --- Split --------------------------------------------------------------
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 1234

# --- Log key ------------------------------------------------------------
# Fields combined (with the Event ID) into the discrete log key the models consume.
# Kept small and categorical so the vocabulary stays compact.
KEY_FIELDS = ["event_id", "logon_type", "ticket_enc", "preauth", "status_class"]
UNK = "<unk>"
PAD = "<pad>"
