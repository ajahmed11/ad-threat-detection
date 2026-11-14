"""Map a raw Wazuh alert (dict) to a vocabulary token id.

The one contract that matters: the log key produced here at inference time must
be IDENTICAL to the one the training pipeline (adpipe/build_dataset.py) produced,
or the model sees <unk> noise.

This project does NOT key on the Wazuh rule id. It builds a 5-part composite
key from Windows event fields, exactly mirroring adpipe:

    key = "event_id|logon_type|ticket_enc|preauth|status_class"

where status_class collapses the raw status code to ok / fail / other / na and
every missing field becomes "na". The field list and the JSON paths come from
the bundle's meta.json, so this stays driven by what was trained, not hard-coded.

The vocabulary is the frozen training vocab (list form, index = id), built from
the TRAIN split only; unseen keys map to <unk>.
"""
from __future__ import annotations

from typing import Any, Optional

UNK = "<unk>"
PAD = "<pad>"


def dotted_get(obj: dict, path: str) -> Optional[Any]:
    """Fetch obj['a']['b'] for path 'a.b'; None if any hop is missing."""
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def status_class(s) -> str:
    """Mirror adpipe build_dataset._status_class exactly."""
    if s is None:
        return "na"
    if isinstance(s, float):
        # NaN check without importing numpy
        if s != s:
            return "na"
    s = str(s).lower()
    if s in ("0x0", "0"):
        return "ok"
    if s.startswith("0x"):
        return "fail"
    return "other"


class Vocab:
    """token (str) <-> id (int). Accepts the training vocab as a LIST
    (index = id, the form build_dataset.py writes) or a dict {token: id}."""

    def __init__(self, vocab):
        if isinstance(vocab, list):
            self.token_to_id = {str(t): i for i, t in enumerate(vocab)}
        elif isinstance(vocab, dict):
            self.token_to_id = {str(k): int(v) for k, v in vocab.items()}
        else:
            raise TypeError("vocab must be a list or a dict")
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        if UNK not in self.token_to_id:
            raise ValueError("Vocabulary must contain an <unk> token.")
        self.unk_id = self.token_to_id[UNK]
        self.pad_id = self.token_to_id.get(PAD, 0)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, token: str) -> int:
        return self.token_to_id.get(str(token), self.unk_id)


# Default key spec = the adpipe pipeline's definition. meta.json overrides it.
DEFAULT_KEY_FIELDS = ["event_id", "logon_type", "ticket_enc", "preauth", "status_class"]
DEFAULT_FIELD_PATHS = {
    "event_id":   "data.win.system.eventID",
    "logon_type": "data.win.eventdata.logonType",
    "ticket_enc": "data.win.eventdata.ticketEncryptionType",
    "preauth":    "data.win.eventdata.preAuthType",
    "status":     "data.win.eventdata.status",
}
DEFAULT_REQUIRE_PATH = "data.win"   # skip alerts that are not Windows events
DEFAULT_HOST_FIELD = "agent.name"


class Featurizer:
    """Turn an alert dict into (host, token_id). Returns None to skip the alert."""

    def __init__(self, vocab: Vocab, *, key_fields=None, field_paths=None,
                 require_path: Optional[str] = DEFAULT_REQUIRE_PATH,
                 host_field: str = DEFAULT_HOST_FIELD):
        self.vocab = vocab
        self.key_fields = list(key_fields or DEFAULT_KEY_FIELDS)
        self.field_paths = dict(field_paths or DEFAULT_FIELD_PATHS)
        self.require_path = require_path
        self.host_field = host_field

    @classmethod
    def from_meta(cls, vocab: Vocab, meta: dict) -> "Featurizer":
        fz = meta.get("featurizer", {})
        return cls(
            vocab,
            key_fields=fz.get("key_fields"),
            field_paths=fz.get("field_paths"),
            require_path=fz.get("require_path", DEFAULT_REQUIRE_PATH),
            host_field=fz.get("host_field", DEFAULT_HOST_FIELD),
        )

    def make_log_key(self, alert: dict) -> str:
        parts = []
        for f in self.key_fields:
            if f == "status_class":
                parts.append(status_class(dotted_get(alert, self.field_paths.get("status", ""))))
            else:
                v = dotted_get(alert, self.field_paths.get(f, ""))
                parts.append("na" if v is None else str(v))
        return "|".join(parts)

    def __call__(self, alert: dict) -> Optional[tuple[str, int]]:
        # Only Windows events carry data.win; everything else had no event_id at
        # training time and was dropped, so we drop it here too.
        if self.require_path and dotted_get(alert, self.require_path) is None:
            return None
        if dotted_get(alert, self.field_paths.get("event_id", "")) is None:
            return None
        host = dotted_get(alert, self.host_field) or "unknown"
        key = self.make_log_key(alert)
        return str(host), self.vocab.encode(key)
