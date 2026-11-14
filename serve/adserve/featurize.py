"""Map a raw Wazuh alert (dict) to a vocabulary token id.

The ONLY contract that matters here: the log key extracted at inference time
must be defined exactly as it was at training time. The training pipeline
(build_dataset.py) turned each event into a discrete log key and built a
vocabulary from the TRAIN split only. We reuse that vocabulary verbatim and
map anything unseen to <unk>, so the live service and the offline evaluation
see identical token streams.
"""
from __future__ import annotations

from typing import Any, Optional

UNK = "<unk>"
MASK = "<mask>"
PAD = "<pad>"


def dotted_get(obj: dict, path: str) -> Optional[Any]:
    """Fetch obj['a']['b'] for path 'a.b'; None if any hop is missing."""
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


class Vocab:
    """token (str) <-> id (int), loaded from the frozen training vocab."""

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = {str(k): int(v) for k, v in token_to_id.items()}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        if UNK not in self.token_to_id:
            raise ValueError("Vocabulary must contain an <unk> token.")
        self.unk_id = self.token_to_id[UNK]
        self.mask_id = self.token_to_id.get(MASK)
        self.pad_id = self.token_to_id.get(PAD)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, token: str) -> int:
        return self.token_to_id.get(str(token), self.unk_id)


class Featurizer:
    """Turn an alert dict into (host, token_id). Returns None to skip."""

    def __init__(self, vocab: Vocab, log_key_field: str, host_field: str):
        self.vocab = vocab
        self.log_key_field = log_key_field
        self.host_field = host_field

    def __call__(self, alert: dict) -> Optional[tuple[str, int]]:
        raw_key = dotted_get(alert, self.log_key_field)
        if raw_key is None:
            return None  # alert carries no usable log key; ignore it
        host = dotted_get(alert, self.host_field) or "unknown"
        return str(host), self.vocab.encode(str(raw_key))
