"""The three detectors, all framed as masked log-key predictors.

* MaskedRNN(bidirectional=False) -> unidirectional LSTM  (RQ2 baseline)
* MaskedRNN(bidirectional=True)  -> BiLSTM                (RQ2: effect of directionality)
* LogBERT                        -> Transformer encoder + hypersphere term (RQ3)

Each returns per-position logits over the vocabulary. LogBERT additionally exposes a
[DIST] embedding whose distance to a fixed centre is the volume-minimising (VHM) score.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from . import config


class MaskedRNN(nn.Module):
    def __init__(self, vocab_size, bidirectional, embed_dim=config.EMBED_DIM,
                 hidden=config.HIDDEN, layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)  # +1 for <mask>
        self.rnn = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, vocab_size + 1)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h)                       # B x W x V


class _PosEnc(nn.Module):
    def __init__(self, dim, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, dim)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class LogBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=config.EMBED_DIM, heads=config.N_HEADS,
                 layers=config.TRANSFORMER_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.dist_tok = nn.Parameter(torch.randn(1, 1, embed_dim))   # [DIST] token
        self.pos = _PosEnc(embed_dim)
        enc = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4, dropout,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(embed_dim, vocab_size + 1)
        self.register_buffer("center", torch.zeros(embed_dim))       # fixed hypersphere centre

    def forward(self, x):
        b = x.size(0)
        tok = self.emb(x)
        dist = self.dist_tok.expand(b, -1, -1)
        h = self.pos(torch.cat([dist, tok], dim=1))
        h = self.enc(h)
        dist_h, seq_h = h[:, 0], h[:, 1:]
        logits = self.head(seq_h)                 # B x W x V
        return logits, dist_h                     # dist_h used for the VHM term


def build(name, vocab_size):
    if name == "lstm":
        return MaskedRNN(vocab_size, bidirectional=False)
    if name == "bilstm":
        return MaskedRNN(vocab_size, bidirectional=True)
    if name == "logbert":
        return LogBERT(vocab_size)
    raise ValueError(name)


class SupervisedRNN(nn.Module):
    """Supervised binary classifier (benign vs malicious window), as the proposal commits to.
    Embedding -> LSTM/BiLSTM -> mean-pool -> linear head -> one logit. Trained with
    class-weighted BCE. bidirectional=True gives the BiLSTM variant (isolates RQ2)."""
    def __init__(self, vocab_size, bidirectional, embed_dim=config.EMBED_DIM,
                 hidden=config.HIDDEN, layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        pooled = h.mean(dim=1)                     # mean-pool over the window
        return self.head(pooled).squeeze(-1)       # (B,) logits
