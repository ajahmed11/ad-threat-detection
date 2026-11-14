"""Hyperparameters and paths for the AD anomaly-detection models.
All three models (LSTM, BiLSTM, LogBERT) are trained on BENIGN windows only and
score anomalies by masked log-key prediction error. Attacks are used only at evaluation.
"""
from pathlib import Path
import torch

# --- paths: point at the dataset the adpipe pipeline produced ---------------
# Default assumes admodel sits next to adpipe. Edit if your layout differs.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "adpipe" / "data" / "processed"
WINDOWS_NPZ = DATA_DIR / "windows.npz"
VOCAB_JSON  = DATA_DIR / "vocab.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- shared ---------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]              # repeats for mean +/- std
EMBED_DIM = 64
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.1
MASK_PROB = 0.15              # fraction of positions masked during training
TOPG = 5                     # a key is "normal" if it is in the model's top-g predictions
BATCH = 64
EPOCHS = 60
PATIENCE = 6                 # early stopping on benign validation loss
LR = 1e-3
WEIGHT_DECAY = 1e-5

# --- LogBERT specific -----------------------------------------------------
N_HEADS = 4
TRANSFORMER_LAYERS = 2
VHM_WEIGHT = 0.1             # weight of the hypersphere (volume-minimising) term

# --- evaluation -----------------------------------------------------------
# threshold is chosen on the validation set to maximise F1 (reported alongside a
# fixed low-FPR operating point).
VAL_TARGET = "f1"
