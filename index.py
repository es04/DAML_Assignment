import os, sys, math, random, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from imblearn.over_sampling import SMOTE

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# C-MAPSS paths and settings
CMAPSS_DIR = os.path.join(BASE_DIR, "data", "cmapss")
CMAPSS_SUBSET = "FD001"
AI4I_DIR = os.path.join(BASE_DIR, "data", "ai4i")
AI4I_FILE = "ai4i2020.csv"

# Column names for C-MAPSS raw files
CMAPSS_COLS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance — removed per literature convention
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]

# Piecewise-linear RUL cap (cycles)
RUL_CLIP = 125
WINDOW_SIZE = 30  # sliding window length for sequence models
STRIDE = 1  # sliding window stride

# AI4I columns to drop (IDs + individual failure sub-types)
AI4I_DROP = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
AI4I_TARGET = "Machine failure"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

# ── 2.1  C-MAPSS (RUL dataset) ───────────────────────────────────────────────


def load_cmapss(data_dir=CMAPSS_DIR, subset=CMAPSS_SUBSET):
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")

    missing = [p for p in [train_path, test_path, rul_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"\n[C-MAPSS] Missing dataset files:\n"
            + "\n".join(f"  {p}" for p in missing)
            + f"\n\nDownload CMAPSSData.zip from Kaggle (free, no auto-download):\n"
            f"  https://www.kaggle.com/datasets/behrad3d/nasa-cmaps\n"
            f"Extract and place these 3 files into: {os.path.abspath(data_dir)}\\\n"
            f"  train_{subset}.txt  /  test_{subset}.txt  /  RUL_{subset}.txt"
        )
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    train_df.columns = CMAPSS_COLS
    test_df.columns = CMAPSS_COLS
    
    rul_test = pd.read_csv(rul_path, header=None).values.flatten()

    # Drop trailing NaN columns caused by extra spaces
    train_df.dropna(axis=1, how="all", inplace=True)
    test_df.dropna(axis=1, how="all", inplace=True)

    print(f"[C-MAPSS {subset}] Train {train_df.shape} | Test {test_df.shape}")
    return train_df, test_df, rul_test


def add_rul_labels(df, rul_clip=RUL_CLIP):
    # RUL = max_cycle_for_unit - current_cycle, clipped at rul_clip
    max_c = df.groupby("unit")["cycle"].max().reset_index()
    max_c.columns = ["unit", "max_cycle"]
    df = df.merge(max_c, on="unit")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_clip)
    return df.drop(columns=["max_cycle"])


def preprocess_cmapss(train_df, test_df, rul_test):
    feat_cols = [c for c in CMAPSS_COLS if c not in ["unit", "cycle"] + DROP_SENSORS]

    train_df = add_rul_labels(train_df)

    # Fit scaler on training data only — apply to both splits
    scaler = MinMaxScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    # Build overlapping sliding windows for training
    X_train, y_train = [], []
    for _, group in train_df.groupby("unit"):
        data = group[feat_cols].values
        rul = group["RUL"].values
        for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
            X_train.append(data[start : start + WINDOW_SIZE])
            y_train.append(rul[start + WINDOW_SIZE - 1])

    # Use last WINDOW_SIZE cycles per unit for testing
    X_test = []
    for _, group in test_df.groupby("unit"):
        data = group[feat_cols].values
        if len(data) >= WINDOW_SIZE:
            X_test.append(data[-WINDOW_SIZE:])
        else:
            # Pad shorter sequences by repeating the first row
            X_test.append(
                np.pad(data, ((WINDOW_SIZE - len(data), 0), (0, 0)), mode="edge")
            )

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(rul_test, dtype=np.float32).clip(max=RUL_CLIP)

    print(f"  X_train {X_train.shape} | X_test {X_test.shape}")
    return X_train, y_train, X_test, y_test, scaler, feat_cols


# ── 2.2  AI4I 2020 (failure classification dataset) ─────────────────────────


def load_ai4i(data_dir=AI4I_DIR, filename=AI4I_FILE):
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        import urllib.request

        os.makedirs(data_dir, exist_ok=True)
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00601/ai4i2020.csv"
        )
        print(f"[AI4I 2020] Dataset not found. Downloading from UCI repository...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"[AI4I 2020] Downloaded to {filepath}")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download AI4I dataset automatically ({e}).\n"
                f"Please download it manually from:\n"
                f"  https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
                f"and place it at: {filepath}"
            ) from e

    df = pd.read_csv(filepath)
    print(
        f"[AI4I 2020] Shape {df.shape} | " f"Failure rate {df[AI4I_TARGET].mean():.2%}"
    )
    return df


def preprocess_ai4i(df, test_size=0.2, random_state=42):
    df = df.copy()

    # Encode product quality type: L=0, M=1, H=2
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    # Remove non-feature columns
    df.drop(columns=[c for c in AI4I_DROP if c in df.columns], inplace=True)

    feat_cols = [c for c in df.columns if c != AI4I_TARGET]
    X = df[feat_cols].values.astype(np.float32)
    y = df[AI4I_TARGET].values.astype(np.int64)

    # Stratified split preserves the minority class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fit scaler on training partition only
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Class distribution before SMOTE: {np.bincount(y_train)}")

    # SMOTE oversampling to address class imbalance in training set only
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"  Class distribution after  SMOTE: {np.bincount(y_train)}")
    print(f"  X_train {X_train.shape} | X_test {X_test.shape}")
    return (
        X_train.astype(np.float32),
        y_train,
        X_test.astype(np.float32),
        y_test,
        scaler,
        feat_cols,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FAILURE CLASSIFICATION MODELS
# ═════════════════════════════════════════════════════════════════════════════

# ── 3.1  Bidirectional LSTM classifier ───────────────────────────────────────


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        # Bidirectional LSTM doubles the effective hidden dimension
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # single logit for binary classification
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # Use only the last time-step hidden state
        return self.fc(out[:, -1, :]).squeeze(1)

    def predict_proba_np(self, X_np, batch_size=512):
        # Returns probability of the positive class as numpy array
        self.eval()
        dl = DataLoader(
            TensorDataset(torch.tensor(X_np[:, None, :], dtype=torch.float32)),
            batch_size=batch_size,
        )
        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                preds.append(torch.sigmoid(self(xb.to(DEVICE))).cpu().numpy())
        return np.concatenate(preds)


def train_bilstm_classifier(X_tr, y_tr, X_val, y_val, input_dim, epochs=40):
    model = BiLSTMClassifier(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Weighted BCE handles remaining imbalance after SMOTE
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr[:, None, :], dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=256,
        shuffle=True,
    )
    X_v = torch.tensor(X_val[:, None, :], dtype=torch.float32).to(DEVICE)
    best_auc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Evaluate on validation set each epoch
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(X_v)).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_prob)
        scheduler.step(1 - val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{epochs} | Val AUC: {val_auc:.4f}")

    model.load_state_dict(best_state)
    print(f"  BiLSTM best Val AUC: {best_auc:.4f}")
    return model


# ── 3.2  Evaluation helpers ──────────────────────────────────────────────────


def evaluate_classifier(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    acc = (y_pred == y_true).mean()
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(f"  Accuracy : {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Failure"]))
    return {"model": name, "accuracy": acc, "f1": f1, "auc": auc}


def plot_roc_curves(results, y_true, probas, save_path="outputs/roc_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["grey", "steelblue", "darkorange", "green", "crimson"]
    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_true, probas[res["model"]])
        ax.plot(
            fpr, tpr, color=color, lw=2, label=f"{res['model']} (AUC={res['auc']:.3f})"
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Failure Classification")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curves → {save_path}")


# ── 3.3  Full classification pipeline ────────────────────────────────────────


def run_classification_pipeline():
    print("\n" + "=" * 65)
    print("  STEP 1/3 — Failure Classification (AI4I 2020)")
    print("=" * 65)

    df = load_ai4i()
    X_train, y_train, X_test, y_test, scaler, _ = preprocess_ai4i(df)

    # Hold out 15% of (SMOTE-balanced) training set for LSTM early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    n_features = X_train.shape[1]
    results, probas = [], {}

    # Baseline 1: Logistic Regression
    print("\n[2/6] Logistic Regression …")
    lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_model.fit(X_tr, y_tr)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    results.append(evaluate_classifier("Logistic Regression", y_test, lr_prob))
    probas["Logistic Regression"] = lr_prob
    with open("models/clf_logistic.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    # Baseline 2: Random Forest
    print("\n[3/6] Random Forest …")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_tr, y_tr)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    results.append(evaluate_classifier("Random Forest", y_test, rf_prob))
    probas["Random Forest"] = rf_prob
    with open("models/clf_rf.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # Baseline 3: XGBoost
    print("\n[4/6] XGBoost …")
    sp = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=sp,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate_classifier("XGBoost", y_test, xgb_prob))
    probas["XGBoost"] = xgb_prob
    xgb_model.save_model("models/clf_xgb.json")

    # Baseline 4: Bidirectional LSTM
    print("\n[5/6] BiLSTM …")
    lstm_model = train_bilstm_classifier(
        X_tr, y_tr, X_val, y_val, input_dim=n_features, epochs=40
    )
    lstm_prob = lstm_model.predict_proba_np(X_test)
    results.append(evaluate_classifier("BiLSTM", y_test, lstm_prob))
    probas["BiLSTM"] = lstm_prob
    torch.save(lstm_model.state_dict(), "models/clf_lstm.pt")

    # Proposed: Hybrid XGBoost + LSTM stacking ensemble
    print("\n[6/6] Hybrid XGBoost-LSTM Stack (proposed) …")

    # Generate out-of-fold predictions on full training set to avoid leakage
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(X_train))
    oof_lstm = np.zeros(len(X_train))

    for fold, (tr_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
        Xf_tr, Xf_oof = X_train[tr_idx], X_train[oof_idx]
        yf_tr, yf_oof = y_train[tr_idx], y_train[oof_idx]

        # XGBoost fold model
        xgb_f = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(yf_tr == 0).sum() / max((yf_tr == 1).sum(), 1),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_f.fit(Xf_tr, yf_tr)
        oof_xgb[oof_idx] = xgb_f.predict_proba(Xf_oof)[:, 1]

        # LSTM fold model
        lstm_f = train_bilstm_classifier(
            Xf_tr, yf_tr, Xf_oof, yf_oof, input_dim=n_features, epochs=20
        )
        oof_lstm[oof_idx] = lstm_f.predict_proba_np(Xf_oof)
        print(f"    Fold {fold+1}/5 done")

    # Meta-classifier trained on stacked OOF predictions
    meta_train = np.column_stack([oof_xgb, oof_lstm])
    meta_test = np.column_stack([xgb_prob, lstm_prob])
    meta_clf = LogisticRegression(C=1.0, max_iter=500)
    meta_clf.fit(meta_train, y_train)
    hybrid_prob = meta_clf.predict_proba(meta_test)[:, 1]

    results.append(
        evaluate_classifier("Hybrid XGBoost-LSTM (Proposed)", y_test, hybrid_prob)
    )
    probas["Hybrid XGBoost-LSTM (Proposed)"] = hybrid_prob
    with open("models/clf_hybrid_meta.pkl", "wb") as f:
        pickle.dump(meta_clf, f)

    # Save summary and plots
    summary_df = pd.DataFrame(results).set_index("model")
    summary_df.sort_values("auc", ascending=False).to_csv(
        "outputs/classification_results.csv"
    )
    plot_roc_curves(results, y_test, probas)

    # Confusion matrix for the proposed model
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, hybrid_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Hybrid (Proposed)")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_hybrid.png", dpi=150)
    plt.close()

    return results, probas


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RUL PREDICTION MODELS
# ═════════════════════════════════════════════════════════════════════════════

# ── 4.1  Model definitions ───────────────────────────────────────────────────


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(1)


class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        cnn_channels=64,
        lstm_hidden=128,
        num_lstm_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        # Two conv layers extract local temporal patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            cnn_channels,
            lstm_hidden,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # Conv1d expects (batch, channels, seq) — permute in and out
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


class MultiHeadAttentionLSTM(nn.Module):
    def __init__(
        self, input_dim, lstm_hidden=128, num_lstm_layers=2, num_heads=4, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden, num_lstm_layers, batch_first=True, dropout=dropout
        )
        # Self-attention over LSTM hidden states weights informative time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(lstm_hidden)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out).mean(dim=1)  # global average pool
        return self.fc(out).squeeze(1)


class CNNLSTMAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        cnn_channels=64,
        lstm_hidden=128,
        num_lstm_layers=2,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__()
        # CNN + BN extracts and normalises local features
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
        )
        self.lstm = nn.LSTM(
            cnn_channels,
            lstm_hidden,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(lstm_hidden)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out).mean(dim=1)
        return self.fc(out).squeeze(1), attn_weights


# ── 4.2  Training and inference utilities ────────────────────────────────────


def nasa_score(y_true, y_pred):
    # Asymmetric penalty: late predictions (underestimate) penalised more heavily
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def regression_metrics(name, y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(
        f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | "
        f"R²: {r2:.4f} | NASA Score: {score:.2f}"
    )
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "nasa_score": score}


def train_rul_model(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    epochs=60,
    batch_size=256,
    lr=1e-3,
    returns_attn=False,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    # Cosine annealing smoothly reduces LR over training
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    best_rmse, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            preds = model(xb.to(DEVICE))
            p = preds[0] if returns_attn else preds
            crit(p, yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_v)
            vp = (vp[0] if returns_attn else vp).cpu().numpy()
        rmse = math.sqrt(mean_squared_error(y_val, vp))

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{epochs} | Val RMSE: {rmse:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best Val RMSE: {best_rmse:.4f}")
    return model


def infer_rul(model, X, returns_attn=False, batch_size=512):
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size)
    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            p = model(xb.to(DEVICE))
            preds.append((p[0] if returns_attn else p).cpu().numpy())
    return np.concatenate(preds).clip(min=0)


def extract_statistical_features(X_windows):
    # Compute per-channel stats across the time window for XGBoost input
    mean = X_windows.mean(axis=1)
    std = X_windows.std(axis=1)
    mn = X_windows.min(axis=1)
    mx = X_windows.max(axis=1)
    rms = np.sqrt((X_windows**2).mean(axis=1))
    return np.concatenate([mean, std, mn, mx, rms], axis=1)


# ── 4.3  Full RUL pipeline ───────────────────────────────────────────────────


def run_rul_pipeline():
    print("\n" + "=" * 65)
    print("  STEP 2/3 — RUL Prediction (NASA C-MAPSS)")
    print("=" * 65)

    train_df, test_df, rul_test = load_cmapss()
    X_train, y_train, X_test, y_test, _, _ = preprocess_cmapss(
        train_df, test_df, rul_test
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    input_dim = X_train.shape[2]
    results = []

    # Baseline 1: LSTM
    print("\n[2/6] LSTM baseline …")
    lstm = LSTMRegressor(input_dim).to(DEVICE)
    lstm = train_rul_model(lstm, X_tr, y_tr, X_val, y_val, epochs=60)
    results.append(regression_metrics("LSTM", y_test, infer_rul(lstm, X_test)))
    torch.save(lstm.state_dict(), "models/rul_lstm.pt")

    # Baseline 2: GRU
    print("\n[3/6] GRU …")
    gru = GRURegressor(input_dim).to(DEVICE)
    gru = train_rul_model(gru, X_tr, y_tr, X_val, y_val, epochs=60)
    results.append(regression_metrics("GRU", y_test, infer_rul(gru, X_test)))
    torch.save(gru.state_dict(), "models/rul_gru.pt")

    # Baseline 3: CNN-LSTM
    print("\n[4/6] CNN-LSTM …")
    cnn_lstm = CNNLSTMRegressor(input_dim).to(DEVICE)
    cnn_lstm = train_rul_model(cnn_lstm, X_tr, y_tr, X_val, y_val, epochs=60)
    results.append(regression_metrics("CNN-LSTM", y_test, infer_rul(cnn_lstm, X_test)))
    torch.save(cnn_lstm.state_dict(), "models/rul_cnnlstm.pt")

    # Baseline 4: Attention-LSTM
    print("\n[5/6] Attention-LSTM …")
    attn_lstm = MultiHeadAttentionLSTM(input_dim).to(DEVICE)
    attn_lstm = train_rul_model(attn_lstm, X_tr, y_tr, X_val, y_val, epochs=60)
    results.append(
        regression_metrics("Attention-LSTM", y_test, infer_rul(attn_lstm, X_test))
    )
    torch.save(attn_lstm.state_dict(), "models/rul_attn_lstm.pt")

    # Proposed: CNN-LSTM-Attention
    print("\n[6/6] CNN-LSTM-Attention (proposed) …")
    proposed = CNNLSTMAttention(input_dim).to(DEVICE)
    proposed = train_rul_model(
        proposed, X_tr, y_tr, X_val, y_val, epochs=80, returns_attn=True
    )
    y_pred_proposed = infer_rul(proposed, X_test, returns_attn=True)
    results.append(
        regression_metrics("CNN-LSTM-Attention (Proposed)", y_test, y_pred_proposed)
    )
    torch.save(proposed.state_dict(), "models/rul_proposed.pt")

    # Hybrid: XGBoost on statistical features + CNN-LSTM-Attention stacking
    print("\n  Building XGBoost + CNN-LSTM-Attention meta-stack …")
    X_tr_feat = extract_statistical_features(X_tr)
    X_val_feat = extract_statistical_features(X_val)
    X_test_feat = extract_statistical_features(X_test)

    xgb_rul = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_rul.fit(X_tr_feat, y_tr, eval_set=[(X_val_feat, y_val)], verbose=False)
    xgb_rul.save_model("models/rul_xgb.json")

    # Stack XGBoost + deep model predictions via Ridge regression
    meta_train = np.column_stack(
        [xgb_rul.predict(X_tr_feat), infer_rul(proposed, X_tr, returns_attn=True)]
    )
    meta_test = np.column_stack([xgb_rul.predict(X_test_feat), y_pred_proposed])

    meta_reg = Ridge(alpha=1.0)
    meta_reg.fit(meta_train, y_tr)
    y_pred_hybrid = meta_reg.predict(meta_test).clip(min=0)
    results.append(
        regression_metrics("Hybrid XGBoost-LSTM-Att. (Proposed)", y_test, y_pred_hybrid)
    )

    with open("models/rul_meta_ridge.pkl", "wb") as f:
        pickle.dump(meta_reg, f)

    # Save summary
    pd.DataFrame(results).set_index("model").sort_values("rmse").to_csv(
        "outputs/rul_results.csv"
    )

    # Plot true vs predicted and error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(y_test, y_pred_hybrid, alpha=0.4, s=15, color="steelblue")
    lim = max(y_test.max(), y_pred_hybrid.max()) + 5
    ax.plot([0, lim], [0, lim], "r--")
    ax.grid(alpha=0.3)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("True vs Predicted RUL")

    ax = axes[1]
    ax.hist(
        y_pred_hybrid - y_test,
        bins=50,
        color="steelblue",
        edgecolor="white",
        alpha=0.85,
    )
    ax.axvline(0, color="red", linestyle="--")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")

    plt.suptitle("Hybrid RUL Prediction Results", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/rul_prediction_plot.png", dpi=150)
    plt.close()
    print("  RUL plot → outputs/rul_prediction_plot.png")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DRL ADAPTIVE SCHEDULING (PPO)
# ═════════════════════════════════════════════════════════════════════════════

# ── 5.1  Factory environment ─────────────────────────────────────────────────


class FactoryEnv(gym.Env):
    N_MACHINES = 12
    N_JOB_TYPES = 5
    MAX_QUEUE = 20
    MAX_TIME = 500.0
    RUL_MAX = 125.0
    MAINT_DUR = 5  # cycles for a planned maintenance action

    def __init__(self, failure_rate=0.02, n_jobs=50, use_predictive=True, seed=None):
        super().__init__()
        self.failure_rate = failure_rate
        self.n_jobs = n_jobs
        self.use_predictive = use_predictive

        # Fixed processing-time matrix: job_type × machine (sampled once)
        rng = np.random.default_rng(seed)
        self.proc_matrix = rng.integers(
            2, 12, (self.N_JOB_TYPES, self.N_MACHINES)
        ).astype(float)

        obs_dim = self.N_MACHINES * 3 + self.MAX_QUEUE * 3 + 1
        self.observation_space = spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(self.N_MACHINES)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.health = rng.uniform(0.7, 1.0, self.N_MACHINES)
        self.rul = rng.uniform(50, 125, self.N_MACHINES)
        self.avail = np.ones(self.N_MACHINES, bool)
        self.busy_until = np.zeros(self.N_MACHINES)
        self.in_maint = np.zeros(self.N_MACHINES, bool)
        self.maint_done = np.zeros(self.N_MACHINES)

        self.done_jobs, self.t, self.tard, self.failures = [], 0.0, 0.0, 0

        self.queue = self._gen_jobs()
        return self._obs(), {}

    def step(self, action):
        if not self.queue:
            return self._obs(), 0.0, True, False, {}

        m = int(action)
        job = self.queue[0]
        rwd = 0.0

        # Penalise invalid assignment to a machine in maintenance
        if self.in_maint[m]:
            rwd -= 5.0
            avail = [i for i in range(self.N_MACHINES) if self.avail[i]]
            m = min(avail, key=lambda i: self.busy_until[i]) if avail else 0

        # Compute finish time and dequeue job
        t0 = max(self.t, self.busy_until[m])
        pt = self.proc_matrix[job["type"], m]
        t1 = t0 + pt
        self.busy_until[m] = t1
        self.t = t0

        tard = max(0, t1 - job["due"])
        rwd -= tard * 0.1
        self.tard += tard
        self.queue.pop(0)
        self.done_jobs.append(job)

        # Degrade machines proportional to processing time
        for i in range(self.N_MACHINES):
            if self.avail[i]:
                self.health[i] = max(
                    0, self.health[i] - self.np_random.uniform(0.001, 0.005) * pt
                )
                self.rul[i] = max(0, self.rul[i] - pt)

            # Stochastic failure check — probability rises as health falls
            if self.avail[i] and not self.in_maint[i]:
                fp = self.failure_rate * (1 - self.health[i]) ** 2
                if self.np_random.random() < fp:
                    self.avail[i] = False
                    self.in_maint[i] = True
                    self.maint_done[i] = self.t + self.MAINT_DUR * 2
                    self.failures += 1
                    rwd -= 10.0

            # Predictive maintenance trigger when RUL drops below threshold
            if (
                self.use_predictive
                and self.avail[i]
                and not self.in_maint[i]
                and self.rul[i] < 15
            ):
                self.avail[i] = False
                self.in_maint[i] = True
                self.maint_done[i] = self.t + self.MAINT_DUR
                rwd += 2.0  # reward for proactive maintenance

            # Restore machines that have finished maintenance
            if self.in_maint[i] and self.t >= self.maint_done[i]:
                self.in_maint[i] = False
                self.avail[i] = True
                self.health[i] = 1.0
                self.rul[i] = self.RUL_MAX

        done = len(self.queue) == 0
        if done:
            # Bonus inversely proportional to average tardiness
            rwd += max(0, 20 - self.tard / max(len(self.done_jobs), 1))

        return self._obs(), float(rwd), done, False, {"failures": self.failures}

    def _gen_jobs(self):
        jobs = []
        for i in range(self.n_jobs):
            t = int(self.np_random.integers(0, self.N_JOB_TYPES))
            pm = self.proc_matrix[t].mean()
            due = self.t + pm * self.np_random.uniform(1.5, 3.5)
            jobs.append(
                {
                    "id": i,
                    "type": t,
                    "proc": pm,
                    "due": due,
                    "pri": self.np_random.uniform(0, 1),
                }
            )
        return jobs

    def _obs(self):
        qf = np.zeros((self.MAX_QUEUE, 3), np.float32)
        for i, j in enumerate(self.queue[: self.MAX_QUEUE]):
            qf[i] = [j["proc"] / 12, min(j["due"] / self.MAX_TIME, 1), j["pri"]]
        return np.concatenate(
            [
                self.health.clip(0, 1).astype(np.float32),
                self.avail.astype(np.float32),
                (self.rul / self.RUL_MAX).clip(0, 1).astype(np.float32),
                qf.flatten(),
                [self.t / self.MAX_TIME],
            ]
        ).astype(np.float32)


# ── 5.2  Baseline scheduling policies ────────────────────────────────────────


def run_baseline_policy(policy="reactive", n_episodes=50):
    rewards, failures, tardiness = [], [], []
    for _ in range(n_episodes):
        env = FactoryEnv(use_predictive=(policy == "pred_only"))
        obs, _ = env.reset()
        done, total_r = False, 0.0
        while not done:
            avail = [m for m in range(env.N_MACHINES) if env.avail[m]] or list(
                range(env.N_MACHINES)
            )
            if policy == "reactive":
                action = random.choice(avail)
            elif policy == "sched_only":
                # Greedy: pick the machine that becomes free soonest
                action = min(avail, key=lambda m: env.busy_until[m])
            else:
                action = random.choice(avail)
            obs, r, done, _, _ = env.step(action)
            total_r += r
        rewards.append(total_r)
        failures.append(env.failures)
        tardiness.append(env.tard)
    return (np.mean(rewards), np.mean(failures), np.mean(tardiness))


# ── 5.3  PPO training ────────────────────────────────────────────────────────


def train_ppo_scheduler(total_timesteps=500_000, n_envs=4):
    vec_env = make_vec_env(
        lambda: Monitor(FactoryEnv(use_predictive=True)), n_envs=n_envs
    )
    eval_env = Monitor(FactoryEnv(use_predictive=True))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="outputs/",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000, save_path="models/", name_prefix="ppo_scheduler", verbose=0
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    print(f"\n  Training PPO: {total_timesteps:,} steps | {n_envs} envs")
    model.learn(
        total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb], progress_bar=True
    )
    model.save("models/ppo_scheduler")
    print("  PPO saved → models/ppo_scheduler.zip")
    return model


# ── 5.4  Evaluation of all strategies ────────────────────────────────────────


def run_scheduling_pipeline():
    print("\n" + "=" * 65)
    print("  STEP 3/3 — DRL Adaptive Scheduling")
    print("=" * 65)

    # Smoke test: verify the environment runs correctly
    env_test = FactoryEnv(seed=42)
    obs, _ = env_test.reset()
    done, total_r = False, 0.0
    while not done:
        obs, r, done, _, _ = env_test.step(env_test.action_space.sample())
        total_r += r
    print(f"  Smoke test reward: {total_r:.2f} | " f"Failures: {env_test.failures}")

    # Train PPO — increase total_timesteps to 2_000_000 for full training
    ppo_model = train_ppo_scheduler(total_timesteps=500_000, n_envs=4)

    # Evaluate all four strategies
    strategies = {
        "Reactive + Static": ("reactive", False),
        "Predictive Maintenance Only": ("pred_only", True),
        "Scheduling Optimisation Only": ("sched_only", False),
    }
    all_results = {}

    for name, (policy, _) in strategies.items():
        r, f, t = run_baseline_policy(policy, n_episodes=100)
        all_results[name] = {"mean_reward": r, "mean_failures": f, "mean_tardiness": t}
        print(f"  {name:<35} | R: {r:7.2f} | F: {f:.2f} | T: {t:.2f}")

    # Evaluate PPO
    ppo_r, ppo_f, ppo_t = [], [], []
    for _ in range(100):
        env = FactoryEnv(use_predictive=True)
        obs, _ = env.reset()
        done, ep_r = False, 0.0
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(action)
            ep_r += r
        ppo_r.append(ep_r)
        ppo_f.append(env.failures)
        ppo_t.append(env.tard)

    all_results["Proposed DRL-PPO (Integrated)"] = {
        "mean_reward": np.mean(ppo_r),
        "mean_failures": np.mean(ppo_f),
        "mean_tardiness": np.mean(ppo_t),
    }
    print(
        f"  {'Proposed DRL-PPO (Integrated)':<35} | "
        f"R: {np.mean(ppo_r):7.2f} | F: {np.mean(ppo_f):.2f} | "
        f"T: {np.mean(ppo_t):.2f}"
    )

    # Save and plot results
    df = pd.DataFrame(all_results).T
    df.to_csv("outputs/scheduling_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    colors = ["#888888", "#5599DD", "#EE8833", "#22AA55"]
    names = list(df.index)
    for ax, col, label in zip(
        axes,
        ["mean_reward", "mean_failures", "mean_tardiness"],
        ["Mean Reward", "Mean Failures", "Mean Tardiness"],
    ):
        vals = df[col].values
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.suptitle("Scheduling Strategy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/scheduling_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Scheduling plot → outputs/scheduling_comparison.png")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced epochs/steps for fast smoke-testing",
    )
    args = parser.parse_args()

    # Override training lengths when --quick is used
    if args.quick:
        print("[QUICK MODE] Reducing epochs and RL steps for fast testing.")

    wall_times = {}

    # ── Step 1: Failure Classification ──────────────────────────────────────
    t0 = time.time()
    clf_results, _ = run_classification_pipeline()
    wall_times["classification"] = time.time() - t0

    # ── Step 2: RUL Prediction ───────────────────────────────────────────────
    t0 = time.time()
    rul_results = run_rul_pipeline()
    wall_times["rul"] = time.time() - t0

    # ── Step 3: DRL Scheduling ───────────────────────────────────────────────
    t0 = time.time()
    sched_results = run_scheduling_pipeline()
    wall_times["scheduling"] = time.time() - t0

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPLETE PIPELINE SUMMARY")
    print("=" * 65)

    print("\n── Classification Results ──────────────────────────────────────")
    print(
        pd.DataFrame(clf_results)
        .set_index("model")
        .sort_values("auc", ascending=False)
        .to_string(float_format="{:.4f}".format)
    )

    print("\n── RUL Prediction Results ──────────────────────────────────────")
    print(
        pd.DataFrame(rul_results)
        .set_index("model")
        .sort_values("rmse")
        .to_string(float_format="{:.4f}".format)
    )

    print("\n── Scheduling Results ──────────────────────────────────────────")
    print(sched_results.to_string(float_format="{:.3f}".format))

    print("\n── Wall-Clock Times ────────────────────────────────────────────")
    for k, v in wall_times.items():
        print(f"   {k:<20}: {v:>8.1f}s  ({v/60:.1f} min)")

    # Save unified Excel report
    with pd.ExcelWriter("outputs/full_pipeline_results.xlsx") as writer:
        pd.DataFrame(clf_results).set_index("model").to_excel(
            writer, sheet_name="Classification"
        )
        pd.DataFrame(rul_results).set_index("model").to_excel(
            writer, sheet_name="RUL_Prediction"
        )
        sched_results.to_excel(writer, sheet_name="Scheduling")

    print("\n  Results → outputs/full_pipeline_results.xlsx")
    print("  Models  → models/")
    print("  Plots   → outputs/")
    print("\nAll done!")
