from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def evaluate_lstm_models(
    df: pd.DataFrame,
    feature_columns: list[str],
    regression_target: str = "change_tomorrow",
    classification_target: str = "change_tomorrow_direction",
    seq_len: int = 20,
    epochs: int = 10,
) -> pd.DataFrame:
    if not TORCH_AVAILABLE:
        return pd.DataFrame(
            [{"model_family": "deep_learning", "model_name": "lstm", "status": "torch_not_installed"}]
        )

    X = df[feature_columns].values.astype(np.float32)
    y_reg = df[regression_target].values.astype(np.float32)
    y_clf = df[classification_target].values.astype(np.float32)

    X_seq_reg, y_seq_reg = make_sequences(X, y_reg, seq_len=seq_len)
    X_seq_clf, y_seq_clf = make_sequences(X, y_clf, seq_len=seq_len)

    rows = []

    if len(X_seq_reg) > 50:
        split = int(len(X_seq_reg) * 0.8)
        X_train = torch.tensor(X_seq_reg[:split])
        y_train = torch.tensor(y_seq_reg[:split]).view(-1, 1)
        X_test = torch.tensor(X_seq_reg[split:])
        y_test = y_seq_reg[split:]

        model = LSTMRegressor(input_size=X_seq_reg.shape[2])
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        ds = TensorDataset(X_train, y_train)
        dl = DataLoader(ds, batch_size=32, shuffle=False)

        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_test).numpy().reshape(-1)

        rows.append(
            {
                "model_family": "deep_learning",
                "model_name": "lstm_regressor",
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "mae": float(mean_absolute_error(y_test, pred)),
            }
        )

    if len(X_seq_clf) > 50:
        split = int(len(X_seq_clf) * 0.8)
        X_train = torch.tensor(X_seq_clf[:split])
        y_train = torch.tensor(y_seq_clf[:split]).view(-1, 1)
        X_test = torch.tensor(X_seq_clf[split:])
        y_test = y_seq_clf[split:]

        model = LSTMClassifier(input_size=X_seq_clf.shape[2])
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        ds = TensorDataset(X_train, y_train)
        dl = DataLoader(ds, batch_size=32, shuffle=False)

        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test).numpy().reshape(-1)
            prob = 1 / (1 + np.exp(-logits))
            pred = (prob >= 0.5).astype(int)

        rows.append(
            {
                "model_family": "deep_learning",
                "model_name": "lstm_classifier",
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)
