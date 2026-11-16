"""Model settings and fold fitting from the archived common-scale analysis."""

import hashlib

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


SEED = 123
MODELS = ("RandomForest", "XGBoost", "LightGBM", "1D-CNN")
EVENT_COLUMNS = ["subject_id", "task_id", "interval_index"]
WEARABLE_FEATURES = [
    "eda_scr_peak_count", "eda_scr_amp_mean", "eda_scr_amp_max",
    "eda_scr_rise_time_mean", "eda_phasic_auc", "eda_slope", "eda_tonic_mean",
    "hrv_mean_ibi", "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
    "temp_mean", "temp_slope", "temp_max_drop",
    "acc_mag_mean", "acc_mag_std", "acc_mag_max",
]
FACIAL_FEATURES = [
    "lip_puckerer_mean", "lower_lip_depressor_mean", "upper_lip_raiser_mean",
    "mouth_stretch_mean", "lip_tightener_mean", "upper_lid_raiser_mean", "cheek_raiser_mean",
]
WEARABLE_CHANNELS = ["eda", "eda_phasic", "eda_tonic", "ibi_ms", "temp", "acc_mag"]
FEATURES = {"physio_only": WEARABLE_FEATURES, "multimodal": WEARABLE_FEATURES + FACIAL_FEATURES}
CHANNELS = {"physio_only": WEARABLE_CHANNELS, "multimodal": WEARABLE_CHANNELS + FACIAL_FEATURES}


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def event_equal_weights(frame):
    counts = frame.groupby(EVENT_COLUMNS, sort=False)[EVENT_COLUMNS[0]].transform("size").to_numpy(dtype=float)
    weights = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    positive = weights > 0
    if positive.any():
        weights = weights / weights[positive].mean()
    return weights


def make_tree_model(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=4,
                                      random_state=SEED, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, objective="multi:softprob", num_class=3,
                             eval_metric="mlogloss", random_state=SEED, n_jobs=1,
                             tree_method="hist", verbosity=0)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=220, max_depth=5, learning_rate=0.04, subsample=0.9,
                              colsample_bytree=0.9, objective="multiclass", num_class=3,
                              random_state=SEED, n_jobs=1, verbosity=-1)
    raise ValueError(f"Unknown tree model: {name}")


def fit_predict_tree(frame, feature_columns, train_idx, test_idx, name):
    features = frame[feature_columns].to_numpy(dtype=np.float64)
    train, test = features[train_idx], features[test_idx]
    medians = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    scaler = StandardScaler()
    train, test = scaler.fit_transform(train), scaler.transform(test)
    training_rows = frame.iloc[train_idx]
    model = make_tree_model(name)
    model.fit(train, training_rows["y_true"].to_numpy(dtype=int),
              sample_weight=event_equal_weights(training_rows))
    probabilities = np.zeros((len(test), 3), dtype=float)
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(test)
    return probabilities


class WeightedTinyCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.BatchNorm1d(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.BatchNorm1d(64), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 3)

    def forward(self, inputs):
        return self.head(self.net(inputs.transpose(1, 2)).squeeze(-1))


def fit_predict_cnn(frame, sequences, train_idx, test_idx, fold_seed):
    torch.manual_seed(fold_seed)
    torch.set_num_threads(1)
    train = sequences[train_idx].astype(np.float32)
    test = sequences[test_idx].astype(np.float32)
    training_rows = frame.iloc[train_idx]
    labels = training_rows["y_true"].to_numpy(dtype=np.int64)
    event_weights = event_equal_weights(training_rows).astype(np.float32)
    mean = np.nanmean(train, axis=(0, 1), keepdims=True)
    std = np.nanstd(train, axis=(0, 1), keepdims=True)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    train = np.nan_to_num((train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num((test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.bincount(labels, weights=event_weights.astype(np.float64), minlength=3).astype(np.float32)
    class_weights = np.zeros(3, dtype=np.float32)
    present = counts > 0
    class_weights[present] = counts[present].sum() / (present.sum() * counts[present])

    device = torch.device("cpu")
    model = WeightedTinyCNN(train.shape[2]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
                                  reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(torch.tensor(train, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.long),
                            torch.tensor(event_weights, dtype=torch.float32))
    generator = torch.Generator().manual_seed(fold_seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    for _ in range(4):
        model.train()
        for values, targets, weights in loader:
            values, targets, weights = values.to(device), targets.to(device), weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model(values), targets)
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()

    model.eval()
    probability_parts = []
    with torch.no_grad():
        for start in range(0, len(test), 512):
            values = torch.tensor(test[start:start + 512], dtype=torch.float32, device=device)
            probability_parts.append(torch.softmax(model(values), dim=1).cpu().numpy())
    return np.concatenate(probability_parts).astype(float)
