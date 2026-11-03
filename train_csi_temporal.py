
#!/usr/bin/env python3
"""
CSI temporal person-zone classification.

Architecture:
    CSI amplitude sequence
        -> per-packet gain normalization
        -> 1D CNN
        -> BiLSTM
        -> temporal mean pooling
        -> fully connected layer
        -> Z1 / Z2 / Z3

Evaluation:
    Train on one session and test on another.
    All feature selection and standardization are fitted on the
    training session only.

This script does not modify the dataset or existing scripts.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import csi_amplitude


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------

def read_metadata(recording_dir):
    """Read meta.json and return its dictionary."""
    meta_path = recording_dir / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label(meta):
    """
    Return the person's zone as 0, 1, or 2.
    Non-person recordings are excluded.
    """
    # Support common metadata layouts.
    candidates = [
        meta.get("label"),
        meta.get("gesture"),
        meta.get("condition"),
        meta.get("class"),
        meta.get("target"),
        meta.get("recording"),
        meta.get("name"),
    ]

    # Also inspect nested metadata dictionaries.
    for key in ("labels", "recording_info", "experiment"):
        value = meta.get(key)
        if isinstance(value, dict):
            candidates.extend([
                value.get("label"),
                value.get("condition"),
                value.get("class"),
                value.get("target"),
                value.get("name"),
            ])

    text = " ".join(str(x) for x in candidates if x is not None)
    text = text.lower().replace("-", "_").replace(" ", "_")

    # Avoid accidentally labelling object recordings as people.
    if "person" not in text:
        return None

    for i, zone in enumerate(("z1", "z2", "z3")):
        if zone in text:
            return i

    return None


def find_csi_file(recording_dir):
    """Find the CSI NPZ file, supporting the link/rx1 naming variants."""
    preferred = [
        recording_dir / "csi_link.npz",
        recording_dir / "csi_rx1.npz",
    ]

    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(recording_dir.glob("csi_*.npz"))
    return candidates[0] if candidates else None


def load_recording(recording_dir, trim_seconds=0.5):
    """
    Return amplitude array (packets, subcarriers) and packet times.

    The modal csi_len is used to retain packets with a consistent CSI
    width. The first and last trim_seconds are discarded.
    """
    csi_path = find_csi_file(recording_dir)
    if csi_path is None:
        return None

    with np.load(csi_path) as z:
        required = {"csi", "host_t"}
        if not required.issubset(z.files):
            print(f"Skipping {csi_path}: missing {required - set(z.files)}")
            return None

        csi = z["csi"]
        times = z["host_t"].astype(np.float64)

        if "csi_len" in z.files:
            lengths = z["csi_len"].astype(int)
            valid_lengths = lengths[lengths > 0]

            if len(valid_lengths) == 0:
                return None

            length = int(np.bincount(valid_lengths).argmax())
            keep = lengths == length
            csi = csi[keep]
            times = times[keep]
        else:
            length = csi.shape[1]

        # CSI values are interleaved I/Q components.
        length = min(length, csi.shape[1])
        csi = csi[:, :length]

    if len(times) < 100:
        return None

    # Ensure chronological order.
    order = np.argsort(times)
    times = times[order]
    csi = csi[order]

    # Remove duplicate or non-increasing timestamps.
    keep = np.r_[True, np.diff(times) > 0]
    times = times[keep]
    csi = csi[keep]

    if len(times) < 100:
        return None

    # host_t is expected to be in seconds for this dataset.
    start = times[0] + trim_seconds
    end = times[-1] - trim_seconds
    keep = (times >= start) & (times <= end)

    times = times[keep]
    csi = csi[keep]

    if len(times) < 50:
        return None

    amplitude = csi_amplitude(csi)

    if amplitude.ndim != 2 or amplitude.shape[0] != len(times):
        return None

    return amplitude.astype(np.float32), times


def get_session(recording_dir):
    """Extract session name such as r01 from the directory structure."""
    for part in recording_dir.parts:
        if part.lower() in ("r01", "r02", "r03"):
            return part.lower()

    # Some datasets may place the session in a parent folder name.
    for part in reversed(recording_dir.parts):
        low = part.lower()
        if low.startswith("r01") or low.startswith("r02") or low.startswith("r03"):
            return low[:3]

    return None


def collect_records(dataset_root):
    """
    Collect person-zone recordings.

    Recursively searches for meta.json files, so the dataset can have
    session folders nested inside other folders.
    """
    records = []

    for meta_path in sorted(dataset_root.rglob("meta.json")):
        recording_dir = meta_path.parent
        session = get_session(recording_dir)

        if session is None:
            continue

        try:
            meta = read_metadata(recording_dir)
            label = get_label(meta)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {meta_path}: {exc}")
            continue

        if label is None:
            continue

        loaded = load_recording(recording_dir)
        if loaded is None:
            print(f"Skipping unusable recording: {recording_dir}")
            continue

        amplitude, times = loaded

        records.append({
            "path": recording_dir,
            "session": session,
            "label": label,
            "amplitude": amplitude,
            "times": times,
        })

    return records


# ------------------------------------------------------------
# Sequence preprocessing
# ------------------------------------------------------------

def resample_sequence(amplitude, times, steps=256):
    """Resample a recording to a fixed number of time steps."""
    relative_time = times - times[0]
    target_time = np.linspace(
        relative_time[0],
        relative_time[-1],
        steps,
        dtype=np.float64,
    )

    output = np.empty((steps, amplitude.shape[1]), dtype=np.float32)

    for subcarrier in range(amplitude.shape[1]):
        output[:, subcarrier] = np.interp(
            target_time,
            relative_time,
            amplitude[:, subcarrier],
        )

    return output


def packet_normalize(sequence, live_mask):
    """
    Log amplitude and remove packet-wise common gain.

    Input shape: (time, subcarriers)
    """
    sequence = np.log1p(np.maximum(sequence, 0.0))

    selected = sequence[:, live_mask]

    # Remove the common packet-level gain component.
    packet_mean = selected.mean(axis=1, keepdims=True)
    selected = selected - packet_mean

    return selected.astype(np.float32)


def prepare_train_test(train_records, test_records, steps=256):
    """
    Fit live-subcarrier selection and standardization on training
    recordings only. Apply the same transformation to test recordings.
    """
    train_raw = [
        resample_sequence(r["amplitude"], r["times"], steps)
        for r in train_records
    ]

    test_raw = [
        resample_sequence(r["amplitude"], r["times"], steps)
        for r in test_records
    ]

    # Find active subcarriers using training recordings only.
    train_stack = np.concatenate(train_raw, axis=0)
    subcarrier_strength = np.mean(train_stack, axis=0)

    threshold = max(
        float(np.median(subcarrier_strength)) * 0.05,
        1e-6,
    )
    live_mask = subcarrier_strength > threshold

    if live_mask.sum() < 8:
        # Fallback if the threshold removes too many channels.
        live_mask = subcarrier_strength > 0

    if live_mask.sum() < 8:
        raise RuntimeError("Too few usable subcarriers in training data.")

    X_train = np.stack([
        packet_normalize(x, live_mask) for x in train_raw
    ])

    X_test = np.stack([
        packet_normalize(x, live_mask) for x in test_raw
    ])

    # Training-only channel standardization.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-4)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Avoid extreme numerical values.
    X_train = np.clip(X_train, -8, 8).astype(np.float32)
    X_test = np.clip(X_test, -8, 8).astype(np.float32)

    return X_train, X_test, int(live_mask.sum())


# ------------------------------------------------------------
# CNN + BiLSTM model
# ------------------------------------------------------------

class CSICNNBiLSTM(nn.Module):
    def __init__(self, input_channels, num_classes=3, dropout=0.4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                input_channels, 32, kernel_size=7, padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                32, 48, kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # Input: (batch, time, subcarriers)
        x = x.transpose(1, 2)  # (batch, subcarriers, time)
        x = self.cnn(x)        # (batch, 48, reduced_time)
        x = x.transpose(1, 2)  # (batch, reduced_time, 48)

        x, _ = self.lstm(x)     # (batch, reduced_time, 64)

        # Aggregate information across the recording.
        x = x.mean(dim=1)       # (batch, 64)

        return self.classifier(x)


# ------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------

def balanced_accuracy(y_true, y_pred, num_classes=3):
    scores = []

    for cls in range(num_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            scores.append(float((y_pred[mask] == cls).mean()))

    return float(np.mean(scores)) if scores else 0.0


def train_one_direction(
    train_records,
    test_records,
    epochs=40,
    batch_size=6,
    learning_rate=1e-3,
    weight_decay=1e-3,
    seed=42,
):
    set_seed(seed)

    X_train, X_test, live_count = prepare_train_test(
        train_records, test_records
    )

    y_train = np.array(
        [r["label"] for r in train_records], dtype=np.int64
    )
    y_test = np.array(
        [r["label"] for r in test_records], dtype=np.int64
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = CSICNNBiLSTM(
        input_channels=X_train.shape[2],
        num_classes=3,
        dropout=0.4,
    ).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    # No test-session data is used for early stopping or model selection.
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # Stabilize gradients through the recurrent layer.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            count += len(xb)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    epoch {epoch + 1:02d}/{epochs} "
                f"loss={epoch_loss / max(count, 1):.4f}"
            )

    model.eval()

    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).to(device)
        logits = model(test_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()

    acc = float((predictions == y_test).mean())
    bal_acc = balanced_accuracy(y_test, predictions)

    return acc, bal_acc, y_test, predictions, live_count, str(device)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-session CSI person-zone CNN-BiLSTM experiment"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset_room"),
        help="Dataset root folder (default: dataset_room)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit(
            f"Dataset folder not found: {dataset_root}\n"
            "Run from the project directory or use --dataset /full/path."
        )

    records = collect_records(dataset_root)

    if not records:
        raise SystemExit(
            "No person-zone recordings were found. Check meta.json label "
            "fields and session folder names."
        )

    sessions = sorted(set(r["session"] for r in records))

    print(f"\nDataset: {dataset_root}")
    print(f"Person-zone recordings loaded: {len(records)}")
    print(f"Sessions: {sessions}")
    print("Classes: Z1=0, Z2=1, Z3=2")
    print("Evaluation: train one session, test on another")
    print(f"Epochs: {args.epochs}")
    print("-" * 72)

    for session in sessions:
        session_records = [r for r in records if r["session"] == session]
        counts = [
            sum(r["label"] == cls for r in session_records)
            for cls in range(3)
        ]
        print(f"{session}: {len(session_records)} recordings; class counts={counts}")

    results = []

    for train_session in sessions:
        for test_session in sessions:
            if train_session == test_session:
                continue

            train_records = [
                r for r in records if r["session"] == train_session
            ]
            test_records = [
                r for r in records if r["session"] == test_session
            ]

            print(f"\nTRAIN {train_session} -> TEST {test_session}")
            print(
                f"  train={len(train_records)}, test={len(test_records)}"
            )

            train_counts = [
                sum(r["label"] == cls for r in train_records)
                for cls in range(3)
            ]
            test_counts = [
                sum(r["label"] == cls for r in test_records)
                for cls in range(3)
            ]

            if min(train_counts) == 0 or min(test_counts) == 0:
                print(
                    "  Skipping: one or more classes are missing. "
                    f"train_counts={train_counts}, test_counts={test_counts}"
                )
                continue

            acc, bal_acc, y_true, y_pred, live_count, device = (
                train_one_direction(
                    train_records,
                    test_records,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                )
            )

            print(f"  device: {device}")
            print(f"  live subcarriers: {live_count}")
            print(f"  accuracy: {acc * 100:.1f}%")
            print(f"  balanced accuracy: {bal_acc * 100:.1f}%")

            confusion = np.zeros((3, 3), dtype=int)
            for actual, predicted in zip(y_true, y_pred):
                confusion[actual, predicted] += 1

            print("  confusion matrix (rows=true Z1/Z2/Z3; columns=predicted):")
            print(confusion)

            results.append({
                "train": train_session,
                "test": test_session,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
            })

    print("\n" + "=" * 72)
    print("CROSS-SESSION SUMMARY")
    print("=" * 72)

    if not results:
        print("No valid train/test directions were completed.")
        return

    print(
        f"{'Direction':<18} {'Accuracy':>12} {'Balanced acc.':>16}"
    )

    for result in results:
        direction = f"{result['train']} -> {result['test']}"
        print(
            f"{direction:<18} "
            f"{result['accuracy'] * 100:>11.1f}% "
            f"{result['balanced_accuracy'] * 100:>15.1f}%"
        )

    mean_acc = np.mean([r["accuracy"] for r in results])
    mean_bal_acc = np.mean([r["balanced_accuracy"] for r in results])

    print("-" * 48)
    print(f"{'Mean':<18} {mean_acc * 100:>11.1f}% {mean_bal_acc * 100:>15.1f}%")
    print("\nThree-class chance accuracy is 33.3% for balanced classes.")
    print(
        "Compare these results with your existing cross-session baseline. "
        "A neural network score alone does not establish generalization."
    )


if __name__ == "__main__":
    main()
