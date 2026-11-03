import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from common import csi_amplitude

# Load labelled dataset
d = np.load("merged/s01_labeled.npz")

# Use exactly the same valid packets for all receiver configurations
valid = (
    (d["y"] >= 0)
    & d["present"][:, 0]
    & d["present"][:, 1]
    & (d["csi_len"][:, 0] == 256)
    & (d["csi_len"][:, 1] == 256)
)

idx = np.where(valid)[0]

a1 = csi_amplitude(d["csi"][idx, 0, :256])
a2 = csi_amplitude(d["csi"][idx, 1, :256])

amplitude = np.stack([a1, a2], axis=1)

times = d["tx_t"][idx]
labels = d["y"][idx]
trials = d["trial_id"][idx]

# Build one-second windows
X1, X2, Y, groups = [], [], [], []

for trial in np.unique(trials):
    mask = trials == trial

    t = times[mask]
    values = amplitude[mask]
    lab = labels[mask]

    start = t.min()
    end = t.max()

    while start + 1.0 <= end:
        w = (t >= start) & (t < start + 1.0)

        if w.sum() >= 70:
            v = values[w]

            f1 = np.concatenate([
                v[:, 0, :].mean(axis=0),
                v[:, 0, :].std(axis=0)
            ])

            f2 = np.concatenate([
                v[:, 1, :].mean(axis=0),
                v[:, 1, :].std(axis=0)
            ])

            X1.append(f1)
            X2.append(f2)
            Y.append(np.bincount(lab[w]).argmax())
            groups.append(trial)

        start += 1.0

X1 = np.asarray(X1)
X2 = np.asarray(X2)
Y = np.asarray(Y)
groups = np.asarray(groups)

print("Windows:", len(Y))
print("RX1 features:", X1.shape)
print("RX2 features:", X2.shape)
print("Combined features:", (X1.shape[0], X1.shape[1] + X2.shape[1]))
print("Unique trials:", len(np.unique(groups)))

# Evaluate each receiver configuration
configurations = {
    "RX1 only": X1,
    "RX2 only": X2,
    "RX1 + RX2": np.concatenate([X1, X2], axis=1)
}

for name, X in configurations.items():
    predictions = np.full(len(Y), -1)

    for trial in np.unique(groups):
        train = groups != trial
        test = groups == trial

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X[train], Y[train])
        predictions[test] = model.predict(X[test])

    good = predictions >= 0

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Overall accuracy:",
          round(accuracy_score(Y[good], predictions[good]) * 100, 2), "%")

    print("Confusion matrix (rows=true A/B/C, columns=predicted A/B/C):")
    print(confusion_matrix(Y[good], predictions[good], labels=[0, 1, 2]))

    print("\nClassification report:")
    print(classification_report(
        Y[good],
        predictions[good],
        labels=[0, 1, 2],
        target_names=["A", "B", "C"],
        zero_division=0
    ))

    trial_scores = []

    for trial in np.unique(groups):
        mask = (groups == trial) & good
        if mask.sum():
            trial_scores.append(
                accuracy_score(Y[mask], predictions[mask])
            )

    print("Mean accuracy across trials:",
          round(np.mean(trial_scores) * 100, 2), "%")
