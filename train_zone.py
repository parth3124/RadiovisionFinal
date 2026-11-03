import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from common import csi_amplitude

# --------------------------------------------------
# 1. Load labelled recording
# --------------------------------------------------

d = np.load("merged/s01_labeled.npz")

tx_t = d["tx_t"]
csi = d["csi"]
csi_len = d["csi_len"]
present = d["present"]
y = d["y"]
trial_id = d["trial_id"]

zone_names = ["A", "B", "C"]

# --------------------------------------------------
# 2. Select packets available at BOTH RX1 and RX2
# --------------------------------------------------

valid = (
    (y >= 0)
    & present[:, 0]
    & present[:, 1]
    & (csi_len[:, 0] == 256)
    & (csi_len[:, 1] == 256)
)

indices = np.where(valid)[0]

print("Valid two-RX packets:", len(indices))

# --------------------------------------------------
# 3. Convert CSI into amplitude
# --------------------------------------------------

# 256 int8 values = 128 I/Q pairs = 128 subcarriers
amp_rx1 = csi_amplitude(csi[indices, 0, :256])
amp_rx2 = csi_amplitude(csi[indices, 1, :256])

# Shape: (packets, 2 receivers, 128 subcarriers)
amplitude = np.stack([amp_rx1, amp_rx2], axis=1)

packet_times = tx_t[indices]
packet_labels = y[indices]
packet_trials = trial_id[indices]

# --------------------------------------------------
# 4. Create one feature vector per 1-second window
# --------------------------------------------------

X = []
Y = []
groups = []

WINDOW_SEC = 1.0
MIN_PACKETS = 70

for trial in np.unique(packet_trials):
    trial_mask = packet_trials == trial

    times = packet_times[trial_mask]
    labels = packet_labels[trial_mask]
    values = amplitude[trial_mask]

    if len(times) == 0:
        continue

    start = times.min()
    end = times.max()

    window_start = start

    while window_start + WINDOW_SEC <= end:
        window_mask = (
            (times >= window_start)
            & (times < window_start + WINDOW_SEC)
        )

        if window_mask.sum() >= MIN_PACKETS:
            window_values = values[window_mask]

            # Mean and standard deviation across packets.
            means = window_values.mean(axis=0)
            stds = window_values.std(axis=0)

            features = np.concatenate([
                means.flatten(),
                stds.flatten()
            ])

            # All packets in a trial should have the same zone label.
            label = int(np.bincount(labels[window_mask]).argmax())

            X.append(features)
            Y.append(label)
            groups.append(int(trial))

        window_start += WINDOW_SEC

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.int64)
groups = np.asarray(groups, dtype=np.int64)

print("\nWindow dataset")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Unique trials:", len(np.unique(groups)))

for i, zone in enumerate(zone_names):
    print(f"Zone {zone}: {(Y == i).sum()} windows")

# --------------------------------------------------
# 5. Leave-one-trial-out evaluation
# --------------------------------------------------

predictions = np.full(len(Y), -1, dtype=np.int64)

unique_trials = np.unique(groups)

for test_trial in unique_trials:
    train_mask = groups != test_trial
    test_mask = groups == test_trial

    if len(np.unique(Y[train_mask])) != 3:
        print(f"Skipping trial {test_trial}: training lacks a zone")
        continue

    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X[train_mask], Y[train_mask])
    predictions[test_mask] = model.predict(X[test_mask])

# --------------------------------------------------
# 6. Results
# --------------------------------------------------

evaluated = predictions >= 0

if not np.any(evaluated):
    raise RuntimeError("No trials were evaluated.")

accuracy = accuracy_score(Y[evaluated], predictions[evaluated])

print("\n================================")
print("LEAVE-ONE-TRIAL-OUT RESULTS")
print("================================")
print(f"Evaluated windows: {evaluated.sum()}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion matrix (rows=true, columns=predicted)")
print(confusion_matrix(
    Y[evaluated],
    predictions[evaluated],
    labels=[0, 1, 2]
))

print("\nClassification report")
print(classification_report(
    Y[evaluated],
    predictions[evaluated],
    labels=[0, 1, 2],
    target_names=zone_names,
    zero_division=0
))

print("\nAccuracy by held-out trial")
for trial in unique_trials:
    mask = (groups == trial) & evaluated
    if mask.sum():
        score = accuracy_score(Y[mask], predictions[mask])
        print(f"Trial {trial + 1:2d}: {score * 100:5.1f}% ({mask.sum()} windows)")
