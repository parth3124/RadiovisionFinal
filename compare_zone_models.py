import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from common import csi_amplitude

d = np.load("merged/s01_labeled.npz")

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

X, Y, groups = [], [], []

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

            features = np.concatenate([
                v.mean(axis=0).flatten(),
                v.std(axis=0).flatten()
            ])

            X.append(features)
            Y.append(np.bincount(lab[w]).argmax())
            groups.append(trial)

        start += 1.0

X = np.asarray(X)
Y = np.asarray(Y)
groups = np.asarray(groups)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RBF SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        )
    )
}

for name, model in models.items():
    predictions = np.full(len(Y), -1)

    for trial in np.unique(groups):
        train = groups != trial
        test = groups == trial

        if len(np.unique(Y[train])) < 3:
            continue

        model.fit(X[train], Y[train])
        predictions[test] = model.predict(X[test])

    good = predictions >= 0

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Overall accuracy:",
          round(accuracy_score(Y[good], predictions[good]) * 100, 2), "%")

    print("Confusion matrix:")
    print(confusion_matrix(
        Y[good], predictions[good], labels=[0, 1, 2]
    ))

    trial_scores = []

    for trial in np.unique(groups):
        mask = (groups == trial) & good
        if mask.sum():
            score = accuracy_score(Y[mask], predictions[mask])
            trial_scores.append(score)

    print("Mean accuracy across trials:",
          round(np.mean(trial_scores) * 100, 2), "%")
