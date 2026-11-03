import argparse
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from zone_eval import load, model


def features_for_direction(recs, names, norm, train_session):
    train_idx = [
        i for i, r in enumerate(recs)
        if r["meta"]["session"] == train_session
    ]

    # Select live subcarriers using training-session recordings only.
    live = {}
    for name in names:
        rm = np.mean(
            [recs[i]["links"][name].mean(0) for i in train_idx], axis=0
        )
        live[name] = rm > 0.05 * np.median(rm)

        if not live[name].any():
            raise SystemExit(f"No live subcarriers found for link {name}")

    raw = []
    for r in recs:
        parts = []
        for name in names:
            A = r["links"][name]
            Lg = np.log1p(A)
            if norm != "none":
                Lg = Lg - Lg[:, live[name]].mean(axis=1, keepdims=True)
            parts.append(Lg[:, live[name]].mean(axis=0))
        raw.append(np.concatenate(parts))

    raw = np.asarray(raw)
    feats = raw.copy()

    if norm == "ref":
        # Each session is calibrated using its own empty-room recordings.
        for session in sorted({r["meta"]["session"] for r in recs}):
            idx = [
                i for i, r in enumerate(recs)
                if r["meta"]["session"] == session
            ]
            emp = [
                i for i in idx
                if recs[i]["meta"]["condition"] == "empty"
            ]
            print(f"Calibration {session}: {len(emp)} empty recordings")

            if len(emp) < 3:
                raise SystemExit(
                    f"Need >=3 empty calibration recordings in {session}"
                )

            empty_mean = raw[emp].mean(axis=0)
            for i in idx:
                if i in emp:
                    other_empty_mean = (
                        raw[emp].sum(axis=0) - raw[i]
                    ) / (len(emp) - 1)
                    feats[i] = raw[i] - other_empty_mean
                else:
                    feats[i] = raw[i] - empty_mean

    return feats


def task_data(recs, feats, condition, task):
    indices, labels = [], []
    for i, r in enumerate(recs):
        m = r["meta"]
        c = m["condition"]

        if task == "zone":
            if c == condition and m.get("zone") in ("Z1", "Z2", "Z3"):
                indices.append(i)
                labels.append(("Z1", "Z2", "Z3").index(m["zone"]))
        elif c in ("empty", condition):
            indices.append(i)
            labels.append(int(c == condition))

    indices = np.asarray(indices, dtype=int)
    return indices, feats[indices], np.asarray(labels, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_room")
    ap.add_argument("--norm", choices=["ref", "packet", "none"], default="ref")
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    recs, names = load(args.data)
    sessions = sorted({r["meta"]["session"] for r in recs})

    if len(sessions) < 2:
        raise SystemExit(f"Need at least 2 sessions; found {sessions}")

    print(
        f"{len(recs)} recordings; sessions={sessions}; "
        f"norm={args.norm}; links={names}"
    )
    print("Live-subcarrier selection uses training-session recordings only.")
    if args.norm == "ref":
        print("Empty-room calibration uses each session's empty recordings.")

    for train_session in sessions:
        for test_session in sessions:
            if train_session == test_session:
                continue

            feats = features_for_direction(
                recs, names, args.norm, train_session
            )
            print(f"\nTRAIN {train_session} -> TEST {test_session}")

            for condition in ("obj", "person"):
                for task in ("zone", "presence"):
                    idx, X, y = task_data(recs, feats, condition, task)

                    tr = np.array([
                        recs[i]["meta"]["session"] == train_session
                        for i in idx
                    ])
                    te = np.array([
                        recs[i]["meta"]["session"] == test_session
                        for i in idx
                    ])

                    X_train, y_train = X[tr], y[tr]
                    X_test, y_test = X[te], y[te]

                    if (
                        len(y_train) == 0
                        or len(y_test) == 0
                        or len(np.unique(y_train)) < 2
                        or len(np.unique(y_test)) < 2
                    ):
                        print(f"{condition}_{task}: skipped (missing class)")
                        continue

                    clf = model(args.C)
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)

                    score = balanced_accuracy_score(y_test, pred)
                    chance = 100 / len(np.unique(y_test))

                    print(
                        f"{condition}_{task:<9} "
                        f"train={len(y_train):2d} test={len(y_test):2d} "
                        f"chance={chance:5.1f}% "
                        f"bal.acc={100 * score:5.1f}%"
                    )

                    if task == "zone":
                        cm = confusion_matrix(
                            y_test, pred, labels=[0, 1, 2]
                        )
                        print(
                            "    confusion rows=true, cols=pred [Z1,Z2,Z3]:",
                            cm.tolist()
                        )


if __name__ == "__main__":
    main()
