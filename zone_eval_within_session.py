
"""
zone_eval_within_session.py

Evaluate CSI classification within each recording session independently.

Examples:
    python3 zone_eval_within_session.py --data dataset_room
    python3 zone_eval_within_session.py --data dataset_room --norm packet
    python3 zone_eval_within_session.py --data dataset_room --norm none

Protocol:
- Stratified 5-fold cross-validation within r01 and r02 separately.
- Feature selection uses training recordings only.
- ref normalization uses empty-room calibration recordings:
    * Zone tasks: all empty recordings are calibration data because
      empty recordings are not examples in the zone classification task.
    * Presence tasks: only empty recordings in the training fold are
      used for calibration.
- Training empty recordings use leave-one-out empty-room references.
- StandardScaler and LogisticRegression are fitted on training data only.
- Existing zone_eval.py and the dataset are not modified.
"""

import argparse
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from zone_eval import load, model


def get_task_data(recs, session, condition, task):
    """Return recording indices and labels for one task and session."""
    indices = []
    labels = []

    for i, r in enumerate(recs):
        meta = r["meta"]

        if meta["session"] != session:
            continue

        c = meta["condition"]

        if task == "zone":
            if c == condition and meta.get("zone") in ("Z1", "Z2", "Z3"):
                indices.append(i)
                labels.append(("Z1", "Z2", "Z3").index(meta["zone"]))

        elif task == "presence":
            if c in ("empty", condition):
                indices.append(i)
                labels.append(int(c == condition))

    return np.asarray(indices, dtype=int), np.asarray(labels, dtype=int)


def make_features(recs, names, train_indices, target_indices,
                  norm, calibration_indices):
    """
    Build features for train/test recordings.

    The live-subcarrier mask is selected using train_indices only.
    calibration_indices are used only for empty-room reference
    normalization when norm == 'ref'.
    """
    train_indices = np.asarray(train_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    calibration_indices = np.asarray(calibration_indices, dtype=int)

    # Include calibration recordings when calculating their reference
    # feature vectors, without using them to select live subcarriers.
    needed = np.unique(
        np.concatenate([train_indices, target_indices, calibration_indices])
    )

    raw_by_index = {}

    for name in names:
        train_mean = np.mean(
            [recs[i]["links"][name].mean(axis=0) for i in train_indices],
            axis=0,
        )

        live = train_mean > 0.05 * np.median(train_mean)

        if not live.any():
            raise RuntimeError(
                f"No live subcarriers found for link {name}"
            )

        for i in needed:
            A = recs[i]["links"][name]
            log_amp = np.log1p(A)

            if norm != "none":
                log_amp = log_amp - log_amp[:, live].mean(
                    axis=1, keepdims=True
                )

            feature = log_amp[:, live].mean(axis=0)

            if i not in raw_by_index:
                raw_by_index[i] = []

            raw_by_index[i].append(feature)

    raw_by_index = {
        i: np.concatenate(parts)
        for i, parts in raw_by_index.items()
    }

    # Apply empty-room reference calibration.
    features = {}

    if norm == "ref":
        if len(calibration_indices) < 3:
            raise RuntimeError(
                "ref normalization requires at least 3 empty-room "
                "calibration recordings in the training fold/session."
            )

        calibration_mean = np.mean(
            [raw_by_index[i] for i in calibration_indices],
            axis=0,
        )

        calibration_set = set(calibration_indices.tolist())
        train_set = set(train_indices.tolist())

        for i in target_indices:
            meta = recs[i]["meta"]

            # Training empty examples must not calibrate against
            # themselves. Use the other training empty recordings.
            if (
                i in train_set
                and i in calibration_set
                and meta["condition"] == "empty"
            ):
                other_indices = [
                    j for j in calibration_indices if j != i
                ]

                reference = np.mean(
                    [raw_by_index[j] for j in other_indices],
                    axis=0,
                )
            else:
                # Test examples use training calibration only.
                reference = calibration_mean

            features[i] = raw_by_index[i] - reference

    else:
        features = {i: raw_by_index[i] for i in target_indices}

    return features


def evaluate_task(recs, names, session, condition, task, norm, Cc,
                  folds=5, seed=0):
    """Run stratified cross-validation for one session and task."""
    indices, y = get_task_data(recs, session, condition, task)

    if len(indices) == 0 or len(np.unique(y)) < 2:
        print(f"{condition}_{task}: skipped (insufficient classes)")
        return

    class_counts = np.bincount(y)
    nonzero_counts = class_counts[class_counts > 0]
    n_splits = min(folds, int(nonzero_counts.min()))

    if n_splits < 2:
        print(f"{condition}_{task}: skipped (too few samples per class)")
        return

    labels = [0, 1, 2] if task == "zone" else [0, 1]
    predictions = np.full(len(indices), -1, dtype=int)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    session_empty = np.asarray([
        i for i, r in enumerate(recs)
        if r["meta"]["session"] == session
        and r["meta"]["condition"] == "empty"
    ], dtype=int)

    for train_pos, test_pos in splitter.split(indices, y):
        train_indices = indices[train_pos]
        test_indices = indices[test_pos]

        if task == "zone":
            # Empty-room recordings are separate calibration data for
            # zone classification, not examples in its test set.
            calibration_indices = session_empty
        else:
            # Never use empty recordings from the test fold to calibrate.
            calibration_indices = np.asarray([
                i for i in train_indices
                if recs[i]["meta"]["condition"] == "empty"
            ], dtype=int)

        if norm == "ref" and len(calibration_indices) < 3:
            raise RuntimeError(
                f"Too few calibration recordings for {session}, "
                f"{condition}_{task}"
            )

        all_targets = np.concatenate([train_indices, test_indices])

        feats = make_features(
            recs=recs,
            names=names,
            train_indices=train_indices,
            target_indices=all_targets,
            norm=norm,
            calibration_indices=calibration_indices,
        )

        X_train = np.asarray([feats[i] for i in train_indices])
        X_test = np.asarray([feats[i] for i in test_indices])
        y_train = y[train_pos]

        clf = model(Cc)
        clf.fit(X_train, y_train)
        predictions[test_pos] = clf.predict(X_test)

    valid = predictions >= 0
    score = balanced_accuracy_score(y[valid], predictions[valid])
    chance = 100.0 / len(np.unique(y))

    print(
        f"{condition}_{task:<9} "
        f"n={len(y):2d} "
        f"chance={chance:5.1f}% "
        f"bal.acc={100 * score:5.1f}%"
    )

    cm = confusion_matrix(y[valid], predictions[valid], labels=labels)

    if task == "zone":
        print("    Confusion matrix (rows=true, columns=predicted; Z1,Z2,Z3):")
        print("   ", cm.tolist())
    else:
        print("    Confusion matrix (rows=true, columns=predicted; empty, present):")
        print("   ", cm.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_room")
    parser.add_argument(
        "--norm", choices=["ref", "packet", "none"], default="ref"
    )
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    recs, names = load(args.data)
    sessions = sorted({r["meta"]["session"] for r in recs})

    print(
        f"{len(recs)} recordings; links={names}; sessions={sessions}; "
        f"norm={args.norm}"
    )
    print(f"Protocol: within-session stratified {args.folds}-fold CV")
    print("Feature selection uses training recordings only.")

    if args.norm == "ref":
        print(
            "Reference calibration: training-fold empties for presence; "
            "session empty recordings for zone tasks."
        )

    for session in sessions:
        print(f"\n{'=' * 60}")
        print(f"SESSION: {session}")
        print(f"{'=' * 60}")

        for condition in ("obj", "person"):
            for task in ("zone", "presence"):
                evaluate_task(
                    recs=recs,
                    names=names,
                    session=session,
                    condition=condition,
                    task=task,
                    norm=args.norm,
                    Cc=args.C,
                    folds=args.folds,
                    seed=args.seed,
                )

    print("\nInterpretation:")
    print("- Zone tasks: 33.3% chance for three balanced zones.")
    print("- Presence tasks: 50% chance for two balanced classes.")
    print("- Compare these results with cross-session performance.")
    print("- This is a diagnostic evaluation, not proof of deployment accuracy.")


if __name__ == "__main__":
    main()
