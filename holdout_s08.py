"""
holdout_s08.py - the pre-registered replication test. Pipeline FROZEN before s08 was analysed:
    norm = session, features = per-subcarrier mean + std of each recording, StandardScaler + logistic
    regression (C = 0.1, class_weight balanced). Train on s06 + s07, test ONCE on s08.

    python holdout_s08.py --data dataset_v2

Outputs: accuracy / macro-F1 / balanced accuracy on s08, Wilson 95% interval, permutation p-value
(training labels shuffled within each training session, whole pipeline refit), per-class recall,
confusion matrix, per-session health. Then, clearly labelled EXPLORATORY, the 3-fold LOSO numbers.
"""
import argparse, os
import numpy as np
import common as C
import dataset as D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def fit_predict(Xtr, ytr, Xte, Cc):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=Cc, max_iter=1000, class_weight="balanced"))
    return m.fit(Xtr, ytr).predict(Xte)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v2")
    ap.add_argument("--train", nargs="+", default=["s06", "s07"])
    ap.add_argument("--test", default="s08")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    D.apply_norm(recs, "session")
    X = np.array([np.concatenate([r.G[r.valid].astype(np.float32).mean(0), r.G[r.valid].astype(np.float32).std(0)]) for r in recs])
    y = np.array([r.label for r in recs]); s = np.array([r.session for r in recs])
    print("per-session health:")
    for u in sorted(set(s)):
        rs = [r for r in recs if r.session == u]
        rssi = [float(np.mean(np.load(os.path.join(r.path, "csi.npz"))["rssi"])) for r in rs]
        print(f"  {u}: {len(rs)} recordings, CSI {np.mean([r.meta.get('csi_rate_hz', np.nan) for r in rs]):.1f} Hz, RSSI {np.mean(rssi):.1f} dBm")
    tr = np.isin(s, a.train); te = s == a.test
    if not tr.any() or not te.any():
        raise SystemExit(f"need sessions {a.train} for training and {a.test} for testing; found {sorted(set(s))}")
    pred = fit_predict(X[tr], y[tr], X[te], a.C)
    acc = accuracy_score(y[te], pred); n = int(te.sum()); lo, hi = wilson(int(round(acc * n)), n)
    null = []
    for _ in range(a.perms):
        yp = y.copy()
        for u in a.train:
            idx = np.where(s == u)[0]; yp[idx] = y[rng.permutation(idx)]
        null.append(accuracy_score(y[te], fit_predict(X[tr], yp[tr], X[te], a.C)))
    null = np.array(null); p = (1 + (null >= acc).sum()) / (1 + a.perms)
    print(f"\n=== PRE-REGISTERED TEST: train {a.train} -> test {a.test}  ({tr.sum()} train, {n} test recordings) ===")
    print(f"accuracy {100 * acc:.1f}%  (95% interval {100 * lo:.0f}-{100 * hi:.0f}%)   balanced {100 * balanced_accuracy_score(y[te], pred):.1f}%   "
          f"macro-F1 {100 * f1_score(y[te], pred, average='macro', zero_division=0):.1f}%")
    print(f"permutation null: mean {100 * null.mean():.1f}%, 95th pct {100 * np.percentile(null, 95):.1f}%   p = {p:.4f}   (chance 20%)")
    cm = confusion_matrix(y[te], pred, labels=range(5))
    print("per-class recall: " + "  ".join(f"{c} {100 * cm[i, i] / max(cm[i].sum(), 1):.0f}%" for i, c in enumerate(C.CLASSES)))
    print("confusion (rows true, cols pred), order " + ", ".join(C.CLASSES) + ":\n", cm)
    print("\n--- EXPLORATORY (not pre-registered): 3-fold leave-one-session-out ---")
    allp = np.zeros_like(y)
    for u in sorted(set(s)):
        t = s == u
        allp[t] = fit_predict(X[~t], y[~t], X[t], a.C)
        print(f"  held out {u}: {100 * np.mean(allp[t] == y[t]):.1f}%")
    print(f"  pooled: {100 * np.mean(allp == y):.1f}%  macro-F1 {100 * f1_score(y, allp, average='macro', zero_division=0):.1f}%")


if __name__ == "__main__":
    main()
    