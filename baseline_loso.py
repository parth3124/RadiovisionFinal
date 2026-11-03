"""
baseline_loso.py - fast, torch-free leave-one-session-out (LOSO) check of the input normalization.

Hypothesis tested: the static multipath fingerprint of a TX/RX placement dominates the CSI, so a classifier
fits "which session" instead of "which gesture". If so, `--norm session` (or `packet`) should lift LOSO
accuracy above chance where `--norm global` does not.

    python baseline_loso.py --data dataset --norm global
    python baseline_loso.py --data dataset --norm packet
    python baseline_loso.py --data dataset --norm session

One feature vector per recording (mean + std over time of each subcarrier), logistic regression, scaler fit
on the training sessions only. The test session is never used for fitting, except that `--norm session`
uses its UNLABELED CSI to compute its own normalization statistics (no labels).
Reads: pooled accuracy over all LOSO folds (every recording is tested exactly once) and a binomial note.
"""
import argparse
import numpy as np
import common as C
import dataset as D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--norm", choices=["global", "packet", "session"], default="global")
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 strength (small = strong regularization)")
    a = ap.parse_args()
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    D.apply_norm(recs, a.norm)
    X, y, s = [], [], []
    for r in recs:
        g = r.G[r.valid].astype(np.float32)
        if len(g):
            X.append(np.concatenate([g.mean(0), g.std(0)])); y.append(r.label); s.append(r.session)
    X, y, s = np.array(X), np.array(y), np.array(s)
    pred = np.zeros_like(y)
    print(f"norm={a.norm}  recordings={len(y)}  sessions={sorted(map(str, set(s)))}")
    for sess in sorted(set(s)):
        te = s == sess
        m = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=3000, class_weight="balanced"))
        m.fit(X[~te], y[~te]); pred[te] = m.predict(X[te])
        print(f"  held out {sess:<10} n={te.sum():>3}  acc {100 * np.mean(pred[te] == y[te]):5.1f}%")
    n = len(y); acc = float(np.mean(pred == y)); se = np.sqrt(0.2 * 0.8 / n)
    print(f"\nPOOLED LOSO over {n} recordings: accuracy {100 * acc:.1f}%  balanced {100 * balanced_accuracy_score(y, pred):.1f}%  "
          f"macro-F1 {100 * f1_score(y, pred, average='macro', zero_division=0):.1f}%")
    print(f"chance = 20%; one-sided 95% threshold for 'better than chance' ~ {100 * (0.2 + 1.645 * se):.1f}% (n={n}, treats recordings as independent)")
    print("confusion (rows true, cols pred):\n", confusion_matrix(y, pred, labels=range(len(C.CLASSES))))


if __name__ == "__main__":
    main()