"""
loso_permutation.py - leave-one-session-out (LOSO) accuracy with an exact permutation p-value.

    python loso_permutation.py --data dataset_v2

For each normalization (global / packet / session) and feature set (mean, std, mean+std):
  observed LOSO accuracy, and a null distribution made by shuffling the labels WITHIN each session and
  repeating the whole LOSO procedure. p = fraction of shuffles at least as accurate as the real labels.
PRIMARY (decided before looking at the new data): norm=session, features=mean+std.
Everything else is exploratory; with 9 variants, only p < ~0.006 would survive a Bonferroni correction.
"""
import argparse
import numpy as np
import common as C
import dataset as D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def loso(X, y, s, Cc):
    pred = np.zeros_like(y)
    for u in np.unique(s):
        te = s == u
        m = make_pipeline(StandardScaler(), LogisticRegression(C=Cc, max_iter=1000, class_weight="balanced"))
        m.fit(X[~te], y[~te]); pred[te] = m.predict(X[te])
    return float(np.mean(pred == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v2")
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    raw = [r.G.copy() for r in recs]
    y = np.array([r.label for r in recs]); s = np.array([r.session for r in recs])
    print(f"{len(y)} recordings, sessions {sorted(set(s))}, {a.perms} label shuffles per variant\n")
    print(f"{'norm':<9}{'features':<11}{'LOSO acc':>9}{'null mean':>11}{'null 95th':>11}{'p':>8}")
    for norm in ("global", "packet", "session"):
        for r, g in zip(recs, raw):
            r.G = g.copy()
        D.apply_norm(recs, norm)
        M = np.array([r.G[r.valid].astype(np.float32).mean(0) for r in recs])
        S = np.array([r.G[r.valid].astype(np.float32).std(0) for r in recs])
        for name, X in (("mean", M), ("std", S), ("mean+std", np.hstack([M, S]))):
            obs = loso(X, y, s, a.C); null = []
            for _ in range(a.perms):
                yp = y.copy()
                for u in np.unique(s):
                    idx = np.where(s == u)[0]; yp[idx] = y[rng.permutation(idx)]
                null.append(loso(X, yp, s, a.C))
            null = np.array(null)
            p = (1 + (null >= obs).sum()) / (1 + a.perms)
            tag = "  <- PRIMARY" if (norm, name) == ("session", "mean+std") else ""
            print(f"{norm:<9}{name:<11}{100 * obs:>8.1f}%{100 * null.mean():>10.1f}%{100 * np.percentile(null, 95):>10.1f}%{p:>8.3f}{tag}")
    print("\nchance = 20%. Small p on the PRIMARY row = the cross-session accuracy is unlikely under no relationship.")


if __name__ == "__main__":
    main()