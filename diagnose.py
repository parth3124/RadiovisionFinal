"""
diagnose.py - is there ANY transferable gesture signal in the recording-level CSI features, and are the
labels consistent with the camera? Torch-free.

    python diagnose.py --data dataset
Prints:
 1. variance breakdown of recording-mean CSI features (session / class-within-session / recording noise)
 2. permutation test (labels shuffled WITHIN each session): does the class explain more variance than chance?
      - pooled across sessions (this is the part that could transfer between sessions)
      - inside each session separately
 3. camera sanity: LOSO gesture classification from camera joints (label / pose consistency)
"""
import argparse, os
import numpy as np
import common as C
import dataset as D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def features(recs):
    rm = np.mean([r.raw_mean for r in recs], 0)
    live = rm > 0.05 * np.median(rm)
    X = np.array([r.G[r.valid].astype(np.float32)[:, live].mean(0) for r in recs])
    return X


def class_share(X, y, s):
    """fraction of variance explained by class means pooled over sessions, after centering each session"""
    Xc = X.copy()
    for u in np.unique(s):
        Xc[s == u] -= Xc[s == u].mean(0)
    tot = (Xc ** 2).sum()
    ss = sum((y == c).sum() * ((Xc[y == c].mean(0)) ** 2).sum() for c in np.unique(y))
    return ss / tot


def within_share(X, y):
    Xc = X - X.mean(0)
    tot = (Xc ** 2).sum()
    ss = sum((y == c).sum() * ((Xc[y == c].mean(0)) ** 2).sum() for c in np.unique(y))
    return ss / tot


def perm_within_sessions(y, s, rng):
    yp = y.copy()
    for u in np.unique(s):
        idx = np.where(s == u)[0]
        yp[idx] = y[rng.permutation(idx)]
    return yp


def camera_features(r):
    p = os.path.join(r.path, "camera.npz")
    if not os.path.exists(p):
        return None
    c = np.load(p)
    j, ok = c["joints"], c["ok"]
    if ok.sum() < 10:
        return None
    j = j[ok]
    neck, rs, ls = j[:, 1, :2], j[:, 2, :2], j[:, 5, :2]
    sw = np.maximum(np.linalg.norm(rs - ls, axis=1), 1e-3)[:, None]
    f = [(j[:, k, :2] - neck) / sw for k in (3, 4, 6, 7)]          # elbows, wrists relative to neck, in shoulder widths
    f = np.concatenate(f, 1)
    return np.concatenate([f.mean(0), f.std(0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--perms", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for mode in ("session", "packet"):
        recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
        D.apply_norm(recs, mode)
        X = features(recs); y = np.array([r.label for r in recs]); s = np.array([r.session for r in recs])
        N, ns, nc = len(y), len(set(s)), len(C.CLASSES)
        print(f"\n===== input normalization: {mode}   ({N} recordings, {ns} sessions) =====")
        # 1. variance breakdown
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        tot = (Xs ** 2).sum()
        ss_sess = sum((s == u).sum() * ((Xs[s == u].mean(0) - Xs.mean(0)) ** 2).sum() for u in np.unique(s))
        cell = 0.0
        for u in np.unique(s):
            for c in np.unique(y):
                m = (s == u) & (y == c)
                if m.any():
                    cell += ((Xs[m] - Xs[m].mean(0)) ** 2).sum()
        print(f"variance share: session {100 * ss_sess / tot:5.1f}% | class-within-session + noise-free remainder: "
              f"{100 * (1 - ss_sess / tot - cell / tot):5.1f}% | recording-to-recording noise inside the same session+class {100 * cell / tot:5.1f}%")
        print(f"  (pure chance would give about {100 * (ns - 1) / (N - 1):.1f}% / {100 * (ns * nc - ns) / (N - 1):.1f}% / {100 * (N - ns * nc) / (N - 1):.1f}%)")
        # 2. permutation tests
        obs = class_share(X, y, s)
        null = np.array([class_share(X, perm_within_sessions(y, s, rng), s) for _ in range(a.perms)])
        p = (1 + (null >= obs).sum()) / (1 + a.perms)
        print(f"POOLED class share (transferable part): {100 * obs:.2f}%   chance mean {100 * null.mean():.2f}%   "
              f"95th pct of chance {100 * np.percentile(null, 95):.2f}%   permutation p = {p:.3f}")
        for u in sorted(np.unique(s)):
            m = s == u
            o = within_share(X[m], y[m])
            nl = np.array([within_share(X[m], rng.permutation(y[m])) for _ in range(a.perms)])
            print(f"  inside {u:<9} class share {100 * o:5.1f}%   chance mean {100 * nl.mean():5.1f}%   p = {(1 + (nl >= o).sum()) / (1 + a.perms):.3f}")
    # 3. camera sanity
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    F = [camera_features(r) for r in recs]
    keep = [i for i, f in enumerate(F) if f is not None]
    print("\n===== camera sanity (label / pose consistency) =====")
    if len(keep) < 0.5 * len(recs):
        print(f"camera data found for only {len(keep)}/{len(recs)} recordings - skipped")
        return
    Xc = np.array([F[i] for i in keep]); yc = np.array([recs[i].label for i in keep]); sc = np.array([recs[i].session for i in keep])
    pred = np.zeros_like(yc)
    for u in sorted(set(sc)):
        te = sc == u
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xc[~te], yc[~te])
        pred[te] = m.predict(Xc[te])
        print(f"  held out {u:<9} camera-joint accuracy {100 * np.mean(pred[te] == yc[te]):5.1f}%")
    print(f"POOLED LOSO camera-joint accuracy {100 * np.mean(pred == yc):.1f}% over {len(yc)} recordings (chance 20%)")
    print("low accuracy here would mean labels and visible poses disagree, or the poses are not distinct.")


if __name__ == "__main__":
    main()