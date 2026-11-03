"""
position_confound.py - does where you STAND (camera-measured) explain the recording-to-recording CSI variation
that stops within-session classification? Torch-free; needs camera.npz in the recordings.

    python position_confound.py --data dataset --norm session

Position covariates per recording (camera, mean over frames with a pose): neck x, neck y, shoulder width
(distance proxy), all in normalized image units.
Part 1: after removing each (session, class) cell mean, do recordings with similar position have similar
        CSI? Spearman correlation of pairwise distances (same session), permutation p-value (position
        shuffled inside each cell).
Part 2: label-free linear removal of position from the CSI features (per session), then the same
        calibration test as fewshot_session.py (nearest class mean, k recordings/class). The camera is
        used at test time here, so this is an UPPER BOUND for what position-invariance could give.
"""
import argparse, os
import numpy as np
from scipy.stats import spearmanr
import common as C
import dataset as D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def position(r):
    p = os.path.join(r.path, "camera.npz")
    if not os.path.exists(p):
        return None
    c = np.load(p); ok = c["ok"]
    if ok.sum() < 10:
        return None
    j = c["joints"][ok]
    sw = np.linalg.norm(j[:, 2, :2] - j[:, 5, :2], axis=1)
    return np.array([j[:, 1, 0].mean(), j[:, 1, 1].mean(), sw.mean()], np.float32)


def ncm(X, y, k, draws, rng):
    sc = StandardScaler().fit(X); Z = sc.transform(X); accs = []
    for _ in range(draws):
        cal = []
        for c in range(len(C.CLASSES)):
            ids = np.where(y == c)[0]; kk = min(k, len(ids) - 1)
            if kk >= 1:
                cal += list(rng.choice(ids, kk, replace=False))
        cal = np.array(cal); te = np.setdiff1d(np.arange(len(y)), cal)
        cls = sorted(set(y[cal])); cent = np.array([Z[cal][y[cal] == c].mean(0) for c in cls])
        d = ((Z[te][:, None] - cent[None]) ** 2).sum(2)
        accs.append(np.mean(np.array(cls)[d.argmin(1)] == y[te]))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--norm", choices=["global", "packet", "session"], default="session")
    ap.add_argument("--perms", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge strength for position removal")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    D.apply_norm(recs, a.norm)
    P = [position(r) for r in recs]
    keep = [i for i, p in enumerate(P) if p is not None]
    if len(keep) < 0.8 * len(recs):
        raise SystemExit(f"camera position available for only {len(keep)}/{len(recs)} recordings")
    recs = [recs[i] for i in keep]; P = np.array([P[i] for i in keep])
    X = np.array([np.concatenate([r.G[r.valid].astype(np.float32).mean(0), r.G[r.valid].astype(np.float32).std(0)]) for r in recs])
    y = np.array([r.label for r in recs]); s = np.array([r.session for r in recs])
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-6)
    print(f"{len(recs)} recordings with camera position; norm={a.norm}")
    for u in sorted(set(s)):
        m = s == u
        print(f"  {u}: position spread (std) neck x {P[m, 0].std():.3f}  neck y {P[m, 1].std():.3f}  shoulder width {P[m, 2].std():.3f}")

    # ---- part 1 ----
    R = X.copy(); cells = {}
    for u in set(s):
        for c in set(y):
            m = np.where((s == u) & (y == c))[0]
            if len(m):
                cells[(u, c)] = m; R[m] -= R[m].mean(0)
    pairs = [(i, j) for u in set(s) for i in np.where(s == u)[0] for j in np.where(s == u)[0] if i < j]
    pi, pj = np.array(pairs).T
    dX = np.linalg.norm(R[pi] - R[pj], axis=1)
    def corr(Pm):
        return spearmanr(dX, np.linalg.norm(Pm[pi] - Pm[pj], axis=1))[0]
    obs = corr(Pz); null = []
    for _ in range(a.perms):
        Pp = Pz.copy()
        for m in cells.values():
            Pp[m] = Pz[rng.permutation(m)]
        null.append(corr(Pp))
    null = np.array(null)
    print(f"\nPART 1  position-distance vs CSI-distance (within session, class effect removed): Spearman {obs:+.3f}   "
          f"chance mean {null.mean():+.3f}   95th pct {np.percentile(null, 95):+.3f}   p = {(1 + (null >= obs).sum()) / (1 + a.perms):.3f}")
    print("  positive and significant = recordings where you stood in a similar place look more alike in CSI")

    # ---- part 2 ----
    print("\nPART 2  calibration test (nearest class mean), raw features vs position-removed features")
    print(f"{'k/class':>8} {'features':<18}" + "".join(f"{u:>10}" for u in sorted(set(s))) + f"{'MEAN':>9}")
    Xadj = X.copy()
    for u in set(s):
        m = s == u
        Xadj[m] = X[m] - Ridge(alpha=a.alpha).fit(Pz[m], X[m]).predict(Pz[m])      # label-free
    for k in (2, 3):
        for name, F in (("raw", X), ("position removed", Xadj)):
            per = [100 * ncm(F[s == u], y[s == u], k, 30, rng) for u in sorted(set(s))]
            print(f"{k:>8} {name:<18}" + "".join(f"{v:>9.1f}%" for v in per) + f"{np.mean(per):>8.1f}%")
    print("\nchance 20%; a gain under ~8 points is within noise. A big gain would justify position-invariant training.")


if __name__ == "__main__":
    main()