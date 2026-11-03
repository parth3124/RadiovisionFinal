"""
time_confound.py - is the within-session "class signal" a GESTURE effect or slow DRIFT that happens to line up
with the recording order? Torch-free. Uses meta.json "created" timestamps (collect.py writes one per recording).

    python time_confound.py --data dataset --norm session

Per session prints:
 * the class order in which the recordings were made (I idle, C hands_crossed, U hands_up, S arm_side, F arm_forward)
 * 'grouped p': are same-class recordings adjacent in time more often than in a random order?
 * calibration accuracy (k recordings/class as calibration, test on the rest of the session) for
     CSI    nearest class mean on CSI features
     time   NO CSI: label of the nearest-in-time calibration recording
 * pair analysis: CSI distance between two recordings ~ same-class + time gap (rank regression).
     coef_same < 0 : same-class recordings are more alike than their time gap explains (gesture effect)
     coef_gap  > 0 : recordings further apart in time are less alike (drift)
"""
import argparse, datetime as dt
import numpy as np
from scipy.stats import rankdata
import common as C
import dataset as D
from sklearn.preprocessing import StandardScaler

LETTERS = "ICUSF"


def tstamp(r):
    try:
        return dt.datetime.fromisoformat(r.meta["created"]).timestamp()
    except Exception:
        return None


def partial_coefs(dX, same, gap):
    z = lambda v: (rankdata(v) - rankdata(v).mean()) / (rankdata(v).std() + 1e-9)
    A = np.stack([np.ones(len(dX)), z(same.astype(float)), z(gap)], 1)
    return np.linalg.lstsq(A, z(dX), rcond=None)[0][1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--norm", choices=["global", "packet", "session"], default="session")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    D.apply_norm(recs, a.norm)
    recs = [r for r in recs if tstamp(r) is not None]
    X = np.array([np.concatenate([r.G[r.valid].astype(np.float32).mean(0), r.G[r.valid].astype(np.float32).std(0)]) for r in recs])
    y = np.array([r.label for r in recs]); s = np.array([r.session for r in recs]); t = np.array([tstamp(r) for r in recs])
    rows = []
    for u in sorted(set(s)):
        idx = np.where(s == u)[0]; idx = idx[np.argsort(t[idx])]
        yo, to, Xo = y[idx], t[idx], X[idx]
        print(f"\n{u}: recording order  " + "".join(LETTERS[c] for c in yo) + f"   (span {(to[-1] - to[0]) / 60:.0f} min)")
        adj = lambda q: int((q[1:] == q[:-1]).sum())
        obs = adj(yo); null = np.array([adj(rng.permutation(yo)) for _ in range(a.perms)])
        gp = (1 + (null >= obs).sum()) / (1 + a.perms)
        Z = StandardScaler().fit_transform(Xo)
        out = {"grouped_p": gp}
        for k in a.ks:
            ac, at = [], []
            for _ in range(30):
                cal = []
                for c in range(len(C.CLASSES)):
                    ids = np.where(yo == c)[0]; kk = min(k, len(ids) - 1)
                    if kk >= 1:
                        cal += list(rng.choice(ids, kk, replace=False))
                cal = np.array(cal); te = np.setdiff1d(np.arange(len(yo)), cal)
                cls = sorted(set(yo[cal])); cent = np.array([Z[cal][yo[cal] == c].mean(0) for c in cls])
                d = ((Z[te][:, None] - cent[None]) ** 2).sum(2)
                ac.append(np.mean(np.array(cls)[d.argmin(1)] == yo[te]))
                nn = np.abs(to[te][:, None] - to[cal][None]).argmin(1)
                at.append(np.mean(yo[cal][nn] == yo[te]))
            out[f"csi{k}"], out[f"time{k}"] = 100 * np.mean(ac), 100 * np.mean(at)
        ii, jj = np.triu_indices(len(idx), 1)
        dX = np.linalg.norm(Z[ii] - Z[jj], axis=1); gap = np.abs(to[ii] - to[jj])
        cs, cg = partial_coefs(dX, yo[ii] == yo[jj], gap)
        nulls = np.array([partial_coefs(dX, (lambda yp: yp[ii] == yp[jj])(rng.permutation(yo)), gap)[0] for _ in range(a.perms // 2)])
        out.update(coef_same=cs, coef_gap=cg, same_p=(1 + (nulls <= cs).sum()) / (1 + len(nulls)))
        rows.append((u, out))
    print("\n" + f"{'session':<10}{'grouped p':>10}" + "".join(f"{f'CSI k={k}':>10}{f'time k={k}':>10}" for k in a.ks)
          + f"{'coef_same':>11}{'p':>7}{'coef_gap':>10}")
    for u, o in rows:
        print(f"{u:<10}{o['grouped_p']:>10.3f}" + "".join(f"{o[f'csi{k}']:>9.1f}%{o[f'time{k}']:>9.1f}%" for k in a.ks)
              + f"{o['coef_same']:>+11.3f}{o['same_p']:>7.3f}{o['coef_gap']:>+10.3f}")
    print("\ngrouped p small = classes were recorded in blocks. time = accuracy WITHOUT any CSI. chance 20%.")
    print("If 'time' is about as good as 'CSI' in a session, that session's class signal cannot be told apart from drift.")


if __name__ == "__main__":
    main()