"""
fewshot_session.py - calibration (few-shot) evaluation across sessions. Torch-free.

Protocol (no leakage): pick a target session. Draw k recordings per class from it as CALIBRATION data;
test on the remaining recordings of the SAME session (never on a calibration recording). Source data =
all other sessions. Repeated over random draws.

Methods
  target_ncm    nearest class mean using only the calibration recordings (no other session used)
  source_only   logistic regression trained on the other sessions (zero-shot, for reference)
  source+cal    logistic regression trained on the other sessions + calibration recordings (up-weighted)

    python fewshot_session.py --data dataset --norm session
Also prints a per-session health table (packet rate, RSSI, malformed counts, amplitude).
NOTE: --norm session computes its statistics from the target session's UNLABELED recordings.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
import common as C
import dataset as D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def health(recs):
    by = defaultdict(list)
    for r in recs:
        by[r.session].append(r)
    print("session health (from meta.json / csi.npz):")
    print(f"{'session':<10}{'recs':>5}{'CSI Hz':>8}{'RSSI':>8}{'malformed':>11}{'mean amp':>10}")
    for u in sorted(by):
        rs = by[u]
        hz = np.mean([r.meta.get("csi_rate_hz", np.nan) for r in rs])
        mal = np.mean([r.meta.get("csi_malformed_total", np.nan) for r in rs])
        rssi = []
        for r in rs:
            p = os.path.join(r.path, "csi.npz")
            try:
                d = np.load(p)
                if "rssi" in d:
                    rssi.append(float(np.mean(d["rssi"])))
            except Exception:
                pass
        amp = np.mean([r.raw_mean.mean() for r in rs])
        print(f"{u:<10}{len(rs):>5}{hz:>8.1f}{(np.mean(rssi) if rssi else float('nan')):>8.1f}{mal:>11.1f}{amp:>10.1f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--norm", choices=["global", "packet", "session"], default="session")
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--cal-weight", type=float, default=10.0)
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    health(recs)
    D.apply_norm(recs, a.norm)
    X = np.array([np.concatenate([r.G[r.valid].astype(np.float32).mean(0), r.G[r.valid].astype(np.float32).std(0)]) for r in recs])
    y = np.array([r.label for r in recs]); s = np.array([r.session for r in recs])
    sessions = sorted(set(s))
    print(f"norm={a.norm}  {len(y)} recordings, calibration weight {a.cal_weight}, {a.draws} random draws per setting\n")
    res = defaultdict(lambda: defaultdict(list))        # (k, method) -> session -> accs
    for T in sessions:
        tgt = np.where(s == T)[0]; src = np.where(s != T)[0]
        zero = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000, class_weight="balanced")).fit(X[src], y[src])
        for k in a.ks:
            for _ in range(a.draws):
                cal = []
                for c in range(len(C.CLASSES)):
                    ids = [i for i in tgt if y[i] == c]
                    kk = min(k, len(ids) - 1)
                    if kk >= 1:
                        cal += list(rng.choice(ids, kk, replace=False))
                cal = np.array(cal); te = np.array([i for i in tgt if i not in set(cal)])
                if len(te) == 0 or len(set(y[cal])) < 2:
                    continue
                # target_ncm: standardize with the target session's own (unlabeled) spread
                sc = StandardScaler().fit(X[tgt])
                cls = sorted(set(y[cal]))
                cent = np.array([sc.transform(X[cal][y[cal] == c]).mean(0) for c in cls])
                d = ((sc.transform(X[te])[:, None, :] - cent[None]) ** 2).sum(2)
                res[(k, "target_ncm")][T].append(np.mean(np.array(cls)[d.argmin(1)] == y[te]))
                res[(k, "source_only")][T].append(np.mean(zero.predict(X[te]) == y[te]))
                idx = np.concatenate([src, cal]); w = np.concatenate([np.ones(len(src)), np.full(len(cal), a.cal_weight)])
                m = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000, class_weight="balanced"))
                m.fit(X[idx], y[idx], logisticregression__sample_weight=w)
                res[(k, "source+cal")][T].append(np.mean(m.predict(X[te]) == y[te]))
    print(f"{'k/class':>8} {'method':<12}" + "".join(f"{u:>10}" for u in sessions) + f"{'MEAN':>9}")
    for k in a.ks:
        for meth in ("source_only", "target_ncm", "source+cal"):
            per = [100 * np.mean(res[(k, meth)][u]) if res[(k, meth)][u] else float("nan") for u in sessions]
            print(f"{k:>8} {meth:<12}" + "".join(f"{v:>9.1f}%" for v in per) + f"{np.nanmean(per):>8.1f}%")
        print()
    print("chance = 20%. Differences under ~8 points are within noise at this sample size.")


if __name__ == "__main__":
    main()