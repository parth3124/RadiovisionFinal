#!/usr/bin/env python3
"""
overhead_eval.py - CSI vs overhead-camera ground truth: presence, motion, zone and (x, y) position/trajectory.

    python overhead_eval.py --data dataset_over --within     # GATE: leave-one-TAKE-out inside each session (use after your first session)
    python overhead_eval.py --data dataset_over              # real test: leave-one-SESSION-out (needs >= 2 sessions)
    python overhead_eval.py --data dataset_over --lag 0.4 --smooth 5

DATA: <data>/<label>/<session>_<NN>_<label>/ with csi_<link>.npz, track.npz, meta.json (written by overhead_collect.py).

GROUND TRUTH per window (2 s, stride 1 s), from the camera track only:
  presence  cap visible in >= 50% of frames (occupied) / <= 5% (empty); in between = excluded
  motion    occupied and visible >= 80%: median speed >= 0.30 m/s moving, <= 0.10 m/s still; in between excluded
  zone      occupied, visible >= 90%, zone constant in >= 95% of frames (grid from calib.json); transitions excluded
  position  occupied, visible >= 90%: mean (x, y) in metres
FEATURES (everything fitted on TRAINING takes only): M1 = per link per subcarrier [mean, std] of packet-normalized log amplitude over
the window (standardized); M0 = per link [mean log-power, std, mean RSSI, std RSSI]; M1G = M1 + M0.
MODELS (frozen): presence/motion -> LogisticRegression on M0;  zone -> LogisticRegression on M1G;  position -> Ridge on M1G.
SCORES: balanced accuracy for classification; mean position error in metres for position, against the baseline that always predicts
the training-set mean position (and zone-from-predicted-position accuracy).
SIGNIFICANCE: block permutation: each session's whole label stream (all its takes in time order) is circularly SHIFTED by a random
offset (keeps block structure and autocorrelation, breaks alignment with the CSI), whole procedure refit. Overlapping windows are NOT independent, so there is no
analytic p-value; the permutation null is the reference.
"""
import argparse, glob, json, os
from collections import Counter
import numpy as np
import common as C
import overhead_common as OC

FS = 50.0
WIN_SEC, STRIDE_SEC = 2.0, 1.0
WIN, STRIDE = int(WIN_SEC * FS), int(STRIDE_SEC * FS)
TRIM, MIN_PACKETS, LIVE_FRAC = 0.5, 200, 0.05


# ------------------------------------------------------------------ loading
def read_link(path):
    z = np.load(path)
    lens = z["csi_len"]; L = Counter(lens.tolist()).most_common(1)[0][0]; keep = lens == L
    t = z["host_t"][keep].astype(np.float64); csi = z["csi"][keep, :L]
    rssi = z["rssi"][keep].astype(np.float32) if "rssi" in z.files else np.zeros(len(t), np.float32)
    o = np.argsort(t, kind="stable"); t, csi, rssi = t[o], csi[o], rssi[o]
    inc = np.concatenate([[True], np.diff(t) > 0]); t, csi, rssi = t[inc], csi[inc], rssi[inc]
    m = (t >= t[0] + TRIM) & (t <= t[-1] - TRIM)
    if m.sum() < MIN_PACKETS:
        return None
    return t[m], C.csi_amplitude(csi[m]), rssi[m]


class Take:
    pass


def load_takes(root):
    takes, skipped = [], []
    for mp in sorted(glob.glob(os.path.join(root, "*", "*", "meta.json"))):
        d = os.path.dirname(mp); meta = json.load(open(mp))
        if meta.get("kind") != "overhead":
            continue
        if not os.path.exists(os.path.join(d, "track.npz")):
            skipped.append((d, "no track.npz")); continue
        links = {}
        for f in sorted(glob.glob(os.path.join(d, "csi_*.npz"))):
            r = read_link(f)
            if r is None:
                skipped.append((d, f"{os.path.basename(f)}: too few packets")); links = {}; break
            links[os.path.basename(f)[4:-4]] = r
        if not links:
            continue
        tk = Take(); tk.id = os.path.basename(d); tk.session = meta["session"]; tk.label = meta["label"]; tk.links = links
        tr = np.load(os.path.join(d, "track.npz")); tk.track = {k: tr[k] for k in tr.files}
        tk.sha = meta.get("calibration_sha"); takes.append(tk)
    if not takes:
        raise SystemExit(f"no overhead takes under {root}/")
    cs = Counter(tuple(sorted(t.links)) for t in takes).most_common(1)[0][0]
    for t in takes:
        if tuple(sorted(t.links)) != cs:
            skipped.append((t.id, f"link set {sorted(t.links)} differs from {list(cs)}"))
    return [t for t in takes if tuple(sorted(t.links)) == cs], list(cs), skipped


def resample(t, X, grid):
    j = np.clip(np.searchsorted(t, grid, side="right"), 1, len(t) - 1)
    w = ((grid - t[j - 1]) / np.maximum(t[j] - t[j - 1], 1e-12))[:, None]
    return X[j - 1] * (1 - w) + X[j] * w


# ------------------------------------------------------------------ ground truth
def take_grid(tk):
    start = max(v[0][0] for v in tk.links.values()); end = min(v[0][-1] for v in tk.links.values())
    return np.arange(start, end, 1.0 / FS)


def window_truth(tk, calib, lag):
    """-> dict of arrays over windows: i0 (start index on the take grid), occ, mov, zone, pos(x,y)"""
    grid = take_grid(tk); n = len(grid)
    i0s = np.arange(0, n - WIN + 1, STRIDE)
    tt = tk.track["t"] + lag; vis = tk.track["visible"]; x, y = tk.track["x"], tk.track["y"]
    area, zones = tuple(calib["area"]), tuple(calib["zones"])
    occ = np.full(len(i0s), -1); mov = np.full(len(i0s), -1); zone = np.full(len(i0s), -1); pos = np.full((len(i0s), 2), np.nan)
    for k, i0 in enumerate(i0s):
        a, b = grid[i0], grid[i0] + WIN_SEC
        m = (tt >= a) & (tt < b)
        if m.sum() < 5:
            continue
        fv = vis[m].mean()
        if fv <= 0.05:
            occ[k] = 0; continue
        if fv < 0.5:
            continue
        occ[k] = 1
        xs, ys, ts = x[m][vis[m]], y[m][vis[m]], tt[m][vis[m]]
        if len(xs) >= 3 and fv >= 0.8:
            dt = np.diff(ts); ok = (dt > 0) & (dt < 0.25)
            if ok.sum() >= 2:
                sp = np.median(np.hypot(np.diff(xs), np.diff(ys))[ok] / dt[ok])
                mov[k] = 1 if sp >= 0.30 else (0 if sp <= 0.10 else -1)
        if fv >= 0.9 and len(xs) >= 3:
            zs = np.array([OC.zone_index(a_, b_, area, zones) for a_, b_ in zip(xs, ys)])
            cnt = Counter(zs.tolist()); zbest, nbest = cnt.most_common(1)[0]
            if zbest >= 0 and nbest / len(zs) >= 0.95:
                zone[k] = zbest
            pos[k] = [xs.mean(), ys.mean()]
    return {"i0": i0s, "occ": occ, "mov": mov, "zone": zone, "pos": pos}


# ------------------------------------------------------------------ features (fitted on training takes only)
def build_fold(takes, names, tr, te, truth, session_norm=False):
    live = {}
    for n in names:
        m = np.mean([takes[i].links[n][1].mean(0) for i in tr], 0); live[n] = m > LIVE_FRAC * np.median(m)
    raw = {}
    for i in list(tr) + list(te):
        tk = takes[i]; g = take_grid(tk); per = {}
        for n in names:
            t, A, rssi = tk.links[n]
            Lg = np.log1p(A); Lg = Lg - Lg[:, live[n]].mean(1, keepdims=True)
            mA = A.mean(0); lv = mA > LIVE_FRAC * np.median(mA)
            per[n] = (resample(t, Lg[:, live[n]], g), resample(t, np.log1p(A)[:, lv].mean(1, keepdims=True), g)[:, 0],
                      resample(t, rssi[:, None], g)[:, 0])
        raw[i] = per
    st = {n: (np.concatenate([raw[i][n][0] for i in tr]).mean(0), np.concatenate([raw[i][n][0] for i in tr]).std(0) + 1e-3) for n in names}
    out = {}
    for i in list(tr) + list(te):
        M1, M0 = [], []
        for n in names:
            X = np.clip((raw[i][n][0] - st[n][0]) / st[n][1], -8, 8); lev, rs = raw[i][n][1], raw[i][n][2]
            i0s = truth[i]["i0"]
            M1.append(np.array([np.concatenate([X[a:a + WIN].mean(0), X[a:a + WIN].std(0)]) for a in i0s]))
            M0.append(np.array([[lev[a:a + WIN].mean(), lev[a:a + WIN].std(), rs[a:a + WIN].mean(), rs[a:a + WIN].std()] for a in i0s]))
        out[i] = {"M1": np.hstack(M1), "M0": np.hstack(M0)}
    if session_norm:       # z-score every feature within its own session (unlabeled statistics of ALL takes of that session)
        for s_ in {takes[i].session for i in out}:
            ids = [i for i in out if takes[i].session == s_]
            for key in ("M1", "M0"):
                cat = np.vstack([out[i][key] for i in ids]); mu, sd = cat.mean(0), cat.std(0) + 1e-6
                for i in ids:
                    out[i][key] = np.clip((out[i][key] - mu) / sd, -8, 8)
    for i in out:
        out[i]["M1G"] = np.hstack([out[i]["M1"], out[i]["M0"]])
    return out


def clf():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))


def reg():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return make_pipeline(StandardScaler(), Ridge(alpha=300.0))


PRIMARY = {"presence": "M0", "motion": "M0", "zone": "M1G", "position": "M1G"}
KEY = {"presence": "occ", "motion": "mov", "zone": "zone", "position": "pos"}


def evaluate(task, fset, folds, truth, smooth=1):
    """-> score (+ extras). folds: list of (tr_idx, te_idx, F_fold). classification: balanced accuracy; position: mean error (m)."""
    from sklearn.metrics import balanced_accuracy_score
    yt_all, yp_all, base_all, meta_all = [], [], [], []
    per = []
    for tr, te, Ff in folds:
        def collect(idx):
            X, Y, G = [], [], []
            for i in idx:
                y = truth[i][KEY[task]]
                ok = np.isfinite(y).all(1) if task == "position" else (y >= 0)
                X.append(Ff[i][fset][ok]); Y.append(y[ok]); G.append(np.full(int(ok.sum()), i))
            return np.vstack(X), np.concatenate(Y) if task != "position" else np.vstack(Y), np.concatenate(G)
        Xtr, ytr, _ = collect(tr); Xte, yte, gte = collect(te)
        if len(Xtr) < 20 or len(Xte) < 5:
            continue
        if task == "position":
            p = reg().fit(Xtr, ytr).predict(Xte)
            if smooth > 1:
                for g in np.unique(gte):
                    m = gte == g
                    if m.sum() >= smooth:
                        pm = p[m].copy(); h = smooth // 2
                        for k in range(len(pm)):
                            pm[k] = np.median(p[m][max(0, k - h):k + h + 1], axis=0)
                        p[m] = pm
            err = np.hypot(*(p - yte).T); base = np.hypot(*(ytr.mean(0) - yte).T)
            yt_all.append(yte); yp_all.append(p); base_all.append(base); per.append(float(err.mean()))
        else:
            if len(set(ytr)) < 2:
                continue
            p = clf().fit(Xtr, ytr).predict(Xte)
            yt_all.append(yte); yp_all.append(p); per.append(float(balanced_accuracy_score(yte, p)))
    if not yt_all:
        return None
    if task == "position":
        yt, yp = np.vstack(yt_all), np.vstack(yp_all)
        return {"score": float(np.hypot(*(yp - yt).T).mean()), "baseline": float(np.concatenate(base_all).mean()), "per": per, "yt": yt, "yp": yp}
    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    return {"score": float(balanced_accuracy_score(yt, yp)), "per": per, "n": len(yt)}


def permuted_truth(truth, takes, rng):
    """null for the permutation test: per session, concatenate the window labels of all takes in time order and roll the stream by a
    random offset (>= 10% of its length). Keeps each label stream's block structure and autocorrelation (also take-level labels such as
    'occupied'), breaks its alignment with the CSI. The same offset is used for all tasks."""
    out = [dict(t) for t in truth]
    for s_ in {t.session for t in takes}:
        idx = sorted((i for i, t in enumerate(takes) if t.session == s_), key=lambda i: take_grid(takes[i])[0])
        n = sum(len(truth[i]["occ"]) for i in idx); lo = max(10, n // 10)
        off = int(rng.integers(lo, max(n - lo, lo + 1)))
        for key in ("occ", "mov", "zone", "pos"):
            cat = np.roll(np.concatenate([truth[i][key] for i in idx], axis=0), off, axis=0); k = 0
            for i in idx:
                m = len(truth[i][key]); out[i][key] = cat[k:k + m]; k += m
    return out


def estimate_lag(takes, names, truth):
    """CSI motion energy vs camera speed, cross-correlated over +-1.5 s. Positive = CSI reacts LATER than the camera."""
    res = []
    for i, tk in enumerate(takes):
        if tk.label == "empty":
            continue
        g = take_grid(tk); n = len(g); e = np.zeros(n)
        for nme in names:
            t, A, _ = tk.links[nme]; Lg = np.log1p(A); mA = A.mean(0); lv = mA > LIVE_FRAC * np.median(mA)
            Lg = Lg - Lg[:, lv].mean(1, keepdims=True); X = resample(t, Lg[:, lv], g)
            e[5:] += np.abs(X[5:] - X[:-5]).mean(1) / len(names)
        tt, vis, x, y = tk.track["t"], tk.track["visible"], tk.track["x"], tk.track["y"]
        sp = np.zeros(len(tt)); ok = vis[1:] & vis[:-1]
        dt = np.maximum(np.diff(tt), 1e-3); sp[1:] = np.where(ok, np.hypot(np.diff(x), np.diff(y)) / dt, 0)
        speed = np.interp(g, tt, np.nan_to_num(sp))
        k = int(0.4 * FS); sm = lambda v: np.convolve(v, np.ones(k) / k, mode="same")
        zm, zs = sm(e), sm(speed); zm = (zm - zm.mean()) / (zm.std() + 1e-9); zs = (zs - zs.mean()) / (zs.std() + 1e-9)
        lags = np.arange(-int(1.5 * FS), int(1.5 * FS) + 1)
        xc = [np.mean(zm[max(0, l):n + min(0, l)] * zs[max(0, -l):n - max(0, l)]) for l in lags]
        res.append((lags[int(np.argmax(xc))] / FS, float(np.max(xc))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_over")
    ap.add_argument("--calib")
    ap.add_argument("--within", action="store_true")
    ap.add_argument("--lag", type=float, default=0.0, help="seconds added to camera timestamps (see the lag report)")
    ap.add_argument("--smooth", type=int, default=1, help="moving-median window (in 1 s windows) applied to predicted positions")
    ap.add_argument("--session-norm", dest="session_norm", action="store_true",
                    help="EXPLORATORY: z-score features within each session using that session's unlabeled data (needs unlabeled data at the new site)")
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    takes, names, skipped = load_takes(a.data)
    calib = OC.load_calib(a.calib or os.path.join(a.data, "calib.json"))
    shas = {t.sha for t in takes}
    print(f"{len(takes)} takes, links {names}, sessions {sorted({t.session for t in takes})}, zones grid {calib['zones']} over {calib['area']} m")
    if len(shas) > 1:
        print(f"  WARNING: takes were recorded with {len(shas)} different calibrations {shas}: positions are not comparable")
    for d, why in skipped:
        print(f"  SKIPPED {d}: {why}")
    truth = [window_truth(t, calib, a.lag) for t in takes]
    sessions = sorted({t.session for t in takes})
    print(f"\n{'session':<8}{'takes':>6}{'windows':>9}{'occupied':>10}{'empty':>7}{'moving':>8}{'still':>7}{'zone-ok':>9}")
    for s in sessions:
        ix = [i for i, t in enumerate(takes) if t.session == s]
        cat = lambda k, v: int(sum((truth[i][k] == v).sum() for i in ix))
        print(f"{s:<8}{len(ix):>6}{sum(len(truth[i]['occ']) for i in ix):>9}{cat('occ', 1):>10}{cat('occ', 0):>7}{cat('mov', 1):>8}{cat('mov', 0):>7}"
              f"{int(sum((truth[i]['zone'] >= 0).sum() for i in ix)):>9}")
    lg = estimate_lag(takes, names, truth)
    if lg:
        L = np.array(lg); print(f"\ncamera/CSI lag (CSI motion energy vs camera speed): median {np.median(L[:, 0]):+.2f} s, median peak correlation "
                               f"{np.median(L[:, 1]):.2f} over {len(L)} takes. Positive = CSI reacts later than the camera.")
        if np.median(L[:, 1]) >= 0.2 and abs(np.median(L[:, 0])) > 0.2 and a.lag == 0:
            print(f"  -> consider re-running with --lag {np.median(L[:, 0]):+.2f}")
        elif np.median(L[:, 1]) < 0.2:
            print("  (weak correlation: the lag estimate is not reliable, do not apply it)")

    # folds: (train take indices, test take indices)
    if a.within:
        units = []
        for s_ in sessions:
            ix = [i for i, t in enumerate(takes) if t.session == s_]
            if len(ix) >= 3:
                units.append((s_, [([j for j in ix if j != i], [i]) for i in ix]))
        mode = "within-session leave-one-take-out (GATE, not the final test)"
    else:
        if len(sessions) < 2:
            raise SystemExit("need >= 2 sessions; use --within after the first session")
        mode = "leave-one-session-out"
        units = [("all", [([i for i, t in enumerate(takes) if t.session != s_], [i for i, t in enumerate(takes) if t.session == s_])
                          for s_ in sessions])]
    print(f"\n=== mode: {mode}; classification = balanced accuracy, position = mean error in metres (lower is better) ===")
    print(f"{'unit':<8}{'task':<10}{'score':>9}{'baseline':>10}{'null95':>9}{'p':>8}   per held-out fold")
    for unit, fl in units:
        folds = []
        for tr, te in fl:
            folds.append((tr, te, build_fold(takes, names, tr, te, truth, a.session_norm)))
        for task in ("presence", "motion", "zone", "position"):
            r = evaluate(task, PRIMARY[task], folds, truth, smooth=a.smooth if task == "position" else 1)
            if r is None:
                print(f"{unit:<8}{task:<10}  not enough labelled windows"); continue
            nulls = []
            for _ in range(a.perms):
                rr = evaluate(task, PRIMARY[task], folds, permuted_truth(truth, takes, rng), smooth=a.smooth if task == "position" else 1)
                if rr:
                    nulls.append(rr["score"])
            nulls = np.array(nulls)
            area, zones = tuple(calib["area"]), tuple(calib["zones"])
            if task == "position":
                p = (1 + (nulls <= r["score"]).sum()) / (1 + len(nulls))
                sc, base, n95s = f"{r['score']:.2f} m", f"{r['baseline']:.2f} m", f"{np.percentile(nulls, 5):.2f} m"
                zacc = np.mean([OC.zone_index(x, y, area, zones) == OC.zone_index(xt, yt, area, zones) for (x, y), (xt, yt) in zip(r["yp"], r["yt"])])
                fold_txt = " ".join(f"{v:5.2f}m" for v in r["per"]) + f"   zone-from-position {100 * zacc:.0f}%   SKILL vs predict-the-mean {100 * (1 - r['score'] / r['baseline']):+.0f}%"
            else:
                p = (1 + (nulls >= r["score"]).sum()) / (1 + len(nulls))
                sc, base, n95s = f"{100 * r['score']:.1f}%", "chance", f"{100 * np.percentile(nulls, 95):.1f}%"
                fold_txt = " ".join(f"{100 * v:4.0f}%" for v in r["per"])
            print(f"{unit:<8}{task:<10}{sc:>9}{base:>10}{n95s:>9}{p:>8.3f}   {fold_txt}")
    print("\nHOW TO READ: classification needs a score above null95 AND small p. Position needs a clearly positive SKILL (error well below the predict-the-mean baseline):")
    print("a model that just predicts the mean also passes the permutation test, because shifted labels give worse-than-mean errors, so p alone is not enough for position.")
    print("Windows overlap and are strongly autocorrelated: trust the permutation null and the per-fold consistency, not the window count.")


if __name__ == "__main__":
    main()
