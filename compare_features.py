#!/usr/bin/env python3
"""
compare_features.py - which feature representations carry presence / zone information ACROSS sessions?

    python compare_features.py                                  # outputs/features/all/features.npz, 100 permutations
    python compare_features.py --perms 200 --norm none          # only the calibration-free protocol
    python compare_features.py --sets amp_mean phase_mean rssi  # subset of feature sets

Protocol
  tasks      person_zone (Z1/Z2/Z3 among person recordings), obj_zone, person_presence (person vs empty), obj_presence (object vs empty)
  model      StandardScaler + LogisticRegression(C=0.1, class_weight=balanced)  - the same baseline model for every feature set
  splits     whole SESSIONS: train on one session, test on another (all 6 directions) and leave-one-session-out (train on two, test on the third).
             Scaling and the classifier are fitted on training sessions only. No feature selection, no hyper-parameter search (nothing to leak).
  norm       'none'  = calibration-free (features as extracted: per-packet-normalised amplitude, cpo-sanitised phase)
             'ref'   = TRANSDUCTIVE calibration: each session's empty-room mean is subtracted (phase: wrapped difference); the held-out session's own
                       empty recordings are used (leave-one-out for the empties themselves). Reported separately and labelled - not calibration-free.
  evidence   permutation test: labels shuffled WITHIN each session, whole leave-one-session-out procedure repeated; p = share of shuffles >= observed.
             null95 = 95th percentile of the shuffled scores.
  nuisance   'nuis_only' = packet mix / loss / rate / timing jitter from audit_summary.csv (not CSI at all), as a reference row.
Reading: a feature set is only interesting if it clearly beats chance AND the nuisance row AND the amplitude baseline in the same task and norm,
consistently across directions. ~100 comparisons are made: expect a few p<0.05 by chance alone; trust p<0.01 that repeats.
Outputs: outputs/compare/comparison_results.csv, comparison_report.txt, comparison_details.json (per direction, per class recall, pooled confusion).
"""
import argparse, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SETS = {   # name -> list of (group, blocks or None for all)
    "amp_mean": [("amp_mean", None)], "amp_mean+std": [("amp_mean", None), ("amp_std", None)],
    "phase_mean": [("phase_mean", None)], "phase_mean+std": [("phase_mean", None), ("phase_std", None)],
    "rssi": [("rssi", None)], "amp+phase": [("amp_mean", None), ("phase_mean", None)], "amp+rssi": [("amp_mean", None), ("rssi", None)],
    "amp+phase+rssi": [("amp_mean", None), ("phase_mean", None), ("rssi", None)], "temporal": [("temporal", None)], "spectral": [("spectral", None)],
    "temporal+spectral": [("temporal", None), ("spectral", None)],
    "everything": [(g, None) for g in ("amp_mean", "amp_std", "phase_mean", "phase_std", "rssi", "temporal", "spectral")],
    "amp_b0": [("amp_mean", [0])], "amp_b1": [("amp_mean", [1])], "amp_b2": [("amp_mean", [2])],
    "phase_b0": [("phase_mean", [0])], "phase_b1": [("phase_mean", [1])], "phase_b2": [("phase_mean", [2])],
}
NUIS = ["pct_len128", "pct_len256", "loss_pct", "rate_ts_hz", "host_resid_p95_ms", "ts_dt_p95_over_p50"]
TASKS = ["person_zone", "obj_zone", "person_presence", "obj_presence"]


def load(path):
    z = np.load(path, allow_pickle=False); idx = json.load(open(os.path.join(os.path.dirname(path), "index.json")))
    meta = {k: z[k] for k in ("session", "label", "condition", "zone", "path", "line", "created")}
    G = {k[2:]: z[k] for k in z.files if k.startswith("G_")}
    return meta, G, idx["names"]


def wrap(x):
    return np.angle(np.exp(1j * x))


def apply_ref(G, names, meta):
    """subtract each session's empty-room reference (leave-one-out for empty recordings); phase_mean uses wrapped / circular arithmetic"""
    out = {}; sess, cond = meta["session"], meta["condition"]
    for g, M in G.items():
        R = np.zeros_like(M)
        for s in np.unique(sess):
            emp = np.where((sess == s) & (cond == "empty"))[0]
            if len(emp) < 3:
                raise SystemExit(f"--norm ref needs >= 3 empty recordings in session {s}")
            for i in np.where(sess == s)[0]:
                ref_idx = emp[emp != i]
                R[i] = np.angle(np.exp(1j * M[ref_idx]).mean(0)) if g == "phase_mean" else M[ref_idx].mean(0)
        out[g] = wrap(M - R) if g == "phase_mean" else M - R
    return out


def build(G, names, spec):
    cols = []
    for g, blocks in spec:
        if g not in G:
            return None
        M = G[g]; nm = names[g]
        keep = [j for j, n in enumerate(nm) if blocks is None or any(n.startswith(f"b{b}_") for b in blocks)]
        if not keep:
            return None
        cols.append(M[:, keep])
    return np.hstack(cols)


def task_data(task, meta):
    cond, zone = meta["condition"], meta["zone"]; kind, what = task.split("_")[0], task.split("_")[1]
    if what == "zone":
        sel = np.where((cond == kind) & np.isin(zone, ["Z1", "Z2", "Z3"]))[0]; y = np.array([int(str(zone[i])[1]) - 1 for i in sel])
    else:
        sel = np.where(np.isin(cond, ["empty", kind]))[0]; y = (cond[sel] == kind).astype(int)
    return sel, y


def model():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))


def bal_acc(y, p, k):
    r = [np.mean(p[y == c] == c) for c in range(k) if (y == c).any()]
    return float(np.mean(r)), [float(v) for v in r]


def fit_pred(Xtr, ytr, Xte):
    return model().fit(Xtr, ytr).predict(Xte)


def loso(X, y, sess, k):
    accs, pooled_y, pooled_p = [], [], []
    for s in np.unique(sess):
        te = sess == s; tr = ~te
        if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < 2:
            continue
        p = fit_pred(X[tr], y[tr], X[te]); accs.append(bal_acc(y[te], p, k)[0]); pooled_y.append(y[te]); pooled_p.append(p)
    return accs, (np.concatenate(pooled_y), np.concatenate(pooled_p)) if accs else (None, None)


def pairs(X, y, sess, k):
    res = {}
    for a in np.unique(sess):
        for b in np.unique(sess):
            if a == b:
                continue
            tr, te = sess == a, sess == b
            if len(np.unique(y[tr])) < k or len(np.unique(y[te])) < 2:
                continue
            p = fit_pred(X[tr], y[tr], X[te]); res[f"{a}->{b}"] = bal_acc(y[te], p, k)
    return res


def evaluate(X, y, sess, k, perms, rng):
    accs, (py, pp) = loso(X, y, sess, k)
    if not accs:
        return None
    obs = float(np.mean(accs)); pr = pairs(X, y, sess, k); null = []
    for _ in range(perms):
        yp = y.copy()
        for s in np.unique(sess):
            idx = np.where(sess == s)[0]; yp[idx] = y[rng.permutation(idx)]
        a, _ = loso(X, yp, sess, k)
        if a:
            null.append(np.mean(a))
    null = np.array(null)
    return dict(loso=obs, loso_per_session=[float(a) for a in accs], pairs_mean=float(np.mean([v[0] for v in pr.values()])) if pr else float("nan"),
                pairs={d: v[0] for d, v in pr.items()}, pairs_recall={d: v[1] for d, v in pr.items()},
                p=float((1 + (null >= obs).sum()) / (1 + len(null))) if len(null) else float("nan"), null95=float(np.percentile(null, 95)) if len(null) else float("nan"),
                confusion=np.array([[int(((py == i) & (pp == j)).sum()) for j in range(k)] for i in range(k)]).tolist())


def nuisance_matrix(meta, path):
    if not os.path.exists(path):
        return None
    rows = {os.path.abspath(r["path"]): r for r in csv.DictReader(open(path)) if not r.get("error")}
    try:
        return np.array([[float(rows[os.path.abspath(p)][c]) for c in NUIS] for p in meta["path"]])
    except KeyError:
        return None


def run(features, perms, norms, sets, audit, seed=0, tasks=TASKS, verbose=True):
    meta, G0, names = load(features); rng = np.random.default_rng(seed); results = []
    nm = nuisance_matrix(meta, audit)
    for norm in norms:
        G = G0 if norm == "none" else apply_ref(G0, names, meta)
        for task in tasks:
            sel, y = task_data(task, meta); k = len(np.unique(y)); sess = meta["session"][sel]
            if len(np.unique(sess)) < 2 or k < 2:
                continue
            cand = {s: build(G, names, SETS[s]) for s in sets}
            if nm is not None:
                cand["nuis_only"] = nm
            for sname, Xall in cand.items():
                if Xall is None:
                    continue
                r = evaluate(Xall[sel], y, sess, k, perms, rng)
                if r:
                    r.update(task=task, set=sname, norm=norm, chance=1.0 / k, n=len(y), dim=int(Xall.shape[1])); results.append(r)
                    if verbose:
                        print(f"  {norm:4s} {task:16s} {sname:18s} LOSO {100 * r['loso']:5.1f}%  p={r['p']:.3f}", flush=True)
    return results


def report(results):
    out = []
    for norm in ("none", "ref"):
        R = [r for r in results if r["norm"] == norm]
        if not R:
            continue
        out.append(f"\n{'=' * 118}\nNORM = {norm}" + ("   (calibration-free)" if norm == "none" else "   (TRANSDUCTIVE: each session's own empty-room recordings are used as reference)"))
        for task in TASKS:
            T = [r for r in R if r["task"] == task]
            if not T:
                continue
            base = next((r["loso"] for r in T if r["set"] == "amp_mean"), float("nan")); nu = next((r["loso"] for r in T if r["set"] == "nuis_only"), float("nan"))
            out.append(f"\n{task}   chance {100 * T[0]['chance']:.1f}%   n={T[0]['n']}   amplitude baseline {100 * base:.1f}%   nuisance-only {100 * nu:.1f}%")
            out.append(f"  {'feature set':18s}{'dim':>5s}{'LOSO bal.acc':>14s}{'pairs mean':>12s}{'null95':>9s}{'p':>8s}{'vs amp':>9s}   per held-out session")
            for r in sorted(T, key=lambda r: -r["loso"]):
                out.append(f"  {r['set']:18s}{r['dim']:>5d}{100 * r['loso']:>13.1f}%{100 * r['pairs_mean']:>11.1f}%{100 * r['null95']:>8.1f}%{r['p']:>8.3f}{100 * (r['loso'] - base):>+8.1f}   "
                           + " ".join(f"{100 * a:4.0f}" for a in r["loso_per_session"]) + ("   <==" if r["p"] < 0.01 and r["loso"] > max(base, nu) + 0.05 else ""))
    out.append("\n'<==' marks p<0.01 AND LOSO accuracy > both the amplitude baseline and the nuisance-only row by at least 5 points. "
               "With ~100 comparisons, treat anything that does not repeat across tasks/directions with suspicion.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--features", default=os.path.join(here, "outputs", "features", "all", "features.npz"))
    ap.add_argument("--audit", default=os.path.join(here, "outputs", "audit", "audit_summary.csv")); ap.add_argument("--out", default=os.path.join(here, "outputs", "compare"))
    ap.add_argument("--perms", type=int, default=100); ap.add_argument("--norm", choices=["none", "ref", "both"], default="both"); ap.add_argument("--sets", nargs="+", default=list(SETS))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = run(a.features, a.perms, ["none", "ref"] if a.norm == "both" else [a.norm], a.sets, a.audit)
    txt = report(res); print(txt)
    with open(os.path.join(a.out, "comparison_results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["norm", "task", "feature_set", "dim", "n", "chance", "loso_bal_acc", "pairs_mean_bal_acc", "null95", "p"] + ["held_out_" + str(i) for i in range(3)])
        for r in res:
            w.writerow([r["norm"], r["task"], r["set"], r["dim"], r["n"], round(r["chance"], 4), round(r["loso"], 4), round(r["pairs_mean"], 4), round(r["null95"], 4), round(r["p"], 4)] + [round(v, 4) for v in r["loso_per_session"]])
    json.dump(res, open(os.path.join(a.out, "comparison_details.json"), "w"), indent=1); open(os.path.join(a.out, "comparison_report.txt"), "w").write(txt); print("\nsaved", a.out)


if __name__ == "__main__":
    main()
