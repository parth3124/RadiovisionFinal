#!/usr/bin/env python3
"""
audit_confounds.py - do NON-signal properties of a recording (packet-type mix, loss, rate, timing jitter, RSSI level, recording order)
depend on the label? If they do, any classifier can exploit them without using the room's CSI.

    python audit_confounds.py                      # reads outputs/audit/audit_summary.csv (+ the meta.json next to each recording)
    python audit_confounds.py --summary path.csv --perms 2000

1. medians per session x label (readable table)
2. within-session permutation tests (labels shuffled WITHIN each session, so session offsets cannot explain a result):
     'label'  : does the variable differ between the 7 labels?          'person': person vs not-person (empty+obj)
     'zone'   : does it differ between Z1/Z2/Z3 among person recordings? 'time' : |Spearman| with protocol line (recording order)
   statistic = share of rank variance explained by the grouping (eta^2); p = share of shuffles >= observed
3. nuisance-only classifiers: leave-one-session-out balanced accuracy using ONLY these non-CSI variables (needs scikit-learn),
   with a within-session permutation p-value. Chance: person-vs-not 50%, person zone 33%.
Read: small p / accuracy above chance = that property carries label information by itself -> a confound to control or remove,
not evidence about the room. Descriptive: 3 sessions, 132 recordings.
"""
import argparse, csv, json, os
import numpy as np

VARS = ["pct_len128", "pct_len256", "loss_pct", "rate_ts_hz", "host_resid_p95_ms", "ts_dt_p95_over_p50", "rssi_mean", "rssi_sd"]
NUIS_NO_RSSI = ["pct_len128", "pct_len256", "loss_pct", "rate_ts_hz", "host_resid_p95_ms", "ts_dt_p95_over_p50"]


def ranks(x):
    x = np.asarray(x, float); u, inv = np.unique(x, return_inverse=True)
    first = np.cumsum(np.r_[0, np.bincount(inv)])[:-1]; cnt = np.bincount(inv)
    return (first + (cnt + 1) / 2.0)[inv]


def eta2(x, g):
    r = ranks(x); sst = ((r - r.mean()) ** 2).sum()
    if sst == 0:
        return 0.0
    ssb = sum(((r[g == k].mean() - r.mean()) ** 2) * (g == k).sum() for k in np.unique(g))
    return float(ssb / sst)


def spear(x, t):
    rx, rt = ranks(x), ranks(t)
    if rx.std() == 0 or rt.std() == 0:          # constant variable (e.g. no 128-type packets in r03): no association
        return 0.0
    return float(abs(np.corrcoef(rx, rt)[0, 1]))


def perm_p(stat, x, g, sess, rng, perms, per_session):
    """within-session permutation p for sum over sessions (per_session=False) or each session (True)"""
    def total(gg):
        return {s: stat(x[sess == s], gg[sess == s]) for s in np.unique(sess)}
    obs = total(g); cnt = {s: 0 for s in obs}; cnt_all = 0; obs_all = np.mean(list(obs.values()))
    for _ in range(perms):
        gp = g.copy()
        for s in np.unique(sess):
            idx = np.where(sess == s)[0]; gp[idx] = g[rng.permutation(idx)]
        t = total(gp)
        for s in obs:
            cnt[s] += t[s] >= obs[s]
        cnt_all += np.mean(list(t.values())) >= obs_all
    return {s: (1 + cnt[s]) / (1 + perms) for s in obs}, (1 + cnt_all) / (1 + perms), obs


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r.get("error"):
            continue
        d = {k: r.get(k) for k in ("session", "label", "condition", "zone", "path")}
        for v in VARS:
            d[v] = float(r[v])
        mp = os.path.join(os.path.dirname(r["path"]), "meta.json")
        d["line"] = float(json.load(open(mp)).get("protocol_line", np.nan)) if os.path.exists(mp) else np.nan
        rows.append(d)
    return rows


def classify_loso(X, y, sess):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    pred = np.full(len(y), -1)
    for s in np.unique(sess):
        te = sess == s; tr = ~te
        if len(np.unique(y[tr])) < len(np.unique(y)):
            continue
        pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr]).predict(X[te])
    ok = pred >= 0
    return balanced_accuracy_score(y[ok], pred[ok]) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "audit", "audit_summary.csv"))
    ap.add_argument("--perms", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); R = load(a.summary)
    sess = np.array([str(r["session"]) for r in R]); cond = np.array([str(r["condition"]) for r in R]); zone = np.array([str(r["zone"] or "") for r in R])
    lab = np.array([str(r["label"]) for r in R]); line = np.array([r["line"] for r in R]); X = {v: np.array([r[v] for r in R]) for v in VARS}
    out = [f"{len(R)} recordings, sessions {sorted(set(map(str, sess)))}, labels {sorted(set(map(str, lab)))}"]
    out.append("\nMEDIANS per session x label  (pct_len128 | loss_pct | rate_ts_hz | rssi_mean)")
    for s in sorted(set(sess)):
        out.append(f"  {s}")
        for l in sorted(set(lab)):
            m = (sess == s) & (lab == l)
            if m.any():
                out.append(f"    {l:10s} n={m.sum()}  " + " | ".join(f"{np.median(X[v][m]):7.1f}" for v in ("pct_len128", "loss_pct", "rate_ts_hz", "rssi_mean")))
    out.append(f"\nWITHIN-SESSION PERMUTATION TESTS ({a.perms} shuffles). cells: eta^2 (share of rank variance explained) and p per session r01/r02/r03 | pooled p")
    sess_list = sorted(set(sess)); per = np.array([r["condition"] == "person" for r in R])
    tests = [("label (7 groups)", np.ones(len(R), bool), lab), ("person vs not", np.ones(len(R), bool), np.where(per, "P", "N")),
             ("zone (person only)", per, zone)]
    for name, mask, g in tests:
        out.append(f"\n  {name}")
        for v in VARS:
            p_s, p_all, obs = perm_p(eta2, X[v][mask], g[mask], sess[mask], rng, a.perms, True)
            out.append(f"    {v:20s} " + "  ".join(f"{s}: {obs[s]:.2f} p={p_s[s]:.3f}" for s in sorted(obs)) + f" | pooled p={p_all:.3f}" + ("   <==" if p_all < 0.01 else ""))
    out.append("\n  recording order (|Spearman| of variable with protocol line, within session)")
    for v in VARS:
        p_s, p_all, obs = perm_p(lambda x, t: spear(x, t), X[v], line, sess, rng, a.perms, True)
        out.append(f"    {v:20s} " + "  ".join(f"{s}: {obs[s]:.2f} p={p_s[s]:.3f}" for s in sorted(obs)) + f" | pooled p={p_all:.3f}" + ("   <==" if p_all < 0.01 else ""))
    try:
        import sklearn  # noqa
        out.append(f"\nNUISANCE-ONLY CLASSIFIERS (leave-one-session-out, balanced accuracy; permutation p within sessions, {min(a.perms, 200)} shuffles)")
        for tname, mask, y in (("person vs not-person", np.ones(len(R), bool), per.astype(int)), ("person zone (3-class)", per, np.array([int(z[1]) - 1 if z else -1 for z in zone]))):
            for fname, feats in (("packet/timing only", NUIS_NO_RSSI), ("packet/timing + RSSI", NUIS_NO_RSSI + ["rssi_mean", "rssi_sd"]), ("RSSI only", ["rssi_mean", "rssi_sd"])):
                Xm = np.column_stack([X[f][mask] for f in feats]); ym = y[mask]; sm = sess[mask]
                obs = classify_loso(Xm, ym, sm); cnt = 0; P = min(a.perms, 200)
                for _ in range(P):
                    yp = ym.copy()
                    for s in np.unique(sm):
                        idx = np.where(sm == s)[0]; yp[idx] = ym[rng.permutation(idx)]
                    cnt += classify_loso(Xm, yp, sm) >= obs
                out.append(f"  {tname:24s} {fname:22s} bal.acc {100 * obs:5.1f}%  (chance {100 / len(np.unique(ym)):.0f}%)  p={(1 + cnt) / (1 + P):.3f}")
    except ImportError:
        out.append("\n(scikit-learn not installed: classifier part skipped)")
    txt = "\n".join(out); print(txt)
    p = os.path.join(os.path.dirname(a.summary), "confounds_report.txt"); open(p, "w").write(txt); print("\nsaved", p)


if __name__ == "__main__":
    main()