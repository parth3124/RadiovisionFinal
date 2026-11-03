#!/usr/bin/env python3
"""
audit_timeline.py - is the packet-type 'regime' tied to the LABEL or to the CLOCK?

    python audit_timeline.py                   # uses outputs/audit/audit_summary.csv and the `created` time in each meta.json

Regime A = recording with >= 3% length-128 packets, regime B = fewer (these also differ in loss and rate, see audit_confounds.py).
For each session, in TRUE recording order (meta.json `created`, not the protocol line):
  - the gaps between recordings (pauses / restarts of the recorder show up as long gaps)
  - the A/B sequence with labels
  - permutation test of the number of A/B runs: few runs = regime clustered in time (clock/run effect);
    about as many as chance = regime not explained by time
  - the same regime sequence split by label, to compare with the label-linked pattern
Saves outputs/audit/timeline.png (regime share, loss and RSSI against real time, coloured by label).
"""
import argparse, csv, datetime as dt, json, os
import numpy as np


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r.get("error"):
            continue
        mp = os.path.join(os.path.dirname(r["path"]), "meta.json")
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        t = dt.datetime.fromisoformat(m["created"]).timestamp() if "created" in m else float("nan")
        rows.append(dict(session=r["session"], label=r["label"], condition=r["condition"], t=t, line=m.get("protocol_line", -1),
                         p128=float(r["pct_len128"]), loss=float(r["loss_pct"]), rate=float(r["rate_ts_hz"]), rssi=float(r["rssi_mean"])))
    return rows


def n_runs(s):
    return 1 + int((np.diff(s) != 0).sum()) if len(s) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "audit", "audit_summary.csv"))
    ap.add_argument("--gap", type=float, default=45.0, help="seconds between recordings counted as a pause")
    ap.add_argument("--perms", type=int, default=5000); ap.add_argument("--thr", type=float, default=3.0)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(); rng = np.random.default_rng(0); R = load(a.summary); out = []
    sessions = sorted({r["session"] for r in R})
    for s in sessions:
        rs = sorted([r for r in R if r["session"] == s], key=lambda r: r["t"])
        t = np.array([r["t"] for r in rs]); reg = np.array([int(r["p128"] >= a.thr) for r in rs]); lab = [r["label"] for r in rs]
        if np.isnan(t).any():
            out.append(f"\n{s}: some meta.json have no `created` time - skipped"); continue
        gaps = np.diff(t); big = [(i + 1, round(g)) for i, g in enumerate(gaps) if g > a.gap]
        obs = n_runs(reg); sim = np.array([n_runs(rng.permutation(reg)) for _ in range(a.perms)])
        p_cluster = (1 + (sim <= obs).sum()) / (1 + a.perms)
        order_ok = np.mean([rs[i]["line"] < rs[i + 1]["line"] for i in range(len(rs) - 1)])
        out.append(f"\n{s}: {len(rs)} recordings over {(t[-1] - t[0]) / 60:.0f} min; protocol-line order matches true order in {100 * order_ok:.0f}% of steps")
        out.append(f"  pauses > {a.gap:.0f} s: {len(big)} {big[:12]}  (index of recording after the pause, seconds)")
        out.append(f"  regime A (>= {a.thr:.0f}% len-128) in {reg.sum()}/{len(reg)} recordings; A/B runs in time order: observed {obs}, shuffled median {int(np.median(sim))} "
                   f"(p that time-clustering this strong arises by chance = {p_cluster:.3f})")
        out.append("  true-order sequence (A/B): " + "".join("A" if x else "B" for x in reg))
        out.append("  labels in that order:      " + " ".join(l.replace("person_", "P").replace("obj_", "O").replace("empty", "E") for l in lab))
        sh = {l: f"{int(reg[[x == l for x in lab]].sum())}/{sum(x == l for x in lab)}" for l in sorted(set(lab))}
        out.append("  regime A count per label:  " + str(sh))
    txt = "\n".join(out); print(txt)
    p = os.path.join(os.path.dirname(a.summary), "timeline_report.txt"); open(p, "w").write(txt)
    if not a.no_plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cols = {"empty": "k", "obj_Z1": "tab:blue", "obj_Z2": "tab:cyan", "obj_Z3": "tab:green", "person_Z1": "tab:red", "person_Z2": "tab:orange", "person_Z3": "tab:purple"}
        fig, ax = plt.subplots(3, len(sessions), figsize=(5.5 * len(sessions), 8), squeeze=False)
        for j, s in enumerate(sessions):
            rs = sorted([r for r in R if r["session"] == s], key=lambda r: r["t"]); t0 = rs[0]["t"]
            for k, (key, name) in enumerate((("p128", "% length-128 packets"), ("loss", "sequence loss %"), ("rssi", "RSSI mean (dBm)"))):
                for r in rs:
                    ax[k, j].scatter((r["t"] - t0) / 60, r[key], c=cols.get(r["label"], "grey"), s=22, marker="s" if r["condition"] == "person" else "o")
                ax[k, j].set_ylabel(name); ax[k, j].grid(alpha=.3)
            ax[0, j].set_title(s); ax[2, j].set_xlabel("minutes since first recording (square = person)")
        for l, c in cols.items():
            ax[0, 0].scatter([], [], c=c, label=l)
        ax[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(a.summary), "timeline.png"), dpi=120)
    print("\nsaved", p)


if __name__ == "__main__":
    main()