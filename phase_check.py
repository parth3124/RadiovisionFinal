#!/usr/bin/env python3
"""
phase_check.py - is CSI phase usable, and which sanitisation makes it stable?   (single link, 384-type packets, per 128-value block)

    python phase_check.py --data dataset_room                    # all recordings -> outputs/phase/phase_summary.csv + phase_report.txt
    python phase_check.py --data dataset_room --limit 6
    python phase_check.py --data dataset_room --plot-rec r02_17  # plots for recordings whose folder name contains this text
    python phase_check.py --npz csi_link.npz --meta meta.json    # one loose pair

Per recording and block, for each method (raw / cpo / cpo_slope / ref, see csi_complex.py):
  R      median over subcarriers of the circular concentration of phase over time (1 = stable, ~0 = random). In an EMPTY room the channel is
         nearly static, so a good sanitisation should push R towards 1; this is the validity test of the sanitisation itself.
  R10    10th percentile of the same per-subcarrier R (the median hides changes confined to a minority of subcarriers)
  jump   share of consecutive-packet phase steps larger than pi/2 (phase discontinuities)
  Rcp    inter-block common-phase relation: concentration of (common phase of block j - block i) over packets. High = the blocks share the
         same per-packet offset (relative phase between blocks is meaningful); ~0 = independent random offsets (treat blocks separately).
Does NOT yet ask whether sanitised phase separates presence/zones; that is the next script. Nothing is written into the dataset.
"""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sura_io as S
import csi_complex as X

MIN_PKTS = 100


def analyse(rec):
    blocks, ts, rssi, host = X.select_type(rec["a"])
    n = len(ts)
    if n < MIN_PKTS:
        return None
    row = dict(session=rec["session"], label=rec["label"], condition=rec["condition"], zone=rec["zone"], link=rec["link"], n384=n, path=rec["path"])
    cps = {}
    for b, c in enumerate(blocks):
        v = X.VALID_384[b]; cps[b] = X.common_phase(c, v)
        for m in X.METHODS:
            cm = X.sanitize(c, v, m)
            R = X.temporal_R(cm, v) if m != "ref" else X.temporal_R(cm, v[v != v[len(v) // 2]])
            row[f"b{b}_R_{m}"] = round(float(np.median(R)), 4); row[f"b{b}_R10_{m}"] = round(float(np.percentile(R, 10)), 4); row[f"b{b}_jump_{m}"] = round(X.jump_fraction(cm, v, ts) if m != "ref" else X.jump_fraction(cm[:, :], v[v != v[len(v) // 2]], ts), 4)
        s, _ = X.slope(c, v); row[f"b{b}_slope_sd"] = round(X.circ_sd(s), 4)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        row[f"Rcp_{i}{j}"] = round(float(abs(np.exp(1j * (cps[j] - cps[i])).mean())), 4)
    return row


def plot_rec(rec, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    blocks, ts, rssi, host = X.select_type(rec["a"]); n = min(len(ts), 400)
    fig, ax = plt.subplots(3, 3, figsize=(17, 12), gridspec_kw=dict(hspace=0.55, wspace=0.25)); fig.suptitle(f"{rec['session']} {rec['label']}  ({os.path.basename(os.path.dirname(rec['path']))}), first {n} 384-type packets")
    for b, c in enumerate(blocks):
        v = X.VALID_384[b]
        for r, m in enumerate(("raw", "cpo_slope")):
            im = ax[r, b].imshow(np.angle(X.sanitize(c, v, m)[:n][:, v]).T, aspect="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax[r, b].set_title(f"block {b}: {m} phase (rad)"); ax[r, b].set_xlabel("packet #"); ax[r, b].set_ylabel("valid subcarrier #")
        for m in X.METHODS:
            vv = v if m != "ref" else v[v != v[len(v) // 2]]
            ax[2, b].plot(vv, X.temporal_R(X.sanitize(c, v, m), vv), label=m)
        ax[2, b].set_ylim(0, 1.02); ax[2, b].set_title(f"block {b}: temporal phase stability R (1 = stable)"); ax[2, b].legend(fontsize=7); ax[2, b].set_xlabel("subcarrier index")
    fig.colorbar(im, ax=ax[:2, :], shrink=.6, label="rad"); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data"); ap.add_argument("--npz"); ap.add_argument("--meta"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plot-rec", nargs="*", default=[]); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "phase"))
    a = ap.parse_args()
    items = [(a.npz, a.meta)] if a.npz else (S.discover(a.data) if a.data else None)
    if not items:
        ap.error("give --data ROOT or --npz FILE [--meta FILE]")
    if a.limit:
        items = items[:a.limit]
    os.makedirs(a.out, exist_ok=True); rows = []
    for npz, mp in items:
        rec = S.load(npz, mp); r = analyse(rec)
        if r is None:
            print(f"skip (too few 384-type packets): {npz}"); continue
        rows.append(r)
        if a.npz or any(t in npz for t in a.plot_rec):
            os.makedirs(os.path.join(a.out, "plots"), exist_ok=True); plot_rec(rec, os.path.join(a.out, "plots", f"{rec['session']}_{rec['label']}_{os.path.basename(os.path.dirname(npz))}.png"))
    keys = list(rows[0].keys()); w = csv.DictWriter(open(os.path.join(a.out, "phase_summary.csv"), "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(rows)
    out = [f"{len(rows)} recordings analysed (384-type packets, >= {MIN_PKTS} each)", "\nTEMPORAL PHASE STABILITY R (median over recordings; 1 = stable, ~0 = random) per session x condition x block"]
    out.append(f"{'session':8s}{'condition':10s}{'blk':>4s}{'n':>4s}" + "".join(f"{m:>11s}" for m in X.METHODS) + f"{'jump_raw':>10s}{'jump_cpo_sl':>12s}{'R10_cpo':>9s}")
    for s in sorted({r["session"] for r in rows}):
        for c in ("empty", "obj", "person"):
            g = [r for r in rows if r["session"] == s and r["condition"] == c]
            for b in range(3):
                if g:
                    out.append(f"{s:8s}{c:10s}{b:>4d}{len(g):>4d}" + "".join(f"{np.median([r[f'b{b}_R_{m}'] for r in g]):>11.3f}" for m in X.METHODS)
                               + f"{np.median([r[f'b{b}_jump_raw'] for r in g]):>10.3f}{np.median([r[f'b{b}_jump_cpo_slope'] for r in g]):>12.3f}{np.median([r[f'b{b}_R10_cpo'] for r in g]):>9.3f}")
    out.append("\nINTER-BLOCK COMMON-PHASE CONCENTRATION Rcp (median over all recordings): " + ", ".join(f"{k}={np.median([r[k] for r in rows]):.3f}" for k in ("Rcp_01", "Rcp_02", "Rcp_12")))
    out.append("  (high = blocks share the per-packet offset -> relative phase between blocks usable; ~0 = independent offsets)")
    out.append("per-packet adjacent-subcarrier slope spread (circular sd, rad): " + ", ".join(f"block{b}={np.median([r[f'b{b}_slope_sd'] for r in rows]):.3f}" for b in range(3)))
    txt = "\n".join(out); print(txt); open(os.path.join(a.out, "phase_report.txt"), "w").write(txt); print("\nsaved", a.out)


if __name__ == "__main__":
    main()