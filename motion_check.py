"""
motion_check.py - does the CSI change much more during big arm motion than during held poses? Camera-free.

    python motion_check.py --data dataset_v2 --sync sync_test/idle/<recording folder>
Motion energy of a recording = mean over live subcarriers and time of |x(t) - x(t - 0.1 s)|, where x is the
per-packet-normalized log amplitude (norm=packet, so no session statistics are involved).
Compares the sync (waving) recording with the held-pose recordings of every class.
"""
import argparse
import numpy as np
import common as C
import dataset as D


def energy(r, live):
    G = r.G.astype(np.float32)[:, live]; lag = int(round(0.1 * r.fs))
    d = np.abs(G[lag:] - G[:-lag]).mean(1)
    ok = r.valid[lag:] & r.valid[:-lag]
    return float(d[ok].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v2")
    ap.add_argument("--sync", required=True)
    a = ap.parse_args()
    recs = D.load_recordings(a.data, 1.0, 0.25, verbose=False)
    sync = D.Recording(a.sync, 1.0, 0.25)
    allr = recs + [sync]
    rm = np.mean([r.raw_mean for r in allr], 0); live = rm > 0.05 * np.median(rm)
    D.apply_norm(allr, "packet")
    E = np.array([energy(r, live) for r in recs]); y = np.array([r.label for r in recs]); es = energy(sync, live)
    print("motion energy (higher = CSI changes more over 0.1 s), held-pose recordings:")
    for c, name in enumerate(C.CLASSES):
        v = E[y == c]
        print(f"  {name:<14} median {np.median(v):.4f}   range {v.min():.4f} - {v.max():.4f}   (n={len(v)})")
    idle = np.median(E[y == 0])
    print(f"\nsync (waving) recording: {es:.4f}")
    print(f"  = {es / idle:.2f} x the median idle recording, {es / np.median(E):.2f} x the median of all held poses")
    print(f"  above {100 * np.mean(E < es):.0f}% of the held-pose recordings")
    print("\nA ratio near 1 means the link barely responds to large arm motion; 2x or more means it clearly does.")


if __name__ == "__main__":
    main()