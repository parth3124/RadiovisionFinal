import glob
import json
import os
from collections import Counter

import numpy as np
import common as C

DATA = "dataset_room"
TRIM_SEC = 0.5
GAP_THRESHOLD_SEC = 0.10


def inspect_file(path):
    with np.load(path) as z:
        required = {"csi", "csi_len", "host_t"}
        if not required.issubset(z.files):
            return {"error": f"missing arrays: {sorted(required - set(z.files))}"}

        csi = z["csi"]
        lens = z["csi_len"].astype(int)
        t = z["host_t"].astype(float)
        raw_packets = len(t)
        if raw_packets == 0:
            return {"error": "no packets"}

        valid_lengths = lens[lens > 0]
        if not valid_lengths.size:
            return {"error": "no valid CSI lengths"}
        common_len, common_count = Counter(valid_lengths.tolist()).most_common(1)[0]
        keep = (lens == common_len) & np.isfinite(t)
        t = t[keep]
        raw_csi = csi[keep, :common_len]
        if len(t) < 2:
            return {"error": "fewer than two packets of the common CSI length"}

        order = np.argsort(t)
        t, raw_csi = t[order], raw_csi[order]
        dt = np.diff(t)
        positive_dt = dt[dt > 0]
        median_dt = float(np.median(positive_dt)) if positive_dt.size else float("nan")
        duration = float(t[-1] - t[0])
        usable = (t > t[0] + TRIM_SEC) & (t < t[-1] - TRIM_SEC)

        amp = C.csi_amplitude(raw_csi)
        packet_amp = np.mean(amp, axis=1)
        amp_mean = float(np.mean(packet_amp))
        amp_cv = float(np.std(packet_amp) / amp_mean) if amp_mean > 0 else float("nan")
        sub_mean = np.mean(amp, axis=0)
        sub_std = np.std(amp, axis=0)
        sub_cv = float(np.median(sub_std) / np.median(sub_mean)) if np.median(sub_mean) > 0 else float("nan")

        rssi_stats = [float("nan")] * 4
        if "rssi" in z.files:
            rssi = z["rssi"][keep][order].astype(float)
            if rssi.size:
                rssi_stats = [float(np.mean(rssi)), float(np.std(rssi)),
                              float(np.min(rssi)), float(np.max(rssi))]

        seq_gaps = float("nan")
        if "seq" in z.files:
            seq = z["seq"][keep][order].astype(np.int64)
            jumps = np.diff(seq)
            seq_gaps = int(np.sum((jumps > 1) & (jumps < 2**31)))

        return {
            "raw_packets": raw_packets,
            "packets": len(t),
            "common_len": int(common_len),
            "common_len_fraction": common_count / max(1, len(valid_lengths)),
            "duration": duration,
            "usable_duration": max(0.0, duration - 2 * TRIM_SEC),
            "usable_packets": int(np.sum(usable)),
            "rate": 1.0 / median_dt if median_dt > 0 else float("nan"),
            "gaps": int(np.sum(dt > GAP_THRESHOLD_SEC)),
            "max_gap_ms": float(np.max(dt) * 1000) if dt.size else float("nan"),
            "amp_mean": amp_mean,
            "amp_cv": amp_cv,
            "sub_cv": sub_cv,
            "rssi_mean": rssi_stats[0],
            "rssi_std": rssi_stats[1],
            "rssi_min": rssi_stats[2],
            "rssi_max": rssi_stats[3],
            "seq_gaps": seq_gaps,
        }


def fmt(value, digits=2):
    try:
        value = float(value)
        return f"{value:.{digits}f}" if np.isfinite(value) else "NA"
    except (TypeError, ValueError):
        return "NA"


def main():
    meta_paths = sorted(glob.glob(os.path.join(DATA, "*", "*", "meta.json")))
    rows = []
    for meta_path in meta_paths:
        folder = os.path.dirname(meta_path)
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            print(f"ERROR reading {meta_path}: {exc}")
            continue

        files = sorted(glob.glob(os.path.join(folder, "csi_*.npz")))
        if not files:
            print(f"NO CSI FILE: {folder}")
        for path in files:
            stats = inspect_file(path)
            if "error" in stats:
                print(f"ERROR {path}: {stats['error']}")
                continue
            stats.update({
                "session": str(meta.get("session", "?")),
                "condition": str(meta.get("condition", "?")),
                "zone": str(meta.get("zone") or "-"),
                "recording": os.path.basename(folder),
                "file": os.path.basename(path),
            })
            rows.append(stats)

    if not rows:
        raise SystemExit(f"No readable CSI recordings found under {DATA}/")

    print(f"\nDATASET QUALITY SUMMARY: {len(rows)} CSI file(s) under {DATA}/")
    print("rate=1/median inter-packet interval; gaps=host_t intervals >100 ms")
    print("amp_CV=temporal CV of packet-mean amplitude; sub_CV=median subcarrier temporal CV")
    print("\nPER-RECORDING RESULTS")
    print("session condition zone recording packets duration usable rate_Hz gaps maxgap_ms amp_CV sub_CV RSSI_mean RSSI_sd seq_gaps")
    for r in sorted(rows, key=lambda x: (x["session"], x["recording"], x["file"])):
        print(
            f"{r['session']:7s} {r['condition']:9s} {r['zone']:4s} "
            f"{r['recording'][:22]:22s} {r['packets']:7d} "
            f"{fmt(r['duration']):>8s} {fmt(r['usable_duration']):>6s} "
            f"{fmt(r['rate']):>7s} {r['gaps']:4d} {fmt(r['max_gap_ms']):>9s} "
            f"{fmt(r['amp_cv']):>6s} {fmt(r['sub_cv']):>6s} "
            f"{fmt(r['rssi_mean']):>9s} {fmt(r['rssi_std']):>7s} {fmt(r['seq_gaps'], 0):>8s}"
        )

    print("\nSESSION-LEVEL MEDIANS")
    for session in sorted({r["session"] for r in rows}):
        group = [r for r in rows if r["session"] == session]
        print(
            f"{session}: files={len(group)}, "
            f"duration={fmt(np.median([r['duration'] for r in group]))}s, "
            f"packets={fmt(np.median([r['packets'] for r in group]), 0)}, "
            f"rate={fmt(np.median([r['rate'] for r in group]))}Hz, "
            f"gaps>100ms={fmt(np.median([r['gaps'] for r in group]), 1)}, "
            f"amp_CV={fmt(np.median([r['amp_cv'] for r in group]))}, "
            f"sub_CV={fmt(np.median([r['sub_cv'] for r in group]))}, "
            f"RSSI={fmt(np.nanmedian([r['rssi_mean'] for r in group]))}dBm"
        )

    print("\nEMPTY-ROOM BASELINE COMPARISON")
    for session in sorted({r["session"] for r in rows}):
        group = [r for r in rows if r["session"] == session and r["condition"] == "empty"]
        if group:
            print(
                f"{session}: n={len(group)}, "
                f"mean CSI amplitude median={fmt(np.median([r['amp_mean'] for r in group]))}, "
                f"amp_CV median={fmt(np.median([r['amp_cv'] for r in group]))}, "
                f"sub_CV median={fmt(np.median([r['sub_cv'] for r in group]))}, "
                f"RSSI median={fmt(np.nanmedian([r['rssi_mean'] for r in group]))}dBm"
            )

    print("\nNOTE: Diagnostic summaries only; no single metric is an automatic pass/fail test.")


if __name__ == "__main__":
    main()
