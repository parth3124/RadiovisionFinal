#!/usr/bin/env python3
"""Merge one or more session folders (one per laptop) into a single aligned dataset.

    python merge_session.py sessions/s01_laptopA sessions/s01_laptopB --out merged/s01.npz
        [--flash-frames 412 9020]   # video frame index where each sync marker's flash is first seen

Alignment key = (tx_boot, tx_seq): every receiver hears the same numbered TX packets, so no
clock sync between boards or laptops is needed. Output rows are TX packets (uniform ~100 Hz),
columns are receivers; `present` marks packets a receiver actually got.

Output npz: tx_seq (N,), tx_t (N,) seconds on the TX clock, rx_ids (R,), present (N,R),
csi int8 (N,R,Lmax) raw bytes as ESP-IDF delivers them, csi_len, rssi, rx_ts_us, flags.
meta json: host<->TX clock fits, events converted to TX time/seq, camera frame -> TX time.
Use csi_to_complex() to turn raw bytes into complex CSI (imaginary byte comes first).
"""
import argparse, collections, csv, glob, json, os, struct
import numpy as np
from sura_frames import HDR, HDR_SIZE, FT_CSI, FT_STATS, STATS, FL_FIRST_WORD_INVALID


def csi_to_complex(csi, csi_len=None):
    """(…, L) int8 -> (…, L//2) complex64. ESP-IDF stores [imag, real] per subcarrier.
    On ESP32-S3 the first 4 bytes of the LLTF block are invalid (first_word_invalid): zeroed here.
    Blocks are 64 complex values each, in order LLTF, HT-LTF, STBC-HT-LTF (when present)."""
    c = csi.astype(np.float32)
    z = c[..., 1::2] + 1j * c[..., 0::2]
    return z.astype(np.complex64)


def read_dir(d):
    rows, stats = [], collections.defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(d, "raw_*.sbin"))):
        data = open(fn, "rb").read()
        i = 0
        while i + 10 <= len(data):
            host_t, n = struct.unpack_from("<dH", data, i); i += 10
            fr = data[i:i + n]; i += n
            if len(fr) < n: break
            h = HDR.unpack_from(fr); body = fr[HDR_SIZE:-2]
            if h[2] == FT_CSI:
                rows.append((host_t, h, body))
            elif h[2] == FT_STATS:
                stats[h[3]].append((host_t,) + STATS.unpack(body))
    return rows, stats


def lower_envelope_fit(x, y, iters=5, keep=0.3):
    """host_t = a + b*tx_t + latency, latency >= 0 and bursty: fit the lower envelope."""
    m = np.ones(len(x), bool)
    for _ in range(iters):
        b, a = np.polyfit(x[m], y[m], 1)
        r = y - (a + b * x)
        m = r <= np.quantile(r[m], keep) if m.sum() > 50 else m
    r = y - (a + b * x)
    return a, b, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flash-frames", nargs="*", type=int, default=[])
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    per_dir = {d: read_dir(d) for d in a.dirs}
    boots = collections.Counter(h[7] for rows, _ in per_dir.values() for _, h, _ in rows)
    if not boots: raise SystemExit("no CSI frames found")
    boot = boots.most_common(1)[0][0]
    if len(boots) > 1:
        print(f"WARNING: {len(boots)} TX boots seen {dict(boots)}; keeping the most common ({boot:08x}). "
              "A TX reset mid-session splits the recording; analyse the other boot separately.")
    seqs = np.array([h[8] for rows, _ in per_dir.values() for _, h, _ in rows if h[7] == boot])
    s0, s1 = int(seqs.min()), int(seqs.max()); N = s1 - s0 + 1
    rx_ids = sorted({h[3] for rows, _ in per_dir.values() for _, h, _ in rows})
    R = len(rx_ids); col = {r: i for i, r in enumerate(rx_ids)}
    Lmax = max(len(b) for rows, _ in per_dir.values() for _, _, b in rows)
    present = np.zeros((N, R), bool); csi = np.zeros((N, R, Lmax), np.int8)
    csi_len = np.zeros((N, R), np.int16); rssi = np.zeros((N, R), np.int8)
    rx_ts = np.zeros((N, R), np.uint32); flags = np.zeros((N, R), np.uint8)
    tx_us_seq = np.full(N, np.nan); dup = 0
    for rows, _ in per_dir.values():
        for _, h, body in rows:
            if h[7] != boot: continue
            n, c = h[8] - s0, col[h[3]]
            if present[n, c]: dup += 1; continue
            present[n, c] = True
            csi[n, c, :len(body)] = np.frombuffer(body, np.int8)
            csi_len[n, c] = len(body); rssi[n, c] = h[6]; flags[n, c] = h[5]; rx_ts[n, c] = h[10]
            tx_us_seq[n] = h[9]
    seen = np.where(~np.isnan(tx_us_seq))[0]
    u = tx_us_seq[seen].astype(np.int64)
    inc = np.mod(np.diff(u), 2**32)                      # unwrap the 32-bit µs counter
    tx_t_seen = np.concatenate([[0], np.cumsum(inc)]) / 1e6
    tx_t = np.interp(np.arange(N), seen, tx_t_seen)
    seq_axis = np.arange(s0, s1 + 1)

    meta = dict(tx_boot=f"{boot:08x}", seq_first=s0, seq_last=s1, n_packets=N, rx_ids=rx_ids,
                tx_rate_hz=float((N - 1) / tx_t[-1]) if tx_t[-1] > 0 else None, dir_fits={}, events=[])
    print(f"\nTX boot {boot:08x}: seq {s0}..{s1} (N={N}), measured TX rate {meta['tx_rate_hz']:.2f} Hz, duplicates {dup}")
    print("rx  coverage  longest_gap(pkts)  mean_rssi  len_hist            sigmode  first_word_inv")
    for r in rx_ids:
        c = col[r]; p = present[:, c]
        gaps = np.diff(np.concatenate([[-1], np.where(p)[0], [N]])) - 1
        lh = dict(collections.Counter(csi_len[p, c].tolist()))
        sm = dict(collections.Counter((flags[p, c] & 3).tolist()))
        print(f"{r:>2}  {100*p.mean():6.1f}%  {gaps.max():>10}        {rssi[p, c].mean():7.1f}   {str(lh):<20} {sm}  {100*np.mean(flags[p,c]&FL_FIRST_WORD_INVALID>0):.0f}%")
    print(f"all {R} receivers present: {100*present.all(1).mean():.1f}% of packets;  >=2 receivers: {100*(present.sum(1)>=2).mean():.1f}%")
    for d, (rows, stats) in per_dir.items():
        for r, lst in stats.items():
            last = lst[-1]
            print(f"  [{os.path.basename(d.rstrip('/'))}] rx{r} final stats: matched={last[1]} unmatched_csi={last[2]} orphan_promisc={last[3]} serial_drops={last[4]}")

    for d, (rows, _) in per_dir.items():
        pts = [(tx_t[h[8] - s0], t) for t, h, _ in rows if h[7] == boot]
        if len(pts) < 100: continue
        x, y = np.array(pts).T
        A, B, res = lower_envelope_fit(x, y)
        meta["dir_fits"][d] = dict(a=A, b=B, latency_ms_p50=float(np.median(res) * 1e3),
                                  latency_ms_p95=float(np.quantile(res, .95) * 1e3), drift_ppm=float((B - 1) * 1e6))
        print(f"{os.path.basename(d.rstrip('/'))}: host_t = a + b*tx_t, b-1 = {(B-1)*1e6:+.1f} ppm; USB/host latency above envelope p50 {np.median(res)*1e3:.1f} ms, p95 {np.quantile(res,.95)*1e3:.1f} ms")
        evf = os.path.join(d, "events.jsonl")
        evs = [json.loads(l) for l in open(evf)] if os.path.exists(evf) else []
        for e in evs:
            tt = (e["host_t"] - A) / B
            e["tx_t"] = tt; e["tx_seq_est"] = float(np.interp(tt, tx_t, seq_axis)); e["dir"] = d
            meta["events"].append(e)
        cf = os.path.join(d, "camera_frames.csv")
        if os.path.exists(cf):
            fr = np.array([[float(v) for v in row] for row in list(csv.reader(open(cf)))[1:]])
            lag = 0.0
            markers = [e for e in evs if e["kind"] == "marker"]
            if a.flash_frames and markers:
                lags = [fr[f, 1] - m["host_t"] for f, m in zip(a.flash_frames, markers)]
                lag = float(np.median(lags))
                print(f"camera delivery lag vs flash markers: {np.round(lags, 3)} s -> using {lag*1e3:.0f} ms (spread {np.ptp(lags)*1e3:.0f} ms)")
            else:
                print("camera: no --flash-frames given, camera delivery lag NOT corrected (typically 30-100 ms).")
            cam_tx_t = (fr[:, 1] - lag - A) / B
            np.save(a.out.replace(".npz", "_camera_frame_tx_t.npy"), cam_tx_t)
            meta["camera_lag_s"] = lag
            print(f"saved camera frame -> TX time mapping ({len(cam_tx_t)} frames)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, tx_seq=seq_axis, tx_t=tx_t, rx_ids=np.array(rx_ids), present=present,
                        csi=csi, csi_len=csi_len, rssi=rssi, rx_ts_us=rx_ts, flags=flags)
    json.dump(meta, open(a.out.replace(".npz", "_meta.json"), "w"), indent=1, default=str)
    print(f"\nwrote {a.out} and meta")


if __name__ == "__main__":
    main()
