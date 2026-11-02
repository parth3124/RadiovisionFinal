#!/usr/bin/env python3
"""No-hardware self-test: fabricate two laptops' recordings (2 RX each, packet loss, bursty USB
timestamps, different host clocks, a TX counter crossing the 32-bit µs wrap) and check that
merge_session.py re-aligns them exactly by TX seq."""
import json, os, struct, subprocess, sys, tempfile
import numpy as np
from sura_frames import pack_frame, FT_CSI, FT_STATS, STATS

rng = np.random.default_rng(0)
N, HZ, BOOT = 3000, 100, 0xDEADBEEF
tx_us0 = 2**32 - 5_000_000                      # wraps after 5 s
tmp = tempfile.mkdtemp()

def make(dirname, rx_ids, host_off, loss, L=384):
    d = os.path.join(tmp, dirname); os.makedirs(d)
    f = open(os.path.join(d, "raw_ttyACM0.sbin"), "wb")
    for seq in range(N):
        t_true = seq / HZ
        for rx in rx_ids:
            if rng.random() < loss: continue
            body = np.full(L, seq % 120, np.int8) + rx          # content identifies (seq, rx)
            fr = pack_frame(FT_CSI, rx, 1, -50 - rx, BOOT, seq, (tx_us0 + seq * 10_000) % 2**32,
                            (seq * 10_000 + rx * 777) % 2**32, body.tobytes())
            host_t = host_off + t_true * (1 + 20e-6) + abs(rng.normal(0, 0.004))   # +20 ppm, USB jitter
            f.write(struct.pack("<dH", host_t, len(fr)) + fr)
    f.close()
    ev = [dict(host_t=host_off + 5.0, wall=0, kind="marker", latest_tx_boot=BOOT, latest_tx_seq=500)]
    open(os.path.join(d, "events.jsonl"), "w").write("\n".join(json.dumps(e) for e in ev) + "\n")
    return d

A = make("laptopA", [1, 2], 1000.0, 0.10)
B = make("laptopB", [3, 4], 52000.0, 0.15)       # totally different monotonic clock
with open(os.path.join(A, "camera_frames.csv"), "w") as f:
    f.write("frame_idx,host_t\n")
    for i in range(900): f.write(f"{i},{1000.0 + 0.060 + i/30:.6f}\n")     # camera lags 60 ms

out = os.path.join(tmp, "m", "s.npz")
r = subprocess.run([sys.executable, "merge_session.py", A, B, "--out", out, "--flash-frames", "150"],
                   capture_output=True, text=True)
print(r.stdout, r.stderr)
z = np.load(out)
ok = True
for j, rx in enumerate(z["rx_ids"]):
    p = z["present"][:, j]
    exp = (z["tx_seq"][p] % 120 + rx).astype(np.int8)
    good = np.all(z["csi"][p, j, 0] == exp)
    ok &= bool(good)
    print(f"rx{rx}: content matches its own seq on all {p.sum()} packets: {good}")
tt = z["tx_t"]; print("tx_t span %.3f s (expected %.3f), wrap handled: %s" % (tt[-1], (N-1)/HZ, abs(tt[-1]-(N-1)/HZ) < 1e-6))
cam = np.load(out.replace(".npz", "_camera_frame_tx_t.npy"))
print("camera frame 150 -> TX time %.3f s (true flash instant 5.000 s)" % cam[150])
print("SELFTEST", "PASS" if ok else "FAIL")
