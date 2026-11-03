"""
test_phase.py - validates csi_complex.py / phase_check.py on SYNTHETIC CSI with known phase offsets, slopes and channel changes.
    python test_phase.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csi_complex as X
import phase_check as P


def make(n=500, slope_sd=0.0, wobble=0.0, shared_offset=True, seed=0):
    rng = np.random.default_rng(seed); k = np.arange(64)
    ts = np.cumsum(rng.uniform(12e3, 22e3, n)).astype(np.int64)
    csi = np.zeros((n, 384), np.int8)
    off = rng.uniform(-np.pi, np.pi, n); s = rng.normal(0, slope_sd, n)
    for b in range(3):
        v = X.VALID_384[b]; H = np.zeros(64, complex); H[v] = 40 * np.exp(1j * (0.3 * np.sin(k[v] / 6.0 + b)))
        ob = off if shared_offset else rng.uniform(-np.pi, np.pi, n)
        w = wobble * rng.normal(0, 1, n)[:, None] * (k[None] > 40)                       # channel change on the upper subcarriers
        c = H[None] * np.exp(1j * (ob[:, None] + s[:, None] * (k[None] - 32) + w))
        blk = np.zeros((n, 128)); blk[:, 0::2] = c.imag; blk[:, 1::2] = c.real
        csi[:, b * 128:(b + 1) * 128] = np.clip(np.round(blk), -127, 127)
    a = dict(host_t=ts * 1e-6, seq=np.arange(n), ts_us=ts, rssi=np.full(n, -50, np.int8), csi=csi, csi_len=np.full(n, 384, np.int16))
    return dict(a=a, meta={}, link="link", session="syn", label="t", condition="t", zone=None, path="syn")


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return cond


ok = True
r = P.analyse(make())
ok &= check("static channel + random offset: raw phase unusable (R<0.2)", all(r[f"b{b}_R_raw"] < 0.2 for b in range(3)), str([r[f"b{b}_R_raw"] for b in range(3)]))
ok &= check("cpo restores stability (R>0.97)", all(r[f"b{b}_R_cpo"] > 0.97 for b in range(3)), str([r[f"b{b}_R_cpo"] for b in range(3)]))
ok &= check("no phase jumps after cpo", all(r[f"b{b}_jump_cpo"] < 0.01 for b in range(3)) and all(r[f"b{b}_jump_raw"] > 0.3 for b in range(3)))
ok &= check("shared offsets -> inter-block R ~ 1", min(r["Rcp_01"], r["Rcp_02"], r["Rcp_12"]) > 0.97, str((r["Rcp_01"], r["Rcp_02"], r["Rcp_12"])))
r2 = P.analyse(make(shared_offset=False))
ok &= check("independent offsets -> inter-block R < 0.2", max(r2["Rcp_01"], r2["Rcp_02"], r2["Rcp_12"]) < 0.2, str((r2["Rcp_01"], r2["Rcp_02"], r2["Rcp_12"])))
r3 = P.analyse(make(slope_sd=0.05))
ok &= check("random slope hurts cpo but cpo_slope fixes it", all(r3[f"b{b}_R_cpo"] < 0.9 and r3[f"b{b}_R_cpo_slope"] > 0.97 for b in range(3)),
            str([(r3[f"b{b}_R_cpo"], r3[f"b{b}_R_cpo_slope"]) for b in range(3)]))
ok &= check("slope spread measured ~0.05", all(0.03 < r3[f"b{b}_slope_sd"] < 0.07 for b in range(3)), str([r3[f"b{b}_slope_sd"] for b in range(3)]))
r4 = P.analyse(make(wobble=0.8))
# theory: wrapped-normal phase noise sigma=0.8 rad gives R = exp(-sigma^2/2) = 0.73 on the affected subcarriers; common-phase removal absorbs part of it,
# so R10 must lie between 0.73 and the static value (1.0), while the median over all subcarriers stays high (documented blind spot)
ok &= check("channel change on 37% of subcarriers: R10 in the analytical range (0.73-0.95), median R still > 0.9",
            all(0.73 < r4[f"b{b}_R10_cpo"] < 0.95 for b in range(3)) and all(r4[f"b{b}_R_cpo"] > 0.9 for b in range(3)),
            f"median {[r4[f'b{b}_R_cpo'] for b in range(3)]}, R10 {[r4[f'b{b}_R10_cpo'] for b in range(3)]}, theory exp(-0.32)={np.exp(-0.32):.2f}")
c = X.to_complex(make()["a"]["csi"])[0]; v = X.VALID_384[0]
ok &= check("null positions are zero in the complex block", np.allclose(np.abs(c[:, np.setdiff1d(np.arange(64), v)]), 0))
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED"); sys.exit(0 if ok else 1)