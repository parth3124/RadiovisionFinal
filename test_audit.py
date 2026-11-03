"""
test_audit.py - validates audit_recordings.py against SYNTHETIC recordings whose true properties are known by construction.
    python test_audit.py
(No real data involved; checks that the measurements the audit reports are the ones that were put in.)
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_recordings as A

def make(random_phase, seed=0):
    rng = np.random.default_rng(seed); n_prod = 700
    seq_all = np.arange(1000, 1000 + n_prod); drop = rng.choice(n_prod, 70, replace=False); keep = np.setdiff1d(np.arange(n_prod), drop)
    n = len(keep); ts_all = np.cumsum(rng.uniform(10e3, 24e3, n_prod)).astype(np.int64)      # irregular radio clock (us)
    ts, seq = ts_all[keep], seq_all[keep]
    host = ts * 1e-6 * (1 + 1000e-6) + 50.0 + 0.25 * np.sin(np.pi * np.linspace(0, 1, n)) ** 2  # +1000 ppm drift + 250 ms latency BUMP (a linear ramp would be indistinguishable from drift)
    lens = np.where((np.arange(n) % 10) == 0, 128, 384).astype(np.int16)                  # 10% length-128, 90% length-384
    csi = np.zeros((n, 384), np.int8)
    null0 = np.r_[0:6, 32, 59:64]; null1 = np.r_[0:2, 59:64]; null2 = np.r_[0:6, 63]
    k = np.arange(64); base = 20 + 5 * np.sin(k / 9.0)
    for i in range(n):
        for b in range(lens[i] // 128):
            nl = [null0, null1, null2][b]; amp = base.copy(); amp[nl] = 0
            ph = (rng.uniform(-np.pi, np.pi) if random_phase else 0.3) + 0.02 * k                # constant per-packet offset + small slope
            c = amp * np.exp(1j * ph)
            blk = np.zeros(128); blk[0::2] = c.imag; blk[1::2] = c.real                          # [imag, real] pairs
            csi[i, b * 128:(b + 1) * 128] = np.clip(np.round(blk), -127, 127)
    a = dict(host_t=host, seq=seq.astype(np.uint32), ts_us=(ts % 2 ** 32).astype(np.uint32), rssi=np.full(n, -50, np.int8), csi=csi, csi_len=lens)
    return dict(a=a, meta={}, link="link", meta_key="link", problems=[], session="synthetic", label="t", condition="t", zone=None), n, 70

def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return cond

ok = True
rec, n, missing = make(random_phase=True); r = A.audit_recording(rec)
ok &= check("packet count", r["schema"]["n"] == n)
ok &= check("length fractions 10% / 90%", r["schema"]["len_counts"] == {128: 70, 384: n - 70} or abs(r["schema"]["len_counts"][128] / n - 0.1) < 0.01, str(r["schema"]["len_counts"]))
ok &= check("missing records recovered exactly", r["seq"]["missing"] == missing, f"{r['seq']['missing']} vs {missing}")
ok &= check("loss % correct", abs(r["seq"]["loss_pct"] - 100 * missing / (n + missing)) < 0.1, str(r["seq"]["loss_pct"]))
ok &= check("host-vs-radio drift ~ +1000 ppm", abs(r["timing"]["drift_ppm_host_vs_ts"] - 1000) < 300, str(r["timing"]["drift_ppm_host_vs_ts"]))
ok &= check("latency bump detected (p95 > 100 ms)", r["timing"]["host_minus_ts_residual_ms"]["p95"] > 100, str(r["timing"]["host_minus_ts_residual_ms"]))
ok &= check("irregular sampling flagged", any(f.startswith("IRREGULAR") for f in r["flags"]))
b384 = r["blocks"][384]["blocks"]
ok &= check("null subcarriers recovered per block", [b["null_subcarriers"] for b in b384] == ["0-5,32,59-63", "0-1,59-63", "0-5,63"], str([b["null_subcarriers"] for b in b384]))
ok &= check("random per-packet phase detected (R < 0.2)", all(b["packet_mean_phase_R"] < 0.2 for b in b384), str([b["packet_mean_phase_R"] for b in b384]))
ok &= check("slope circular mean ~ 0.02 rad/subcarrier", all(abs(b["adj_step_rad_circ_mean"] - 0.02) < 0.01 for b in b384), str([b["adj_step_rad_circ_mean"] for b in b384]))
ok &= check("within-packet phase smooth (R > 0.99)", all(b["adj_step_within_packet_R_median"] > 0.99 for b in b384))
ok &= check("PHASE_OFFSET_RANDOM flagged", any(f.startswith("PHASE_OFFSET_RANDOM") for f in r["flags"]))
ok &= check("MIXED_TYPES + COMMON_LEN_FILTER flagged", any(f.startswith("MIXED") for f in r["flags"]) and any(f.startswith("COMMON_LEN") for f in r["flags"]))
ok &= check("RSSI stats", r["rssi"]["mean"] == -50.0 and r["rssi"]["n_distinct"] == 1)
rec2, *_ = make(random_phase=False); r2 = A.audit_recording(rec2); b2 = r2["blocks"][384]["blocks"]
ok &= check("stable phase -> R > 0.95 and no PHASE_OFFSET flag", all(b["packet_mean_phase_R"] > 0.95 for b in b2) and not any(f.startswith("PHASE_OFFSET") for f in r2["flags"]), str([b["packet_mean_phase_R"] for b in b2]))
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED"); sys.exit(0 if ok else 1)
