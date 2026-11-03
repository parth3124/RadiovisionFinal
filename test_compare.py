"""
test_compare.py - end-to-end validation of extract_features.py + compare_features.py on SYNTHETIC recordings with planted effects.
    python test_compare.py          (about 2-4 minutes)
Case A: person-zone signature in sanitised PHASE (same in every session), person RSSI shift, session-specific AMPLITUDE profiles, objects = pure null.
Case B: everything null. The pipeline must find A's planted effects (and prefer phase over session-specific amplitude) and must not 'find' anything in B.
"""
import csv, datetime as dt, json, os, shutil, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sura_io as S
import csi_complex as X
import extract_features as E
import compare_features as C

SETS_T = ["amp_mean", "phase_mean", "rssi", "amp+phase", "everything"]


def smooth(rng, sd, scale=8.0):
    k = np.arange(64); return sum(rng.normal(0, sd) * np.sin(2 * np.pi * (f * k / 64) + rng.uniform(0, 6.28)) for f in (1, 2, 3)) / 1.7


def build(root, effect, seed):
    rng = np.random.default_rng(seed); shutil.rmtree(root, ignore_errors=True)
    base_phase = [smooth(rng, 0.4) for _ in range(3)]; zone_vec = {z: [smooth(rng, 1.0) for _ in range(3)] for z in (1, 2, 3)}; common = [smooth(rng, 1.0) for _ in range(3)]
    t0 = dt.datetime(2026, 10, 29, 1, 0, 0)
    for si, s in enumerate(("r01", "r02", "r03")):
        prof = [np.exp(smooth(rng, 0.5)) * 30 for _ in range(3)]; rssi_off = rng.uniform(-5, 5)
        labels = ["empty"] * 6 + [f"{c}_Z{z}" for c in ("person", "obj") for z in (1, 2, 3) for _ in range(6)]
        for line, i in enumerate(rng.permutation(len(labels)), 1):
            lab = labels[i]; cond, _, z = lab.partition("_Z"); cond = cond or "empty"; d = os.path.join(root, lab, f"{s}_{line:02d}_{lab}"); os.makedirs(d)
            n = 640; ts = np.cumsum(rng.uniform(10e3, 24e3, n)).astype(np.int64); csi = np.zeros((n, 384), np.int8); off = rng.uniform(-np.pi, np.pi, n)
            for b in range(3):
                v = X.VALID_384[b]; eff = np.zeros(64)
                if cond == "person":
                    eff = effect * (0.6 * common[b] + zone_vec[int(z)][b])
                H = np.zeros(64, complex); H[v] = (prof[b] * np.exp(1j * (base_phase[b] + eff + 0.03 * si)))[v]
                c = H[None] * np.exp(1j * off)[:, None] * (1 + 0.03 * rng.normal(0, 1, (n, 1))) + 0.8 * (rng.normal(0, 1, (n, 64)) + 1j * rng.normal(0, 1, (n, 64))) * (np.abs(H) > 0)[None]
                blk = np.zeros((n, 128)); blk[:, 0::2] = c.imag; blk[:, 1::2] = c.real; csi[:, b * 128:(b + 1) * 128] = np.clip(np.round(blk), -127, 127)
            rssi = np.round(-55 + rssi_off + (-2.0 * effect / 0.25 if cond == "person" else 0) + rng.normal(0, 0.6, n)).astype(np.int8)
            np.savez_compressed(os.path.join(d, "csi_link.npz"), host_t=ts * 1e-6 + 100, seq=np.arange(n), ts_us=ts, rssi=rssi, csi=csi, csi_len=np.full(n, 384, np.int16))
            json.dump({"label": lab, "condition": cond, "zone": ("Z" + z) if z else None, "session": s, "protocol_line": line,
                       "created": (t0 + dt.timedelta(days=si, seconds=20 * line)).isoformat(), "links": ["link"]}, open(os.path.join(d, "meta.json"), "w"))


def run_case(effect, seed, tmp):
    root = os.path.join(tmp, f"data_{seed}"); build(root, effect, seed)
    meta, mats, names = E.extract(S.discover(root), [0, 1, 2], E.ALL_GROUPS, verbose=False)
    out = os.path.join(tmp, f"feat_{seed}"); E.save(out, meta, mats, names, {})
    return C.run(os.path.join(out, "features.npz"), 40, ["none", "ref"], SETS_T, "none.csv", verbose=False), len(meta)


def get(res, norm, task, s):
    return next(r for r in res if r["norm"] == norm and r["task"] == task and r["set"] == s)


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{info}]" if info else "")); return cond


tmp = tempfile.mkdtemp(); ok = True
res, n = run_case(0.25, 1, tmp)
ok &= check("extraction produced one vector per recording", n == 126, str(n))
pz, az = get(res, "none", "person_zone", "phase_mean"), get(res, "none", "person_zone", "amp_mean")
ok &= check("planted zone signature in phase is found across sessions (LOSO > 0.8, p < 0.05)", pz["loso"] > 0.8 and pz["p"] < 0.05, f"LOSO {pz['loso']:.2f}, p {pz['p']:.3f}")
ok &= check("phase beats session-specific amplitude by >= 0.2 (calibration-free)", pz["loso"] - az["loso"] >= 0.2, f"phase {pz['loso']:.2f} vs amp {az['loso']:.2f}")
pp_n, pp_r = get(res, "none", "person_presence", "rssi"), get(res, "ref", "person_presence", "rssi")
ok &= check("RSSI person effect (-2 dB) hidden by +-5 dB session offsets without calibration, recovered with the empty-room reference",
            pp_n["loso"] < pp_r["loso"] - 0.2 and pp_r["loso"] > 0.8, f"none {pp_n['loso']:.2f} -> ref {pp_r['loso']:.2f}")
ap_none, ap_ref = get(res, "none", "person_presence", "amp_mean"), get(res, "ref", "person_presence", "amp_mean")
ok &= check("empty-room reference calibration removes the session-specific amplitude profile (ref >= none)", ap_ref["loso"] >= ap_none["loso"] - 0.02, f"none {ap_none['loso']:.2f}, ref {ap_ref['loso']:.2f}")
obj = [r for r in res if r["task"].startswith("obj")]
sig = [r for r in obj if r["p"] < 0.05 and r["loso"] > r["chance"] + 0.15]
ok &= check("objects carry no planted effect: at most 1 of %d object results looks significant" % len(obj), len(sig) <= 1, f"{len(sig)} flagged; max LOSO {max(r['loso'] for r in obj):.2f}")
resB, _ = run_case(0.0, 2, tmp)
zon = [r["loso"] for r in resB if r["task"].endswith("zone")]
sigB = [r for r in resB if r["p"] < 0.05 and r["loso"] > r["chance"] + 0.15]
ok &= check("full null: mean zone accuracy ~ chance (0.33 +- 0.07)", abs(np.mean(zon) - 1 / 3) < 0.07, f"mean {np.mean(zon):.3f} over {len(zon)} results")
ok &= check("full null: at most 20%% of results flagged (expected ~5%%)", len(sigB) <= 0.2 * len(resB), f"{len(sigB)}/{len(resB)} flagged")
shutil.rmtree(tmp, ignore_errors=True)
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED"); sys.exit(0 if ok else 1)
