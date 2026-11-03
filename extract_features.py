#!/usr/bin/env python3
"""
extract_features.py - one feature vector per recording (single link, 384-type packets, first/last 0.5 s trimmed), saved for reuse.

    python extract_features.py --data dataset_room                        # all groups, all blocks -> outputs/features/all/
    python extract_features.py --data dataset_room --groups amp_mean phase_mean rssi --blocks 1 2 --tag b12
    python extract_features.py --data dataset_room --limit 5

Groups (each can be switched on/off; names carry the block, e.g. b0_phase_mean_17):
  amp_mean    mean over time of per-packet-normalised log amplitude (log1p|c| minus its mean over the block's valid subcarriers), per valid subcarrier
  amp_std     std over time of the same
  phase_mean  circular mean over time of the cpo-sanitised phase (common per-packet offset removed), per valid subcarrier
  phase_std   circular sd over time of the same
  rssi        RSSI mean/std/min/max of the 384-type packets and mean/std of all packets
  temporal    per block: mean |first difference| and |second difference| of amplitude and of phase (packet pairs <= 40 ms apart)
  spectral    per block: share of fluctuation power in 0.1-0.5, 0.5-2, 2-8, 8-20 Hz plus log total power, for amplitude and phase
              (series resampled to 40 Hz; a Doppler-LIKE fluctuation spectrum, not a true Doppler estimate - the scenes are static)
Outputs: features.npz (one matrix per group + metadata) and index.json (feature names, settings). Raw recordings are only read.
"""
import argparse, datetime as dt, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sura_io as S
import csi_complex as X

ALL_GROUPS = ["amp_mean", "amp_std", "phase_mean", "phase_std", "rssi", "temporal", "spectral"]
BANDS = [(0.1, 0.5), (0.5, 2.0), (2.0, 8.0), (8.0, 20.0)]
TRIM_S, FS, MAXDT, MIN_PKTS = 0.5, 40.0, 0.040, 100


def resample(t, Y, grid):
    i1 = np.clip(np.searchsorted(t, grid), 1, len(t) - 1); i0 = i1 - 1
    w = ((grid - t[i0]) / np.maximum(t[i1] - t[i0], 1e-9))[:, None]
    return Y[i0] * (1 - w) + Y[i1] * w


def band_feats(Yg, fs=FS):
    Y = (Yg - Yg.mean(0)) * np.hanning(len(Yg))[:, None]
    P = (np.abs(np.fft.rfft(Y, axis=0)) ** 2).mean(1); f = np.fft.rfftfreq(len(Y), 1 / fs)
    tot = P[(f >= 0.1) & (f < 20)].sum() + 1e-12
    return [float(P[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in BANDS] + [float(np.log10(tot))]


def rec_features(rec, blocks, groups):
    bl, ts, rssi, host = X.select_type(rec["a"])
    if len(ts) < MIN_PKTS:
        return None
    t = (ts - ts[0]) * 1e-6; keep = (t >= TRIM_S) & (t <= t[-1] - TRIM_S)
    if keep.sum() < MIN_PKTS:
        return None
    t = t[keep]; ok1 = np.diff(t) <= MAXDT; ok2 = ok1[:-1] & ok1[1:]
    F = {g: ([], []) for g in groups}                                               # group -> (names, values)
    for b in blocks:
        v = X.VALID_384[b]; c = bl[b][keep]; la = np.log1p(np.abs(c[:, v])); la = la - la.mean(1, keepdims=True)
        cs = X.sanitize(c, v, "cpo")[:, v]; u = cs / np.maximum(np.abs(cs), 1e-9); m = u.mean(0)
        if "amp_mean" in groups:
            F["amp_mean"][0].extend(f"b{b}_amp_mean_{k}" for k in v); F["amp_mean"][1].extend(la.mean(0))
        if "amp_std" in groups:
            F["amp_std"][0].extend(f"b{b}_amp_std_{k}" for k in v); F["amp_std"][1].extend(la.std(0))
        if "phase_mean" in groups:
            F["phase_mean"][0].extend(f"b{b}_phase_mean_{k}" for k in v); F["phase_mean"][1].extend(np.angle(m))
        if "phase_std" in groups:
            F["phase_std"][0].extend(f"b{b}_phase_std_{k}" for k in v); F["phase_std"][1].extend(np.sqrt(-2 * np.log(np.clip(np.abs(m), 1e-6, 1.0))))
        if "temporal" in groups:
            dphi = np.angle(u[1:] * np.conj(u[:-1])); d2phi = np.angle(np.exp(1j * dphi[1:]) * np.conj(np.exp(1j * dphi[:-1])))
            vals = [np.abs(np.diff(la, axis=0))[ok1].mean(), np.abs(np.diff(la, 2, axis=0))[ok2].mean(), np.abs(dphi)[ok1].mean(), np.abs(d2phi)[ok2].mean()]
            F["temporal"][0].extend(f"b{b}_temporal_{n}" for n in ("d1amp", "d2amp", "d1phase", "d2phase")); F["temporal"][1].extend(vals)
        if "spectral" in groups:
            grid = np.arange(t[0], t[-1], 1 / FS)
            if len(grid) < 64:
                return None
            A = resample(t, la, grid); U = resample(t, u.real, grid) + 1j * resample(t, u.imag, grid)
            Pm = np.angle(U * np.conj(m / np.maximum(np.abs(m), 1e-9))[None, :])
            nm = [f"b{b}_spectral_{s}_{n}" for s in ("amp", "phase") for n in ("0.1-0.5Hz", "0.5-2Hz", "2-8Hz", "8-20Hz", "logpower")]
            F["spectral"][0].extend(nm); F["spectral"][1].extend(band_feats(A) + band_feats(Pm))
    if "rssi" in groups:
        tsa = rec["a"]["ts_us"].astype(np.int64); ta = (tsa - tsa.min()) * 1e-6; ra = rec["a"]["rssi"].astype(float)[(ta >= TRIM_S) & (ta <= ta.max() - TRIM_S)]
        r4 = rssi[keep]
        F["rssi"][0].extend(["rssi384_mean", "rssi384_std", "rssi384_min", "rssi384_max", "rssiall_mean", "rssiall_std"])
        F["rssi"][1].extend([r4.mean(), r4.std(), r4.min(), r4.max(), ra.mean(), ra.std()])
    return F


def extract(items, blocks, groups, limit=0, verbose=True):
    meta, mats, names = [], {g: [] for g in groups}, {}
    for i, (npz, mp) in enumerate(items):
        if limit and len(meta) >= limit:
            break
        rec = S.load(npz, mp); F = rec_features(rec, blocks, groups)
        if F is None:
            if verbose:
                print("skip (too few packets):", npz)
            continue
        for g in groups:
            names[g] = F[g][0]; mats[g].append(np.array(F[g][1], float))
        created = rec["meta"].get("created"); ts = dt.datetime.fromisoformat(created).timestamp() if created else float("nan")
        meta.append(dict(session=rec["session"], label=rec["label"], condition=rec["condition"], zone=rec["zone"] or "", path=os.path.abspath(npz),
                         line=rec["meta"].get("protocol_line", -1), created=ts))
    return meta, {g: np.array(m) for g, m in mats.items()}, names


def save(out, meta, mats, names, settings):
    os.makedirs(out, exist_ok=True)
    arr = dict(session=np.array([m["session"] for m in meta]), label=np.array([m["label"] for m in meta]), condition=np.array([m["condition"] for m in meta]),
               zone=np.array([m["zone"] for m in meta]), path=np.array([m["path"] for m in meta]), line=np.array([m["line"] for m in meta]), created=np.array([m["created"] for m in meta]))
    np.savez_compressed(os.path.join(out, "features.npz"), **arr, **{f"G_{g}": v for g, v in mats.items()})
    json.dump(dict(settings=settings, names=names, n_recordings=len(meta)), open(os.path.join(out, "index.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--groups", nargs="+", default=ALL_GROUPS, choices=ALL_GROUPS)
    ap.add_argument("--blocks", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--tag", default="all"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "features"))
    a = ap.parse_args(); items = S.discover(a.data)
    if not items:
        raise SystemExit("no recordings found (check --data)")
    meta, mats, names = extract(items, a.blocks, a.groups, a.limit)
    out = os.path.join(a.out, a.tag); save(out, meta, mats, names, dict(blocks=a.blocks, groups=a.groups, trim_s=TRIM_S, fs=FS, bands=BANDS, packet_type=384, sanitisation="cpo"))
    print(f"{len(meta)} recordings -> {out}\n" + "\n".join(f"  {g:11s} {mats[g].shape[1]:4d} features" for g in a.groups))


if __name__ == "__main__":
    main()
