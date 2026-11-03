#!/usr/bin/env python3
"""
audit_recordings.py - STAGE 1 audit of the original single-link CSI recordings (read-only).

    python audit_recordings.py --data dataset_room                       # every recording
    python audit_recordings.py --data dataset_room --limit 5 --detail    # first 5 only, full per-recording detail
    python audit_recordings.py --npz csi_link.npz --meta meta.json       # one loose file pair
    add --plots to save a diagnostic figure per recording

It answers, per recording and per packet type (csi_len):
  1. SCHEMA   arrays, dtypes, shapes, zero padding, agreement with meta.json
  2. TYPES    which csi_len values occur (128 / 256 / 384 ...), their fractions, how often the type changes
  3. TIMING   host_t (Mac clock at read time) vs ts_us (ESP32 receive clock): drift, latency jitter, irregular sampling, duplicates
  4. LOSS     sequence-number gaps (records the board produced but the Mac never stored)
  5. RSSI     range, mean, per packet type
  6. I/Q      per 128-value block: null (zero) subcarriers, amplitude level, saturation, interleave pairing check
  7. PHASE    raw-phase indicators only: adjacent-subcarrier phase step (the 'slope'), per-packet phase offset randomness,
              raw per-subcarrier phase stability over time  -> tells whether sanitisation is needed (done in a later script)
  8. IMPACT   what the existing zone_eval.py pipeline keeps/drops (it keeps only the most common csi_len)
Outputs (default sura_features/outputs/audit/): audit_summary.csv, audit_details.json, audit_report.txt, plots/*.png
Nothing is written into the dataset. Facts are measured from the files; hypotheses are labelled as such.
"""
import argparse, csv, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sura_io as S

NULL_THR = 0.5        # a subcarrier whose mean amplitude is below this is treated as null
MIN_TYPE_PKTS = 20    # packet types with fewer packets are counted but not analysed in detail


def pct(x, q=(5, 50, 95)):
    x = np.asarray(x, float)
    return [round(float(v), 2) for v in np.percentile(x, q)] + [round(float(x.max()), 2)] if len(x) else [float("nan")] * (len(q) + 1)


def compress(idx):
    """[0,1,2,3,32,59,60] -> '0-3,32,59-60'"""
    idx = list(map(int, idx)); out = []; i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}"); i = j + 1
    return ",".join(out) or "none"


def circ_sd(angles):
    R = abs(np.exp(1j * np.asarray(angles)).mean())
    return float(np.sqrt(-2 * np.log(max(R, 1e-12))))


# --------------------------------------------------------------------------------------------- schema / types
def audit_schema(rec):
    a, meta, out = rec["a"], rec["meta"], {}
    n = len(a["host_t"]); out["n"] = n
    out["dtypes"] = {k: str(a[k].dtype) for k in S.REQUIRED if k in a}
    out["csi_shape"] = list(a["csi"].shape)
    L = a["csi_len"].astype(int)
    out["pad_nonzero_packets"] = int(sum(np.any(a["csi"][i, L[i]:] != 0) for i in range(n)))
    cnt = Counter(L.tolist()); out["len_counts"] = {int(k): int(v) for k, v in sorted(cnt.items())}
    out["len_changes"] = int((np.diff(L) != 0).sum())
    mc = cnt.most_common(1)[0][0]; out["most_common_len"] = int(mc)
    out["dropped_by_common_len_filter"] = int(n - cnt[mc])
    out["odd_or_nonblock_len"] = [int(k) for k in cnt if k % S.BLOCK]
    if meta:
        ms = meta.get("csi_samples", {}).get(rec["meta_key"])
        out["meta_samples"] = ms; out["meta_samples_match"] = (ms == n) if ms is not None else None
        out["meta_malformed_total"] = meta.get("csi_malformed_total", {}).get(rec["meta_key"])
        out["meta_duration_s"] = meta.get("duration_s")
    return out


# --------------------------------------------------------------------------------------------- timing
def audit_timing(a):
    host = a["host_t"].astype(np.float64); n = len(host); out = {}
    dh = np.diff(host)
    out.update(host_dups=int((dh == 0).sum()), host_backwards=int((dh < 0).sum()), host_dt_ms=pct(dh * 1e3), host_span_s=round(float(host[-1] - host[0]), 3))
    d = np.diff(a["ts_us"].astype(np.int64)); d = ((d + 2 ** 31) % 2 ** 32) - 2 ** 31          # signed, wrap-safe
    out["ts_reorder_or_dup"] = int((d <= 0).sum()); s = np.concatenate([[0], np.cumsum(d)]) * 1e-6
    out["ts_dt_ms"] = pct(d * 1e-3); out["ts_span_s"] = round(float(s.max() - s.min()), 3)
    out["rate_host_hz"] = round((n - 1) / max(out["host_span_s"], 1e-9), 1); out["rate_ts_hz"] = round((n - 1) / max(out["ts_span_s"], 1e-9), 1)
    y = host - host[0]; b, c = np.polyfit(s, y, 1)
    out["drift_ppm_host_vs_ts"] = round(float((b - 1) * 1e6), 0)
    r = y - (c + b * s); lat = (r - r.min()) * 1e3
    out["host_minus_ts_residual_ms"] = {"std": round(float(r.std() * 1e3), 2), "p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2), "max": round(float(lat.max()), 2)}
    e, l = s < s.min() + 1.0, s > s.max() - 1.0
    out["residual_first1s_vs_last1s_ms"] = [round(float(np.median(lat[e])), 2), round(float(np.median(lat[l])), 2)]
    out["ts_dt_p95_over_p50"] = round(float(np.percentile(d * 1e-3, 95) / max(np.percentile(d * 1e-3, 50), 1e-9)), 2)
    return out


def audit_seq(a):
    sq = a["seq"].astype(np.int64); n = len(sq); d = np.diff(sq); gaps = d[d > 1] - 1; missing = int(gaps.sum())
    return dict(seq_first=int(sq[0]), seq_last=int(sq[-1]), seq_nonincreasing=int((d <= 0).sum()), gap_events=int(len(gaps)), missing=missing,
                loss_pct=round(100 * missing / max(missing + n, 1), 1), max_burst=int(gaps.max()) if len(gaps) else 0,
                burst_hist={int(k): int(v) for k, v in sorted(Counter(gaps.tolist()).items())[:8]},
                offered_rate_hz=round((n + missing) / max(float(a["host_t"][-1] - a["host_t"][0]), 1e-9), 1))


def audit_rssi(a):
    r = a["rssi"].astype(float); L = a["csi_len"].astype(int)
    return dict(dtype=str(a["rssi"].dtype), min=int(r.min()), max=int(r.max()), mean=round(float(r.mean()), 2), std=round(float(r.std()), 2),
                n_distinct=int(len(np.unique(r))), by_len={int(k): round(float(r[L == k].mean()), 2) for k in sorted(set(L.tolist()))})


# --------------------------------------------------------------------------------------------- I/Q structure + phase indicators
def phase_indicators(X, valid):
    """X [n,128] int8 values. Assumed convention (ESP-IDF docs): first value of each pair = imaginary, second = real.
    Mirroring (swapping the two) flips the sign of the phase; all indicators below are insensitive to that sign."""
    c = X[:, 1::2] + 1j * X[:, 0::2]
    k = np.where(valid[:-1] & valid[1:])[0]
    z = np.exp(1j * np.angle(c[:, k + 1] * np.conj(c[:, k])))
    slope = np.angle(z.mean(1)); vi = np.where(valid)[0]
    phi = np.angle(c[:, vi].sum(1)); Rsc = np.abs(np.exp(1j * np.angle(c[:, vi])).mean(0))
    return dict(adj_step_rad_circ_mean=round(float(np.angle(np.exp(1j * slope).mean())), 4), adj_step_across_packets_sd=round(circ_sd(slope), 3),
                adj_step_within_packet_R_median=round(float(np.median(np.abs(z.mean(1)))), 3),
                packet_mean_phase_R=round(float(abs(np.exp(1j * phi).mean())), 3), subcarrier_phase_R_median=round(float(np.median(Rsc)), 3))


def audit_blocks(a):
    """per packet type (>= MIN_TYPE_PKTS packets) and per 128-value block"""
    L = a["csi_len"].astype(int); res = {}
    for ln in sorted(set(L.tolist())):
        m = L == ln
        if m.sum() < MIN_TYPE_PKTS or ln % S.BLOCK:
            res[int(ln)] = {"n": int(m.sum()), "analysed": False}; continue
        X = a["csi"][m][:, :ln].astype(float); blocks = []
        for b in range(ln // S.BLOCK):
            B = X[:, b * S.BLOCK:(b + 1) * S.BLOCK]
            amp = np.hypot(B[:, 0::2], B[:, 1::2]); prof = amp.mean(0); valid = prof >= NULL_THR
            va = np.where(valid)[0]
            sh = B[:, 1:-1]; amp_sh = np.hypot(sh[:, 0::2], sh[:, 1::2]); vs = amp_sh.mean(0) >= NULL_THR
            rough = float(np.median(np.abs(np.diff(amp[:, va], axis=1)).mean(1))) if len(va) > 2 else float("nan")
            rough_sh = float(np.median(np.abs(np.diff(amp_sh[:, vs], axis=1)).mean(1))) if vs.sum() > 2 else float("nan")
            gain = np.log(amp[:, valid].mean(1) + 1e-6)
            blocks.append(dict(block=b, null_subcarriers=compress(np.where(~valid)[0]), n_valid=int(valid.sum()), mean_amp=round(float(prof[valid].mean()), 2),
                               saturated_pct=round(float(100 * (np.abs(B) >= 127).mean()), 3), roughness_aligned=round(rough, 3), roughness_shifted_by_1=round(rough_sh, 3),
                               packet_gain_sd_log=round(float(gain.std()), 3), **phase_indicators(B, valid), _prof=prof))
        sig = f"{ln}:" + "|".join(f"[{bl['null_subcarriers']}]" for bl in blocks)
        corr = {}
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                v = (blocks[i]["_prof"] >= NULL_THR) & (blocks[j]["_prof"] >= NULL_THR)
                corr[f"b{i}_b{j}"] = round(float(np.corrcoef(blocks[i]["_prof"][v], blocks[j]["_prof"][v])[0, 1]), 3) if v.sum() > 3 else None
        res[int(ln)] = dict(n=int(m.sum()), analysed=True, signature=sig, blocks=blocks, profile_corr_between_blocks=corr)
    return res


def flags_for(r):
    f = []; sc, tm, sq, bl = r["schema"], r["timing"], r["seq"], r["blocks"]
    tot = sc["n"]
    if len([k for k, v in sc["len_counts"].items() if v / tot >= 0.01]) > 1:
        f.append("MIXED_TYPES: " + str({k: f"{100 * v / tot:.0f}%" for k, v in sc["len_counts"].items()}) + " - layouts differ (see block table)")
    if sc["dropped_by_common_len_filter"]:
        f.append(f"COMMON_LEN_FILTER: existing pipeline drops {sc['dropped_by_common_len_filter']} packets ({100 * sc['dropped_by_common_len_filter'] / tot:.1f}%) by keeping only csi_len={sc['most_common_len']}")
    if sc["odd_or_nonblock_len"]:
        f.append(f"ODD_LEN: csi_len not a multiple of 128: {sc['odd_or_nonblock_len']}")
    if sc["pad_nonzero_packets"]:
        f.append(f"PADDING: {sc['pad_nonzero_packets']} packets have non-zero data beyond csi_len")
    if sc.get("meta_samples_match") is False:
        f.append(f"META_MISMATCH: meta.json csi_samples ({sc['meta_samples']}) != npz packets ({tot})")
    if sq["loss_pct"] > 5:
        f.append(f"SEQ_LOSS: {sq['loss_pct']}% ({sq['missing']} records never stored; max burst {sq['max_burst']})")
    if sq["seq_nonincreasing"] or tm["ts_reorder_or_dup"] or tm["host_dups"] or tm["host_backwards"]:
        f.append(f"ORDERING: anomalies: seq {sq['seq_nonincreasing']}, ts {tm['ts_reorder_or_dup']}, host dup {tm['host_dups']}, host backwards {tm['host_backwards']}")
    if tm["host_minus_ts_residual_ms"]["p95"] > 20:
        f.append(f"HOST_CLOCK: host_t departs from the radio clock (p95 {tm['host_minus_ts_residual_ms']['p95']} ms): use ts_us for temporal/spectral work")
    if tm["ts_dt_p95_over_p50"] > 1.5:
        f.append(f"IRREGULAR_SAMPLING: irregular sampling (ts_us dt p95/p50 = {tm['ts_dt_p95_over_p50']}): resample on ts_us before FFT/Doppler")
    if tm["rate_ts_hz"] < 80:
        f.append(f"LOW_RATE: effective rate {tm['rate_ts_hz']} Hz (Nyquist {tm['rate_ts_hz'] / 2:.0f} Hz)")
    for ln, v in bl.items():
        if not v.get("analysed"):
            continue
        for b in v["blocks"]:
            if b["saturated_pct"] > 1:
                f.append(f"CLIPPING: len {ln} block {b['block']}: {b['saturated_pct']}% of values at int8 limit (clipping)")
            if b["packet_mean_phase_R"] < 0.5:
                f.append(f"PHASE_OFFSET_RANDOM: len {ln} block {b['block']}: raw per-packet phase offset is not stable (R={b['packet_mean_phase_R']}) -> sanitisation needed")
    return f


def audit_recording(rec):
    a = rec["a"]
    if rec["problems"] and any("missing arrays" in p for p in rec["problems"]):
        return dict(error=rec["problems"])
    r = dict(schema=audit_schema(rec), timing=audit_timing(a), seq=audit_seq(a), rssi=audit_rssi(a), blocks=audit_blocks(a))
    r["flags"] = flags_for(r); return r


# --------------------------------------------------------------------------------------------- output
def detail_text(rec, r):
    L = [f"\n{'=' * 100}\n{rec['session']}  {rec['label']}   link={rec['link']}   file={rec['path']}"]
    sc, tm, sq, rs = r["schema"], r["timing"], r["seq"], r["rssi"]
    L.append(f"SCHEMA  packets {sc['n']}, csi shape {sc['csi_shape']}, dtypes {sc['dtypes']}")
    L.append(f"        meta csi_samples {sc.get('meta_samples')} (match={sc.get('meta_samples_match')}), meta malformed_total (cumulative since reader start) {sc.get('meta_malformed_total')}")
    L.append(f"TYPES   csi_len counts {sc['len_counts']}, type changes along the recording: {sc['len_changes']}, nonzero padding packets: {sc['pad_nonzero_packets']}")
    L.append(f"TIMING  host span {tm['host_span_s']} s ({tm['rate_host_hz']} Hz) | radio ts_us span {tm['ts_span_s']} s ({tm['rate_ts_hz']} Hz) | apparent host-vs-radio drift {tm['drift_ppm_host_vs_ts']} ppm (a linear fit cannot tell clock drift from steadily growing delay)")
    L.append(f"        dt ms [p5,p50,p95,max]: host {tm['host_dt_ms']}   radio {tm['ts_dt_ms']}   (dups host {tm['host_dups']}, radio reorder {tm['ts_reorder_or_dup']})")
    L.append(f"        host_t minus radio-clock fit (ms): {tm['host_minus_ts_residual_ms']}   median first 1 s vs last 1 s: {tm['residual_first1s_vs_last1s_ms']}")
    L.append(f"LOSS    seq {sq['seq_first']}..{sq['seq_last']}: {sq['gap_events']} gap events, {sq['missing']} missing = {sq['loss_pct']}% of produced records; burst sizes {sq['burst_hist']}; offered rate {sq['offered_rate_hz']} Hz")
    L.append(f"RSSI    dBm min {rs['min']} max {rs['max']} mean {rs['mean']} sd {rs['std']} distinct {rs['n_distinct']}; mean by csi_len {rs['by_len']}")
    for ln, v in r["blocks"].items():
        if not v.get("analysed"):
            L.append(f"I/Q len {ln}: {v['n']} packets (too few to analyse)"); continue
        L.append(f"I/Q     len {ln} ({v['n']} packets), layout signature {v['signature']}, between-block mean-profile corr {v['profile_corr_between_blocks']}")
        for b in v["blocks"]:
            L.append(f"          block {b['block']}: null subcarriers [{b['null_subcarriers']}] valid {b['n_valid']}, mean amp {b['mean_amp']}, clipped {b['saturated_pct']}%, "
                     f"smoothness(aligned/shifted-by-1) {b['roughness_aligned']}/{b['roughness_shifted_by_1']}, packet-gain sd(log) {b['packet_gain_sd_log']}")
            L.append(f"            phase: adjacent-subcarrier step circ-mean {b['adj_step_rad_circ_mean']} rad (sd across packets {b['adj_step_across_packets_sd']}), "
                     f"within-packet R {b['adj_step_within_packet_R_median']}, per-packet mean-phase R {b['packet_mean_phase_R']}, per-subcarrier phase R over time {b['subcarrier_phase_R_median']}")
    L.append("FLAGS   " + ("none" if not r["flags"] else "\n        - " + "\n        - ".join(r["flags"])))
    return "\n".join(L)


def plot_recording(rec, r, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    a = rec["a"]; host = a["host_t"] - a["host_t"][0]; Ln = a["csi_len"].astype(int)
    d = np.diff(a["ts_us"].astype(np.int64)); d = ((d + 2 ** 31) % 2 ** 32) - 2 ** 31; s = np.concatenate([[0], np.cumsum(d)]) * 1e-6
    fig, ax = plt.subplots(2, 3, figsize=(16, 8)); fig.suptitle(f"{rec['session']} {rec['label']} ({rec['link']})")
    for ln in sorted(set(Ln.tolist())):
        ax[0, 0].scatter(host[Ln == ln], np.full((Ln == ln).sum(), ln), s=6, label=f"len {ln}")
    sq = a["seq"].astype(np.int64); gi = np.where(np.diff(sq) > 1)[0]
    ax[0, 0].scatter(host[gi], np.full(len(gi), Ln.min() - 40), marker="|", c="r", s=40, label="seq gap"); ax[0, 0].set_title("packet type over time"); ax[0, 0].legend(fontsize=7); ax[0, 0].set_xlabel("host s")
    ax[0, 1].hist(np.diff(host) * 1e3, bins=60, alpha=.6, label="host_t"); ax[0, 1].hist(d * 1e-3, bins=60, alpha=.6, label="ts_us"); ax[0, 1].set_title("inter-packet interval (ms)"); ax[0, 1].legend()
    b, c = np.polyfit(s, host, 1); res = (host - (c + b * s)) * 1e3
    ax[0, 2].plot(s, res - res.min(), lw=.7); ax[0, 2].set_title("host_t minus radio-clock fit (ms)"); ax[0, 2].set_xlabel("radio s")
    for ln, v in r["blocks"].items():
        if v.get("analysed"):
            X = a["csi"][Ln == ln][:, :ln].astype(float)
            for bi in range(ln // S.BLOCK):
                B = X[:, bi * S.BLOCK:(bi + 1) * S.BLOCK]; ax[1, 0].plot(np.hypot(B[:, 0::2], B[:, 1::2]).mean(0), label=f"len{ln} blk{bi}", lw=1)
    ax[1, 0].set_title("mean amplitude per subcarrier index (null pattern)"); ax[1, 0].set_yscale("log"); ax[1, 0].legend(fontsize=6); ax[1, 0].set_xlabel("subcarrier index in block")
    for ln in sorted(set(Ln.tolist())):
        ax[1, 1].scatter(host[Ln == ln], a["rssi"][Ln == ln], s=6, label=f"len {ln}")
    ax[1, 1].set_title("RSSI (dBm)"); ax[1, 1].legend(fontsize=7)
    ln0 = r["schema"]["most_common_len"]; m = Ln == ln0; B = a["csi"][m][:, :S.BLOCK].astype(float); cc = B[:, 1::2] + 1j * B[:, 0::2]
    prof = np.hypot(B[:, 0::2], B[:, 1::2]).mean(0); vi = np.where(prof >= NULL_THR)[0]
    ax[1, 2].plot(host[m], np.angle(cc[:, vi].sum(1)), ".", ms=3); ax[1, 2].set_title(f"raw mean phase, block 0 (len {ln0})"); ax[1, 2].set_ylim(-3.3, 3.3)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def summary_row(rec, r):
    if "error" in r:
        return dict(session=rec["session"], label=rec["label"], link=rec["link"], error=";".join(r["error"]))
    sc, tm, sq, rs = r["schema"], r["timing"], r["seq"], r["rssi"]; n = sc["n"]; lc = sc["len_counts"]
    row = dict(session=rec["session"], label=rec["label"], condition=rec["condition"], zone=rec["zone"], link=rec["link"], n=n,
               rate_host_hz=tm["rate_host_hz"], rate_ts_hz=tm["rate_ts_hz"], offered_rate_hz=sq["offered_rate_hz"], loss_pct=sq["loss_pct"], max_burst=sq["max_burst"],
               pct_len128=round(100 * lc.get(128, 0) / n, 1), pct_len256=round(100 * lc.get(256, 0) / n, 1), pct_len384=round(100 * lc.get(384, 0) / n, 1),
               len_changes=sc["len_changes"], dropped_by_common_len_pct=round(100 * sc["dropped_by_common_len_filter"] / n, 1),
               host_dups=tm["host_dups"], ts_reorder=tm["ts_reorder_or_dup"], drift_ppm=tm["drift_ppm_host_vs_ts"],
               host_resid_p95_ms=tm["host_minus_ts_residual_ms"]["p95"], ts_dt_p95_over_p50=tm["ts_dt_p95_over_p50"],
               rssi_mean=rs["mean"], rssi_sd=rs["std"], rssi_min=rs["min"], rssi_max=rs["max"],
               meta_samples_match=sc.get("meta_samples_match"), n_flags=len(r["flags"]), path=rec["path"])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data"); ap.add_argument("--npz"); ap.add_argument("--meta")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "audit"))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--detail", action="store_true"); ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    if a.npz:
        items = [(a.npz, a.meta)]
    elif a.data:
        items = S.discover(a.data)
    else:
        ap.error("give --data ROOT or --npz FILE [--meta FILE]")
    if a.limit:
        items = items[:a.limit]
    if not items:
        msg = ["no recordings found."]
        if a.data:
            p = os.path.abspath(a.data); msg.append(f"  looked in: {p}   (exists: {os.path.isdir(p)})")
            if os.path.isdir(p):
                msg.append(f"  contents: {sorted(os.listdir(p))[:12]}")
            msg.append("  Run this from the folder that CONTAINS dataset_room, e.g.  python audit_recordings.py --data dataset_room")
            msg.append("  or pass the full path: --data /full/path/to/dataset_room. To locate it:  find ~ -name meta.json -path '*dataset_room*' 2>/dev/null | head -3")
        raise SystemExit("\n".join(msg))
    os.makedirs(a.out, exist_ok=True)
    detail_on = a.detail or len(items) <= 3
    report, rows, details, sigs, allflags = [], [], {}, Counter(), Counter()
    for npz, mp in items:
        rec = S.load(npz, mp); r = audit_recording(rec)
        rows.append(summary_row(rec, r)); key = npz   # unique per recording (session/label alone collide)
        if "error" in r:
            print(f"ERROR {npz}: {r['error']}"); continue
        for ln, v in r["blocks"].items():
            if v.get("analysed"):
                sigs[v["signature"]] += 1
        for fl in r["flags"]:
            allflags[fl.split(":")[0]] += 1
        for v in r["blocks"].values():
            for b in v.get("blocks", []):
                b.pop("_prof", None)
        details[key] = r
        if detail_on:
            t = detail_text(rec, r); print(t); report.append(t)
        if a.plots:
            os.makedirs(os.path.join(a.out, "plots"), exist_ok=True); plot_recording(rec, r, os.path.join(a.out, "plots", f"{rec['session']}_{rec['label']}_{os.path.basename(os.path.dirname(npz))}.png"))
    good = [x for x in rows if "error" not in x]
    head = f"\n{'=' * 100}\nSUMMARY: {len(rows)} file(s)\n"
    cols = ["session", "label", "n", "rate_ts_hz", "loss_pct", "pct_len128", "pct_len256", "pct_len384", "dropped_by_common_len_pct", "host_resid_p95_ms", "rssi_mean", "n_flags"]
    tbl = " ".join(f"{c:>12s}" for c in cols) + "\n" + "\n".join(" ".join(f"{str(x.get(c, ''))[:12]:>12s}" for c in cols) for x in good)
    sg = "\nLAYOUT SIGNATURES (csi_len: null subcarriers per 128-value block) - count of recordings:\n" + "\n".join(f"  {v:4d} x {k}" for k, v in sigs.most_common())
    fl = "\nFLAG FREQUENCY:\n" + "\n".join(f"  {v:4d} x {k}" for k, v in allflags.most_common())
    out = head + tbl + sg + fl; print(out); report.append(out)
    with open(os.path.join(a.out, "audit_summary.csv"), "w", newline="") as f:
        keys = sorted({k for x in rows for k in x}, key=lambda k: (k not in ("session", "label"), k)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(details, open(os.path.join(a.out, "audit_details.json"), "w"), indent=1); open(os.path.join(a.out, "audit_report.txt"), "w").write("\n".join(report))
    print(f"\nwritten to {a.out}: audit_summary.csv, audit_details.json, audit_report.txt" + (", plots/" if a.plots else ""))


if __name__ == "__main__":
    main()
