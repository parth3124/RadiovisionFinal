"""
dataset.py - load labelled recordings, synchronize CSI with camera heatmaps, cut windows,
and split by session/recording (never by window) to avoid leakage.

Recording folder (written by collect.py):
    dataset/<gesture>/<recording_id>/
        csi.npz       host_t, seq, ts_us, rssi, csi (raw int8), csi_len
        camera.npz    host_t, kp33, joints, ok, heatmaps (uint8 [N,14,32,32])   (optional)
        video.mp4     preview video                                              (optional)
        meta.json     gesture, label, session, recording_id, ...

Synchronization
    * Every CSI line and every camera frame is stamped with the same Mac clock
      (time.monotonic) the moment it arrives.
    * USB delivery adds a few ms of jitter to CSI arrival times, so the CSI times are
      re-derived from the RX radio's own microsecond clock (ts_us) with a linear fit
      to the Mac clock (removes jitter, keeps the Mac time base).
    * CSI is resampled onto a uniform 100 Hz grid (last sample at or before each grid
      point). Grid points more than 0.1 s from a real sample are marked invalid.

Windows
    * One window = window_sec of CSI (default 1.0 s = 100 grid steps x S subcarriers),
      taken every stride_sec (default 0.25 s) along the recording.
    * Its gesture label = the recording's label.
    * Its heatmap target = the average of the teacher heatmaps of all camera frames whose
      timestamp falls inside the same window (t_end - window_sec, t_end]. A joint counts
      as visible for the window if the teacher saw it in >= 50% of those frames; the
      window has a heatmap target only if >= 3 camera frames had a pose.
"""
from __future__ import annotations
import glob, json, os
from collections import Counter, defaultdict
import numpy as np
import common as C
import heatmaps as HM


class Recording:
    def __init__(self, path, window_sec, stride_sec, trim_sec=0.5, fs=C.CSI_FS):
        self.path = os.path.normpath(path)
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.gesture = self.meta["gesture"]
        if self.gesture not in C.CLASS_TO_IDX:
            raise ValueError(f"unknown gesture {self.gesture!r}")
        self.label = C.CLASS_TO_IDX[self.gesture]
        self.session = self.meta.get("session", "default")
        self.rec_id = self.meta.get("recording_id", os.path.basename(self.path))
        self.fs = fs
        self.T = int(round(window_sec * fs))

        # ---- CSI -> uniform grid ----
        d = np.load(os.path.join(path, "csi.npz"))
        if "amp" in d:                                          # legacy recordings (no timestamps)
            amp = d["amp"].astype(np.float32)
            t = np.arange(len(amp)) / fs
            self.legacy = True
        else:
            lens = d["csi_len"]
            L = Counter(lens.tolist()).most_common(1)[0][0]
            keep = lens == L
            amp = C.csi_amplitude(d["csi"][keep, :L])
            t = self._sync_times(d["host_t"][keep], d["ts_us"][keep])
            self.legacy = False
        self.n_sub = amp.shape[1]
        self.t_csi = t
        g0, g1 = t[0] + trim_sec, t[-1] - trim_sec
        self.grid = np.arange(g0, g1, 1.0 / fs) if g1 > g0 else np.zeros(0)
        idx = np.clip(np.searchsorted(t, self.grid, side="right") - 1, 0, len(t) - 1)
        self.valid = (self.grid - t[idx] <= C.MAX_GAP_SEC) & (self.grid >= t[0])
        self.G = amp[idx].astype(np.float16)                     # [M, S]
        self.norm_mode = "global"                                # set by apply_norm()
        self.raw_mean = self.G[self.valid].astype(np.float32).mean(0) if self.valid.any() else np.zeros(self.n_sub, np.float32)

        # ---- windows ----
        bad = np.concatenate([[0], np.cumsum(~self.valid)])
        step = max(1, int(round(stride_sec * fs)))
        ends = [e for e in range(self.T - 1, len(self.grid), step) if bad[e + 1] - bad[e - self.T + 1] == 0]
        self.ends = np.array(ends, int)

        # ---- camera teacher targets for each window ----
        n = len(self.ends)
        self.hm = np.zeros((n, HM.J, HM.HM_SIZE, HM.HM_SIZE), np.uint8)
        self.jmask = np.zeros((n, HM.J), bool)
        self.has_hm = np.zeros(n, bool)
        cam = os.path.join(path, "camera.npz")
        self.has_camera = os.path.exists(cam) and not self.legacy
        if self.has_camera:
            c = np.load(cam)
            ct, ok, joints, chm = c["host_t"], c["ok"], c["joints"], c["heatmaps"]
            vis = HM.joint_mask(joints) & ok[:, None]
            for i, e in enumerate(self.ends):
                te = self.grid[e]
                lo, hi = np.searchsorted(ct, te - window_sec, "right"), np.searchsorted(ct, te, "right")
                sel = np.arange(lo, hi)[ok[lo:hi]]
                if len(sel) < 3:
                    continue
                self.hm[i] = np.round(chm[sel].astype(np.float32).mean(0)).astype(np.uint8)
                self.jmask[i] = vis[sel].mean(0) >= 0.5
                self.has_hm[i] = self.jmask[i].any()

    @staticmethod
    def _sync_times(host_t, ts_us):
        """Mac arrival times de-jittered with the radio clock (handles u32 wrap-around)."""
        ts = ts_us.astype(np.int64)
        ts = np.concatenate([[0], np.cumsum(np.diff(ts) % (1 << 32))]) * 1e-6
        if len(ts) < 20:
            return host_t
        a, b = np.polyfit(ts, host_t, 1)
        t = a * ts + b
        t += np.percentile(host_t - t, 5)          # USB only delays, so align to the earliest arrivals
        return np.maximum.accumulate(t)

    def window(self, k, shift=0):
        e = int(np.clip(self.ends[k] + shift, self.T - 1, len(self.G) - 1))
        if shift and not self.valid[e - self.T + 1:e + 1].all():
            e = self.ends[k]
        return self.G[e - self.T + 1:e + 1]


def find_recordings(root):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "*", "meta.json")))


def load_recordings(root, window_sec, stride_sec, verbose=True):
    recs = []
    for p in find_recordings(root):
        try:
            r = Recording(p, window_sec, stride_sec)
        except Exception as e:
            print(f"  skip {p}: {e}"); continue
        if len(r.ends) == 0:
            print(f"  skip {p}: shorter than one window"); continue
        recs.append(r)
    if not recs:
        raise SystemExit(f"no usable recordings under {root}/")
    S = Counter(r.n_sub for r in recs).most_common(1)[0][0]
    odd = [r for r in recs if r.n_sub != S]
    for r in odd:
        print(f"  skip {r.path}: {r.n_sub} subcarriers (expected {S})")
    recs = [r for r in recs if r.n_sub == S]
    if verbose:
        summarize(recs)
    return recs


def summarize(recs):
    by = defaultdict(lambda: [0, 0, 0, set()])
    for r in recs:
        b = by[r.gesture]; b[0] += 1; b[1] += len(r.ends); b[2] += int(r.has_hm.sum()); b[3].add(r.session)
    print(f"{'gesture':<14}{'recordings':>11}{'windows':>9}{'w/ heatmap':>12}  sessions")
    for c in C.CLASSES:
        b = by.get(c, [0, 0, 0, set()])
        print(f"{c:<14}{b[0]:>11}{b[1]:>9}{b[2]:>12}  {sorted(b[3])}")


# ---------------------------------------------------------------------------
# Splits (by session, falling back to by recording) - never split one recording
# ---------------------------------------------------------------------------
def split(recs, test_sessions=None, seed=0, val_mode="recording"):
    """-> (train_ids, val_ids, test_ids, description). val = 1 recording per class from
    the non-test data, used only for early stopping / model selection."""
    rng = np.random.default_rng(seed)
    sessions = sorted({r.session for r in recs})
    if test_sessions:
        test = [i for i, r in enumerate(recs) if r.session in set(test_sessions)]
        how = f"test = session(s) {sorted(set(test_sessions))}"
    elif len(sessions) >= 2:
        newest = max(sessions, key=lambda s: max(r.meta.get("created", "") for r in recs if r.session == s))
        test = [i for i, r in enumerate(recs) if r.session == newest]
        how = f"test = newest session '{newest}'"
    else:
        test = []
        for c in range(len(C.CLASSES)):
            ids = [i for i, r in enumerate(recs) if r.label == c]
            rng.shuffle(ids)
            if len(ids) >= 3:
                test += ids[:max(1, round(0.2 * len(ids)))]
        how = "only one session: test = ~20% of the recordings of each class"
    rest = [i for i in range(len(recs)) if i not in set(test)]
    rest_sessions = sorted({recs[i].session for i in rest})
    val = []
    if val_mode == "session" and len(rest_sessions) >= 2:
        # validate on a WHOLE session the model does not train on (newest remaining one), so that
        # early stopping / model selection rewards cross-session transfer, not session memorization
        vs = max(rest_sessions, key=lambda s: max(recs[i].meta.get("created", "") for i in rest if recs[i].session == s))
        val = [i for i in rest if recs[i].session == vs]
        how += f"; val = whole session '{vs}'"
    elif val_mode == "none":
        how += "; no validation set (fixed epochs, last epoch kept)"
    else:
        for c in range(len(C.CLASSES)):
            ids = [i for i in rest if recs[i].label == c]
            if len(ids) >= 2:
                val.append(ids[rng.integers(len(ids))])
        how += "; val = 1 recording/class from the SAME sessions as training"
    train = [i for i in rest if i not in set(val)]
    return train, val, test, how


def cv_folds(recs):
    """Leave-one-session-out folds."""
    return [[i for i, r in enumerate(recs) if r.session == s] for s in sorted({r.session for r in recs})]


def items_of(recs, ids):
    return [(i, k) for i in ids for k in range(len(recs[i].ends))]


def norm_stats(recs, ids):
    """Per-subcarrier mean/std from training recordings only; drop near-empty (null) subcarriers."""
    if recs[ids[0]].norm_mode != "global":      # inputs are already normalized by apply_norm()
        rm = np.mean([recs[i].raw_mean for i in ids], 0)
        S = len(rm)
        return np.zeros(S, np.float32), np.ones(S, np.float32), rm > 0.05 * np.median(rm)
    A = np.concatenate([recs[i].G[recs[i].valid].astype(np.float32) for i in ids])
    mu, sd = A.mean(0), A.std(0)
    keep = mu > 0.05 * np.median(mu)
    return mu, sd + 1.0, keep


def make_batch(recs, items, augment=False, rng=None, aug_offset=0.0):
    """-> x [B,T,S] float32, y [B] int64, hm [B,14,32,32] float32, jmask [B,14] float32, has_hm [B] float32"""
    X, Y, H, JM, HH = [], [], [], [], []
    for i, k in items:
        r = recs[i]
        shift = int(rng.integers(-5, 6)) if augment else 0           # +-50 ms jitter
        X.append(r.window(k, shift)); Y.append(r.label)
        H.append(r.hm[k]); JM.append(r.jmask[k]); HH.append(r.has_hm[k])
    x = np.stack(X).astype(np.float32)
    if augment:
        if recs[items[0][0]].norm_mode == "global":                  # raw amplitude units
            x *= 1 + 0.05 * rng.standard_normal((len(x), 1, 1)).astype(np.float32)   # gain change
            x += 0.5 * rng.standard_normal(x.shape).astype(np.float32)               # amplitude noise
        else:                                                        # z-scored units (std ~ 1)
            x += 0.05 * rng.standard_normal(x.shape).astype(np.float32)
            if aug_offset > 0:   # constant per-window, per-subcarrier shift = residual static-channel change
                x += aug_offset * rng.standard_normal((len(x), 1, x.shape[2])).astype(np.float32)
    return (x, np.array(Y, np.int64), np.stack(H).astype(np.float32) / 255.0,
            np.stack(JM).astype(np.float32), np.array(HH, np.float32))


# ---------------------------------------------------------------------------
# Session-invariant input normalization
# ---------------------------------------------------------------------------
def apply_norm(recs, mode):
    """In-place normalization of every recording's CSI grid G. Uses NO labels.

    global  : unchanged. Raw amplitude; the model z-scores with training-set statistics (old behaviour).
    packet  : log(1 + amplitude) minus that packet's mean over live subcarriers. Removes the overall
              gain (AGC, TX-RX distance, path loss) of each packet. Deployable sample by sample.
    session : packet, then per-subcarrier z-score with statistics pooled over ALL recordings of the same
              session (all classes together). Removes the static multipath fingerprint of one TX/RX
              placement, which is what separates sessions, while keeping differences between gestures
              inside the session. Needs a short unlabeled calibration at a new placement.
    Note: per-RECORDING normalization is deliberately not offered: a gesture is held for the whole
    recording, so subtracting the recording mean would delete the signal."""
    if mode not in ("global", "packet", "session"):
        raise ValueError(mode)
    for r in recs:
        r.norm_mode = mode
    if mode == "global":
        return recs
    rm = np.mean([r.raw_mean for r in recs], 0)
    live = rm > 0.05 * np.median(rm)                    # structural null subcarriers (same hardware everywhere)
    for r in recs:
        L = np.log1p(r.G.astype(np.float32))
        L -= L[:, live].mean(1, keepdims=True)
        L[:, ~live] = 0
        r.G = L.astype(np.float16)
    if mode == "session":
        by = defaultdict(list)
        for r in recs:
            by[r.session].append(r)
        for sess, rs in by.items():
            A = np.concatenate([r.G[r.valid].astype(np.float32) for r in rs])
            mu, sd = A.mean(0), A.std(0) + 1e-2
            for r in rs:
                r.G = (((r.G.astype(np.float32) - mu) / sd) * live).astype(np.float16)
    return recs