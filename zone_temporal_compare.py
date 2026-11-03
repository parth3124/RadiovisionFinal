#!/usr/bin/env python3
"""
zone_temporal_compare.py - controlled comparison of CSI person-zone classifiers across sessions.

    python zone_temporal_compare.py --data dataset_room --inspect          # print the metadata schema, then exit
    python zone_temporal_compare.py --data dataset_room --models A B1 B2 B3 --norms packet none   # no torch needed
    python zone_temporal_compare.py --data dataset_room --smoke            # fast crash test of the neural part
    python zone_temporal_compare.py --data dataset_room                    # full run: A B1 B2 B3 C, 3 seeds

All models consume the SAME tensor X[N, T=400, S_live] (see build_sequences). Differences between models are then
only (a) which features are extracted and (b) the classifier.
  A   logistic regression on per-subcarrier [mean, std]                   (re-implementation of the baseline, same path)
  B1  A + per-subcarrier mean |first difference|                           (adds temporal-difference energy)
  B2  per-subcarrier [std, mean |first difference|]                        (dynamics only, no static mean)
  B3  compact band descriptors: 8 subcarrier bands x [mean, std, mean|d|, rms d, 4 spectral log-powers]
  C   compact CNN-BiLSTM, epochs chosen by 3-fold stratified CV INSIDE the training session, then retrained on all of it
Protocol: all ordered pairs of sessions (r01->r02, ...). Live subcarriers, standardization and (for C) epoch selection use the
TRAINING session only. The test session is only transformed and predicted. No hyperparameter is chosen with test data.
"""
import argparse, csv, glob, itertools, json, os, sys, time
from collections import Counter, defaultdict
import numpy as np
import common as C

ZONES = ["Z1", "Z2", "Z3"]
FS = 50.0                       # Hz, uniform grid
SEG_SEC = 8.0                   # central segment used by every model
T = int(round(FS * SEG_SEC))    # 400 steps
CROP = 360                      # CNN training crop / eval centre crop
OFF = (T - CROP) // 2
TRIM_SEC = 0.5
MIN_PACKETS = 200
MAX_GAP_SEC = 0.5
LIVE_FRAC = 0.05                # live subcarrier = mean amplitude > LIVE_FRAC * median over subcarriers (as before)


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
class Rec:
    pass


def parse_meta(meta):
    """explicit schema: condition + zone (written by multi_collect.py); fall back to label 'person_Z1' and report it"""
    if "condition" in meta:
        return meta.get("condition"), meta.get("zone"), "fields"
    lab = str(meta.get("label", ""))
    if "_Z" in lab:
        c, _, z = lab.partition("_Z")
        return c, "Z" + z, "label-fallback"
    return (lab or None), None, "label-fallback"


def load_recording(d, meta, session):
    files = sorted(glob.glob(os.path.join(d, "csi_*.npz")))
    if len(files) != 1:
        return None, f"expected exactly one csi_*.npz, found {len(files)}"
    z = np.load(files[0])
    for k in ("host_t", "csi", "csi_len", "ts_us"):
        if k not in z.files:
            return None, f"missing array {k}"
    lens = z["csi_len"]
    L = Counter(lens.tolist()).most_common(1)[0][0]
    keep = lens == L
    n_len_dropped = int((~keep).sum())
    t = z["host_t"][keep].astype(np.float64); csi = z["csi"][keep, :L]; ts_us = z["ts_us"][keep]
    if len(t) < 10 or not np.all(np.isfinite(t)):
        return None, "too few or non-finite timestamps"
    o = np.argsort(t, kind="stable"); t, csi, ts_us = t[o], csi[o], ts_us[o]
    inc = np.concatenate([[True], np.diff(t) > 0]); n_dup = int((~inc).sum())
    t, csi, ts_us = t[inc], csi[inc], ts_us[inc]
    radio_dt = (np.diff(ts_us.astype(np.int64)) % (1 << 32)) * 1e-6
    keep_t = (t >= t[0] + TRIM_SEC) & (t <= t[-1] - TRIM_SEC)
    t, csi = t[keep_t], csi[keep_t]
    if len(t) < MIN_PACKETS:
        return None, f"only {len(t)} packets after trimming (< {MIN_PACKETS})"
    dur = t[-1] - t[0]
    if dur < SEG_SEC:
        return None, f"duration after trimming {dur:.2f} s < {SEG_SEC} s"
    gaps = np.diff(t)
    if gaps.max() > MAX_GAP_SEC:
        return None, f"packet gap of {gaps.max():.2f} s (> {MAX_GAP_SEC} s)"
    r = Rec()
    r.id = os.path.basename(d); r.session = session; r.t = t; r.A = C.csi_amplitude(csi)
    tc = 0.5 * (t[0] + t[-1]); r.grid = tc - SEG_SEC / 2 + np.arange(T) / FS
    assert r.grid[0] >= t[0] - 1e-9 and r.grid[-1] <= t[-1] + 1e-9
    r.stats = dict(dur=dur, rate=(len(t) - 1) / dur, host_dt=float(np.median(gaps)),
                   radio_dt=float(np.median(radio_dt)) if len(radio_dt) else float("nan"),
                   max_gap=float(gaps.max()), n_dup=n_dup, n_len_dropped=n_len_dropped, n_sub=r.A.shape[1])
    return r, None


def load_all(root):
    recs, skipped, other, fallback = [], [], 0, 0
    for mp in sorted(glob.glob(os.path.join(root, "*", "*", "meta.json"))):
        d = os.path.dirname(mp)
        try:
            meta = json.load(open(mp))
        except Exception as e:
            skipped.append((d, f"meta.json unreadable: {e}")); continue
        cond, zone, how = parse_meta(meta)
        if cond != "person":
            other += 1; continue
        fallback += how != "fields"
        sess = meta.get("session")
        if not sess:
            skipped.append((d, "no session field")); continue
        if zone not in ZONES:
            skipped.append((d, f"person recording with zone {zone!r}")); continue
        if not os.path.basename(d).startswith(str(sess) + "_"):
            print(f"  warning: folder {os.path.basename(d)} does not start with session {sess}")
        r, why = load_recording(d, meta, sess)
        if r is None:
            skipped.append((d, why)); continue
        r.zone = ZONES.index(zone)
        recs.append(r)
    return recs, skipped, other, fallback


def inspect(root):
    paths = sorted(glob.glob(os.path.join(root, "*", "*", "meta.json")))
    print(f"{len(paths)} meta.json files under {root}/")
    for p in paths[:2]:
        m = json.load(open(p)); print(os.path.dirname(p)); print("  keys:", sorted(m)); print("  condition/zone/session/label:",
              m.get("condition"), m.get("zone"), m.get("session"), m.get("label"))
        print("  csi files:", [os.path.basename(f) for f in sorted(glob.glob(os.path.join(os.path.dirname(p), '*.npz')))])
    cnt = Counter()
    for p in paths:
        m = json.load(open(p)); cnt[(m.get("session"), m.get("condition"), m.get("zone"))] += 1
    print("\n(session, condition, zone) -> count")
    for k, v in sorted(cnt.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {v}")


# ----------------------------------------------------------------------------------------------
# shared preprocessing (everything fitted on the TRAINING session only)
# ----------------------------------------------------------------------------------------------
def live_mask(train_recs):
    m = np.mean([r.A.mean(0) for r in train_recs], 0)
    return m > LIVE_FRAC * np.median(m)


def resample(t, X, grid):
    n = len(t)
    j = np.clip(np.searchsorted(t, grid, side="right"), 1, n - 1)
    t0, t1 = t[j - 1], t[j]
    w = ((grid - t0) / np.maximum(t1 - t0, 1e-12))[:, None]
    return X[j - 1] * (1 - w) + X[j] * w


def build_sequences(recs, live, norm):
    """packet-level normalization first, then resampling -> [N, T, S_live] float32 (not yet standardized)"""
    out = []
    for r in recs:
        Lg = np.log1p(r.A)
        if norm == "packet":
            Lg = Lg - Lg[:, live].mean(1, keepdims=True)
        elif norm != "none":
            raise ValueError(norm)
        out.append(resample(r.t, Lg[:, live], r.grid))
    X = np.stack(out).astype(np.float32)
    assert X.ndim == 3 and X.shape[1] == T and X.shape[2] == int(live.sum()), X.shape
    return X


def fit_stats(X):
    return X.mean((0, 1)), X.std((0, 1)) + 1e-3


def standardize(X, mu, sd):
    return np.clip((X - mu) / sd, -8, 8).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# features for the logistic-regression models
# ----------------------------------------------------------------------------------------------
def band_features(X, nb=8):
    N, Tn, S = X.shape
    groups = np.array_split(np.arange(S), nb)
    dX = np.diff(X, axis=1)
    mean_s, std_s = X.mean(1), X.std(1)
    mad_s, rms_s = np.abs(dX).mean(1), np.sqrt((dX ** 2).mean(1))
    Xd = X - X.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(Xd * np.hanning(Tn)[None, :, None], axis=1)) ** 2
    f = np.fft.rfftfreq(Tn, 1.0 / FS)
    spec = np.stack([P[:, (f >= lo) & (f < hi), :].sum(1) for lo, hi in [(0.2, 1), (1, 3), (3, 8), (8, 20)]], 1)  # N,4,S
    feats = []
    for g in groups:
        feats += [mean_s[:, g].mean(1), std_s[:, g].mean(1), mad_s[:, g].mean(1), rms_s[:, g].mean(1)]
        feats += [np.log10(spec[:, k, g].mean(1) + 1e-6) for k in range(4)]
    return np.stack(feats, 1)


def make_features(name, X):
    mean_s, std_s = X.mean(1), X.std(1)
    mad_s = np.abs(np.diff(X, axis=1)).mean(1)
    if name == "A":
        return np.concatenate([mean_s, std_s], 1)
    if name == "B1":
        return np.concatenate([mean_s, std_s, mad_s], 1)
    if name == "B2":
        return np.concatenate([std_s, mad_s], 1)
    if name == "B3":
        return band_features(X)
    raise ValueError(name)


def fit_predict_lr(Ftr, ytr, Fte):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
    return m.fit(Ftr, ytr).predict(Fte)


# ----------------------------------------------------------------------------------------------
# neural model C (torch imported lazily)
# ----------------------------------------------------------------------------------------------
def build_cnn(S, arch):
    import torch.nn as nn
    import torch.nn.functional as F

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            if arch == "compact":
                c1, c2, hid, fc_h, drop = 16, 24, 16, 0, 0.5
            else:                                   # the larger topology described in the report
                c1, c2, hid, fc_h, drop = 32, 48, 32, 32, 0.4
            self.conv1 = nn.Conv1d(S, c1, 7, padding=3)
            self.conv2 = nn.Conv1d(c1, c2, 5, padding=2)
            self.pool = nn.MaxPool1d(4)
            self.lstm = nn.LSTM(c2, hid, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(drop)
            self.pre_drop = fc_h == 0                              # 'existing' head carries its own dropout
            self.head = nn.Linear(2 * hid, 3) if fc_h == 0 else nn.Sequential(
                nn.Linear(2 * hid, fc_h), nn.ReLU(), nn.Dropout(drop), nn.Linear(fc_h, 3))

        def forward(self, x, verbose=False):                       # x: (B, T, S)
            assert x.dim() == 3, x.shape
            h = x.transpose(1, 2)                                  # (B, S, T)  Conv1d wants (batch, channels, time)
            h = self.pool(F.relu(self.conv1(h)))                   # (B, c1, T/4)
            h = self.pool(F.relu(self.conv2(h)))                   # (B, c2, T/16)
            h = h.transpose(1, 2)                                  # (B, T/16, c2)  LSTM(batch_first) wants (batch, time, features)
            h, _ = self.lstm(h)                                    # (B, T/16, 2*hid)
            h = h.mean(1)                                          # temporal mean pooling
            if verbose:
                print(f"    shapes: input {tuple(x.shape)} -> conv input {tuple(x.transpose(1, 2).shape)} -> pooled sequence "
                      f"{tuple(h.shape)} after LSTM+mean (B, 2*hidden)")
            return self.head(self.drop(h) if self.pre_drop else h)
    return CRNN()


_SHAPES_PRINTED = False


def train_cnn(Xtr, ytr, epochs, seed, a, Xva=None, yva=None):
    global _SHAPES_PRINTED
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dev = torch.device(a.device)
    model = build_cnn(Xtr.shape[2], a.c_arch).to(dev)
    if not _SHAPES_PRINTED:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"    CNN-BiLSTM ({a.c_arch}): {n_par} parameters, input channels {Xtr.shape[2]}")
        model.eval()
        with torch.no_grad():
            model(torch.zeros(2, CROP, Xtr.shape[2], device=dev), verbose=True)
        _SHAPES_PRINTED = True
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.97 ** e)   # independent of the total number of epochs
    Xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(ytr.astype(np.int64)).to(dev)
    curve = []
    for ep in range(1, epochs + 1):
        model.train(); perm = rng.permutation(len(Xtr)); tl, nb = 0.0, 0
        for i in range(0, len(perm), a.batch):
            idx = torch.from_numpy(perm[i:i + a.batch]).to(dev)
            off = int(rng.integers(0, T - CROP + 1))
            xb = Xt[idx, off:off + CROP, :]
            xb = xb + 0.05 * torch.randn_like(xb)
            loss = F.cross_entropy(model(xb), yt[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        row = {"epoch": ep, "train_loss": tl / max(nb, 1)}
        if Xva is not None:
            pr = predict_proba(model, Xva, dev)
            row["val_loss"] = float(-np.mean(np.log(np.clip(pr[np.arange(len(yva)), yva], 1e-8, 1))))
            row["val_acc"] = float(np.mean(pr.argmax(1) == yva))
        curve.append(row)
    return model, curve


def predict_proba(model, X, dev):
    import torch
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.ascontiguousarray(X[:, OFF:OFF + CROP, :])).to(dev)
        return torch.softmax(model(xb), 1).cpu().numpy()


def select_epochs(Xraw_tr, ytr, seed, a, curves_out, tag):
    """3-fold stratified CV inside the training session -> number of epochs (smoothed mean validation loss)"""
    from sklearn.model_selection import StratifiedKFold
    k = int(min(3, np.bincount(ytr).min()))
    if k < 2:
        return None, "too few recordings per class for inner validation"
    all_curves = []
    for f, (itr, iva) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(Xraw_tr, ytr)):
        mu, sd = fit_stats(Xraw_tr[itr])
        _, cv = train_cnn(standardize(Xraw_tr[itr], mu, sd), ytr[itr], a.max_epochs, seed * 100 + f, a,
                          standardize(Xraw_tr[iva], mu, sd), ytr[iva])
        all_curves.append(cv)
        for row in cv:
            curves_out.append({**tag, "fold": f, **row})
    vl = np.mean([[r["val_loss"] for r in cv] for cv in all_curves], 0)
    va = np.mean([[r["val_acc"] for r in cv] for cv in all_curves], 0)
    sm = np.convolve(np.pad(vl, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")
    lo = min(a.min_epochs, len(sm))
    best = int(np.argmin(sm[lo - 1:])) + lo
    info = {"folds": k, "n_inner_train": int(len(ytr) - len(ytr) // k), "chosen_epochs": best,
            "val_loss_at_choice": float(vl[best - 1]), "val_acc_at_choice": float(va[best - 1]),
            "val_acc_chance_level": bool(va[best - 1] < 0.45)}
    return best, info


# ----------------------------------------------------------------------------------------------
# metrics and reporting
# ----------------------------------------------------------------------------------------------
def metrics(y, p):
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    labs = [0, 1, 2]
    pr, rc, _, _ = precision_recall_fscore_support(y, p, labels=labs, zero_division=0)
    return {"n": int(len(y)), "accuracy": float(np.mean(y == p)), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=labs, average="macro", zero_division=0)),
            "precision": pr.tolist(), "recall": rc.tolist(), "confusion": confusion_matrix(y, p, labels=labs).tolist(),
            "pred_counts": np.bincount(p, minlength=3).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_room")
    ap.add_argument("--out", default="results_temporal")
    ap.add_argument("--models", nargs="+", default=["A", "B1", "B2", "B3", "C"], choices=["A", "B1", "B2", "B3", "C"])
    ap.add_argument("--norms", nargs="+", default=["packet", "none"], choices=["packet", "none"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-epochs", dest="max_epochs", type=int, default=60)
    ap.add_argument("--min-epochs", dest="min_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--c-arch", dest="c_arch", choices=["compact", "existing"], default="compact")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 3 epochs: only checks that the code runs")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.data); return
    if a.smoke:
        a.seeds, a.max_epochs, a.min_epochs = [a.seeds[0]], 3, 2
    run_dir = os.path.join(a.out, time.strftime("run_%Y%m%d_%H%M%S") + ("_smoke" if a.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)
    np.random.seed(0)

    # ---- load + data quality ----
    recs, skipped, other, fallback = load_all(a.data)
    print(f"loaded {len(recs)} person recordings; {other} non-person recordings ignored (not part of this task); "
          f"{len(skipped)} person recordings skipped; {fallback} parsed via label fallback")
    for d, why in skipped:
        print(f"  SKIPPED {d}: {why}")
    sessions = sorted({r.session for r in recs})
    by = {s: [r for r in recs if r.session == s] for s in sessions}
    print(f"\n{'session':<8}{'n':>4}{'Z1':>4}{'Z2':>4}{'Z3':>4}{'dur s':>8}{'rate Hz (min/med/max)':>26}{'host dt ms':>12}{'radio dt ms':>13}{'max gap s':>11}{'dups':>6}{'len-dropped':>13}")
    for s in sessions:
        st = [r.stats for r in by[s]]; z = np.bincount([r.zone for r in by[s]], minlength=3)
        rates = [x["rate"] for x in st]
        print(f"{s:<8}{len(by[s]):>4}{z[0]:>4}{z[1]:>4}{z[2]:>4}{np.median([x['dur'] for x in st]):>8.2f}"
              f"{min(rates):>10.1f}/{np.median(rates):.1f}/{max(rates):.1f}{1e3 * np.median([x['host_dt'] for x in st]):>12.2f}"
              f"{1e3 * np.median([x['radio_dt'] for x in st]):>13.2f}{max(x['max_gap'] for x in st):>11.2f}"
              f"{sum(x['n_dup'] for x in st):>6}{sum(x['n_len_dropped'] for x in st):>13}")
    ratio = [r.stats["radio_dt"] / r.stats["host_dt"] for r in recs]
    if not (0.8 < np.median(ratio) < 1.25):
        print(f"  WARNING: radio-clock and host-clock packet intervals differ (median ratio {np.median(ratio):.2f}); check time units")
    usable = [s for s in sessions if np.bincount([r.zone for r in by[s]], minlength=3).min() >= 3]
    if len(usable) < 2:
        raise SystemExit("need at least two sessions with >= 3 recordings in every zone")
    if set(usable) != set(sessions):
        print(f"  dropping sessions without all zones represented: {sorted(set(sessions) - set(usable))}")
    sessions = usable
    directions = [(tr, te) for tr in sessions for te in sessions if tr != te]
    print(f"\n{len(directions)} directions: " + ", ".join(f"{tr}->{te}" for tr, te in directions))
    masks = {s: live_mask(by[s]) for s in sessions}
    print("live subcarriers per training session: " + ", ".join(f"{s}={int(m.sum())}" for s, m in masks.items()))
    for s1, s2 in itertools.combinations(sessions, 2):
        j = (masks[s1] & masks[s2]).sum() / max((masks[s1] | masks[s2]).sum(), 1)
        print(f"  live-mask overlap (Jaccard) {s1} vs {s2}: {j:.2f}")

    json.dump({"args": vars(a), "constants": dict(FS=FS, SEG_SEC=SEG_SEC, T=T, CROP=CROP, TRIM_SEC=TRIM_SEC, LIVE_FRAC=LIVE_FRAC,
               MAX_GAP_SEC=MAX_GAP_SEC, MIN_PACKETS=MIN_PACKETS), "sessions": sessions, "directions": directions,
               "numpy": np.__version__}, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    rows, preds, curves, epoch_info = [], [], [], []

    def record(model, norm, seed, tr, te, ids, y, p):
        m = metrics(y, p)
        rows.append({"model": model, "norm": norm, "seed": seed, "train": tr, "test": te, **m})
        for i, yy, pp in zip(ids, y, p):
            preds.append({"model": model, "norm": norm, "seed": seed, "train": tr, "test": te, "rec": i, "true": int(yy), "pred": int(pp)})
        return m

    for norm in a.norms:
        print(f"\n================ normalization: {norm} ================")
        for tr in sessions:
            live = masks[tr]
            Xtr_raw = build_sequences(by[tr], live, norm); ytr = np.array([r.zone for r in by[tr]])
            mu, sd = fit_stats(Xtr_raw)
            Xtr = standardize(Xtr_raw, mu, sd)
            tests = {te: (by[te], standardize(build_sequences(by[te], live, norm), mu, sd), np.array([r.zone for r in by[te]]))
                     for te in sessions if te != tr}
            for name in [m for m in a.models if m != "C"]:
                Ftr = make_features(name, Xtr)
                for te, (rl, Xte, yte) in tests.items():
                    p = fit_predict_lr(Ftr, ytr, make_features(name, Xte))
                    m = record(name, norm, -1, tr, te, [r.id for r in rl], yte, p)
                    print(f"  {name:<3} {tr}->{te}: acc {100 * m['accuracy']:5.1f}%  bal {100 * m['balanced_accuracy']:5.1f}%  "
                          f"F1 {100 * m['macro_f1']:5.1f}%  predicted per zone {m['pred_counts']}")
            if "C" in a.models:
                for seed in a.seeds:
                    best, info = select_epochs(Xtr_raw, ytr, seed, a, curves, {"norm": norm, "train": tr, "seed": seed})
                    if best is None:
                        print(f"  C   train {tr} seed {seed}: skipped ({info})"); continue
                    epoch_info.append({"norm": norm, "train": tr, "seed": seed, **info})
                    flag = "  <- validation accuracy near chance: epoch choice unreliable" if info["val_acc_chance_level"] else ""
                    print(f"  C   train {tr} seed {seed}: inner CV ({info['folds']} folds, {info['n_inner_train']} train / "
                          f"{len(ytr) - info['n_inner_train']} val recordings) -> {best} epochs, val loss {info['val_loss_at_choice']:.3f}, "
                          f"val acc {100 * info['val_acc_at_choice']:.0f}%{flag}")
                    model, cv = train_cnn(Xtr, ytr, best, seed, a)
                    import torch
                    tr_acc = float(np.mean(predict_proba(model, Xtr, torch.device(a.device)).argmax(1) == ytr))
                    for te, (rl, Xte, yte) in tests.items():
                        p = predict_proba(model, Xte, torch.device(a.device)).argmax(1)
                        m = record("C", norm, seed, tr, te, [r.id for r in rl], yte, p)
                        print(f"      {tr}->{te}: acc {100 * m['accuracy']:5.1f}%  bal {100 * m['balanced_accuracy']:5.1f}%  "
                              f"F1 {100 * m['macro_f1']:5.1f}%  predicted per zone {m['pred_counts']}   (final train acc {100 * tr_acc:.0f}%)")

    # ---- save raw outputs ----
    with open(os.path.join(run_dir, "per_direction.csv"), "w", newline="") as f:
        cols = ["model", "norm", "seed", "train", "test", "n", "accuracy", "balanced_accuracy", "macro_f1"] + \
               [f"precision_{z}" for z in ZONES] + [f"recall_{z}" for z in ZONES] + [f"pred_{z}" for z in ZONES] + ["confusion"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            d = {k: r[k] for k in cols[:9]}
            d.update({f"precision_{z}": round(r["precision"][i], 4) for i, z in enumerate(ZONES)})
            d.update({f"recall_{z}": round(r["recall"][i], 4) for i, z in enumerate(ZONES)})
            d.update({f"pred_{z}": r["pred_counts"][i] for i, z in enumerate(ZONES)}); d["confusion"] = json.dumps(r["confusion"])
            w.writerow(d)
    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0])); w.writeheader(); w.writerows(preds)
    if curves:
        with open(os.path.join(run_dir, "learning_curves.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
        json.dump(epoch_info, open(os.path.join(run_dir, "epoch_selection.json"), "w"), indent=2)
        try:
            plot_curves(curves, epoch_info, run_dir)
        except Exception as e:
            print(f"(curve plot skipped: {e})")

    # ---- summary ----
    lines = []
    for norm in a.norms:
        lines.append(f"\n=== normalization: {norm}  (mean over seeds for C; mean over directions in the first columns) ===")
        lines.append(f"{'model':<6}{'mean acc':>10}{'mean bal':>10}{'mean F1':>9}   " + "".join(f"{tr}>{te:<5}" for tr, te in directions))
        for name in a.models:
            sel = [r for r in rows if r["model"] == name and r["norm"] == norm]
            if not sel:
                continue
            per = {}
            for tr, te in directions:
                rr = [r for r in sel if r["train"] == tr and r["test"] == te]
                if rr:
                    per[(tr, te)] = (np.mean([r["accuracy"] for r in rr]), np.mean([r["balanced_accuracy"] for r in rr]), np.mean([r["macro_f1"] for r in rr]))
            if not per:
                continue
            ma, mb, mf = (np.mean([v[i] for v in per.values()]) for i in range(3))
            sd_txt = ""
            if name == "C" and len(a.seeds) > 1:
                seed_means = [np.mean([r["accuracy"] for r in sel if r["seed"] == s]) for s in a.seeds if any(r["seed"] == s for r in sel)]
                sd_txt = f"   (seed-to-seed std of the mean: {100 * np.std(seed_means):.1f} pts)"
            lines.append(f"{name:<6}{100 * ma:>9.1f}%{100 * mb:>9.1f}%{100 * mf:>8.1f}%   " +
                         "".join(f"{100 * per[d][0]:>7.1f}%" if d in per else "      - " for d in directions) + sd_txt)
        lines.append("\nper-direction detail (seed-summed confusion, rows = true zone, cols = predicted):")
        for name in a.models:
            for tr, te in directions:
                rr = [r for r in rows if r["model"] == name and r["norm"] == norm and r["train"] == tr and r["test"] == te]
                if not rr:
                    continue
                cm = np.sum([np.array(r["confusion"]) for r in rr], 0)
                rec = np.mean([r["recall"] for r in rr], 0); prec = np.mean([r["precision"] for r in rr], 0)
                lines.append(f"  {name} {tr}->{te}: recall {np.round(rec, 2).tolist()} precision {np.round(prec, 2).tolist()} "
                             f"pred counts {np.sum([r['pred_counts'] for r in rr], 0).tolist()}  confusion {cm.tolist()}")
        # paired bootstrap vs A over recordings
        if "A" in a.models:
            lines.append("\npaired bootstrap over recordings, accuracy difference vs A (95% interval; each recording is tested in 2 directions):")
            def corr_by_rec(name):
                d = defaultdict(list)
                for p in preds:
                    if p["model"] == name and p["norm"] == norm:
                        d[p["rec"]].append(p["true"] == p["pred"])
                return {k: np.mean(v) for k, v in d.items()}
            ca = corr_by_rec("A"); rng = np.random.default_rng(0)
            for name in a.models:
                if name == "A":
                    continue
                cb = corr_by_rec(name); keys = sorted(set(ca) & set(cb))
                if len(keys) < 10:
                    continue
                diff = np.array([cb[k] - ca[k] for k in keys])
                boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
                lines.append(f"  {name} - A: {100 * diff.mean():+.1f} pts  [{100 * np.percentile(boots, 2.5):+.1f}, {100 * np.percentile(boots, 97.5):+.1f}]  over {len(keys)} recordings")
    n_eff = len(recs)
    lines.append(f"\nchance = 33.3%. With {n_eff} distinct test recordings (each tested in {len(sessions) - 1} directions) a mean accuracy "
                 f"must exceed about {100 * (1 / 3 + 1.645 * np.sqrt(2 / 9 / n_eff)):.0f}% to be distinguishable from chance (one-sided 95%).")
    lines.append("A difference between models is only meaningful if the bootstrap interval excludes 0.")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(run_dir, "summary.txt"), "w").write(txt + "\n")
    print(f"\nsaved everything to {run_dir}/  (per_direction.csv, predictions.csv, summary.txt, config.json"
          + (", learning_curves.csv, epoch_selection.json, curves PNG" if curves else "") + ")")


def plot_curves(curves, epoch_info, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norms = sorted({c["norm"] for c in curves}); trains = sorted({c["train"] for c in curves})
    fig, ax = plt.subplots(len(norms), len(trains), figsize=(4.2 * len(trains), 3.4 * len(norms)), squeeze=False)
    for i, nm in enumerate(norms):
        for j, tr in enumerate(trains):
            sel = [c for c in curves if c["norm"] == nm and c["train"] == tr]
            for key, lab in (("train_loss", "train"), ("val_loss", "validation")):
                ep = sorted({c["epoch"] for c in sel})
                ax[i][j].plot(ep, [np.mean([c[key] for c in sel if c["epoch"] == e]) for e in ep], label=lab)
            for e in epoch_info:
                if e["norm"] == nm and e["train"] == tr:
                    ax[i][j].axvline(e["chosen_epochs"], color="grey", alpha=0.3)
            ax[i][j].set_title(f"{nm}, trained on {tr}", fontsize=9); ax[i][j].set_xlabel("epoch"); ax[i][j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(run_dir, "learning_curves.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
