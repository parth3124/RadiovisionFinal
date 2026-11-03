
"""
zone_eval.py - STEP 1 analysis: does CSI from the room carry information
about WHERE an object / person is?

    python zone_eval.py --data dataset_room
    python zone_eval.py --data dataset_room --norm packet
    python zone_eval.py --data dataset_room --norm none

Works for one link and multiple links. All csi_<name>.npz files of a
recording are concatenated.

FROZEN PIPELINE:
  features  per link: mean over time (first/last 0.5 s trimmed) of
            per-packet-normalized log(1+amplitude), per live subcarrier
  --norm ref     subtract the mean feature vector of the empty recordings
                 of the SAME session; for an empty recording itself, use
                 the other empty recordings. Needs >= 3 empties per session.
  --norm packet  per-packet gain removal only
  --norm none    raw log amplitude
  model     StandardScaler + LogisticRegression(C=0.1, balanced)
  evidence  permutation test: labels shuffled within each session
  sanity    time-only baseline: nearest-in-time recording of same session

Tasks: obj_zone, person_zone, obj_presence, person_presence.
"""

import argparse
import glob
import json
import os
import datetime as dt
from collections import Counter

import numpy as np
import common as C
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def load(root, trim=0.5):
    recs = []

    for mp in sorted(glob.glob(os.path.join(root, "*", "*", "meta.json"))):
        d = os.path.dirname(mp)
        with open(mp) as fh:
            meta = json.load(fh)

        links = {}

        for f in sorted(glob.glob(os.path.join(d, "csi_*.npz"))):
            with np.load(f) as z:
                lens = z["csi_len"]
                L = Counter(lens.tolist()).most_common(1)[0][0]
                keep = lens == L

                if keep.sum() < 100:
                    continue

                A = C.csi_amplitude(z["csi"][keep, :L])
                t = z["host_t"][keep]

            # r01/r02 use csi_link.npz; r03 uses csi_rx1.npz.
            # Standardize the filename so these are treated as one link.
            name = os.path.basename(f)[4:-4]
            if name == "rx1":
                name = "link"

            trimmed = A[(t > t[0] + trim) & (t < t[-1] - trim)]
            links[name] = trimmed

        if links:
            recs.append({
                "meta": meta,
                "links": links,
                "path": d,
            })

    if not recs:
        raise SystemExit(f"No recordings found under {root}/")

    # Retain recordings that contain the same set of logical links.
    common_set = Counter(
        tuple(sorted(r["links"])) for r in recs
    ).most_common(1)[0][0]

    recs = [
        r for r in recs
        if tuple(sorted(r["links"])) == common_set
    ]

    return recs, list(common_set)


def featurize(recs, names, norm):
    live = {}

    for n in names:
        rm = np.mean(
            [r["links"][n].mean(0) for r in recs],
            axis=0,
        )
        live[n] = rm > 0.05 * np.median(rm)

        if not live[n].any():
            raise SystemExit(f"No live subcarriers found for link {n}")

    for r in recs:
        parts = []

        for n in names:
            Lg = np.log1p(r["links"][n])

            if norm != "none":
                Lg = Lg - Lg[:, live[n]].mean(
                    axis=1,
                    keepdims=True,
                )

            parts.append(Lg[:, live[n]].mean(axis=0))

        r["feat"] = np.concatenate(parts)

    if norm == "ref":
        for u in sorted({r["meta"]["session"] for r in recs}):
            idx = [
                i for i, r in enumerate(recs)
                if r["meta"]["session"] == u
            ]
            emp = [
                i for i in idx
                if recs[i]["meta"]["condition"] == "empty"
            ]

            if len(emp) < 3:
                raise SystemExit(
                    f"--norm ref needs >= 3 empty recordings "
                    f"in session {u} (found {len(emp)})"
                )

            F = np.array([recs[i]["feat"] for i in emp])
            tot = F.sum(axis=0)
            new = {}

            for i in idx:
                if i in emp:
                    new[i] = recs[i]["feat"] - (
                        tot - recs[i]["feat"]
                    ) / (len(emp) - 1)
                else:
                    new[i] = recs[i]["feat"] - tot / len(emp)

            for i, v in new.items():
                recs[i]["feat"] = v


def model(Cc):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=Cc,
            max_iter=1000,
            class_weight="balanced",
        ),
    )


def run_cv(X, y, s, Cc, seed=0):
    pred = np.zeros_like(y)
    sess = np.unique(s)

    if len(sess) >= 2:
        for u in sess:
            te = s == u

            if len(set(y[~te])) < len(set(y)):
                pred[te] = -1
                continue

            pred[te] = model(Cc).fit(
                X[~te], y[~te]
            ).predict(X[te])

        mode = "leave-one-session-out"

    else:
        counts = np.bincount(y)
        nonzero = counts[counts > 0]
        k = int(min(5, nonzero.min()))

        if k < 2:
            return None, "too few recordings"

        for tr, te in StratifiedKFold(
            k,
            shuffle=True,
            random_state=seed,
        ).split(X, y):
            pred[te] = model(Cc).fit(
                X[tr], y[tr]
            ).predict(X[te])

        mode = f"{k}-fold CV within one session"

    ok = pred >= 0
    return (pred, ok), mode


def time_only(y, s, t):
    pred = np.zeros_like(y)

    for i in range(len(y)):
        idx = [
            j for j in range(len(y))
            if j != i and s[j] == s[i]
        ]

        pred[i] = (
            y[idx[int(np.argmin(np.abs(t[idx] - t[i])))]]
            if idx else y[i]
        )

    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_room")
    ap.add_argument(
        "--norm",
        choices=["ref", "packet", "none"],
        default="ref",
    )
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--C", type=float, default=0.1)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    recs, names = load(a.data)
    featurize(recs, names, a.norm)

    sessions = sorted({
        r["meta"]["session"] for r in recs
    })

    print(
        f"{len(recs)} recordings, links {names}, "
        f"sessions {sessions}, norm={a.norm}, "
        f"{a.perms} permutations per task"
    )
    print(
        f"{'task':<16}{'n':>4}{'classes':>8}"
        f"{'chance':>8}{'bal.acc':>9}{'null95':>8}"
        f"{'p':>8}{'time-only':>11}   scheme"
    )

    for cond in ("obj", "person"):
        for kind in ("zone", "presence"):

            if kind == "zone":
                sel = [
                    i for i, r in enumerate(recs)
                    if r["meta"]["condition"] == cond
                    and r["meta"].get("zone")
                ]
                labs = sorted({
                    recs[i]["meta"]["zone"] for i in sel
                })
                y = np.array([
                    labs.index(recs[i]["meta"]["zone"])
                    for i in sel
                ])

            else:
                sel = [
                    i for i, r in enumerate(recs)
                    if r["meta"]["condition"] in (cond, "empty")
                ]
                labs = ["empty", cond]
                y = np.array([
                    int(recs[i]["meta"]["condition"] == cond)
                    for i in sel
                ])

            if len(sel) < 6 or len(set(y)) < 2:
                continue

            X = np.array([recs[i]["feat"] for i in sel])
            s = np.array([
                recs[i]["meta"]["session"] for i in sel
            ])
            t = np.array([
                dt.datetime.fromisoformat(
                    recs[i]["meta"]["created"]
                ).timestamp()
                for i in sel
            ])

            res, mode = run_cv(X, y, s, a.C)

            if res is None:
                print(f"{cond + '_' + kind:<16} skipped: {mode}")
                continue

            pred, ok = res
            obs = balanced_accuracy_score(y[ok], pred[ok])
            null = []

            for _ in range(a.perms):
                yp = y.copy()

                for u in np.unique(s):
                    idx = np.where(s == u)[0]
                    yp[idx] = y[rng.permutation(idx)]

                r2, _ = run_cv(X, yp, s, a.C)

                if r2 is None:
                    continue

                p_pred, p_ok = r2

                if p_ok.any():
                    null.append(
                        balanced_accuracy_score(
                            yp[p_ok], p_pred[p_ok]
                        )
                    )

            null = np.array(null)

            if len(null):
                p = (1 + (null >= obs).sum()) / (1 + len(null))
                null95 = 100 * np.percentile(null, 95)
            else:
                p = float("nan")
                null95 = float("nan")

            tb = balanced_accuracy_score(y, time_only(y, s, t))

            print(
                f"{cond + '_' + kind:<16}{len(y):>4}"
                f"{len(labs):>8}{100 / len(labs):>7.0f}%"
                f"{100 * obs:>8.1f}%{null95:>7.1f}%"
                f"{p:>8.3f}{100 * tb:>10.1f}%   {mode}"
            )

            if kind == "zone":
                cm = confusion_matrix(
                    y[ok], pred[ok],
                    labels=range(len(labs)),
                )
                err = np.mean(np.abs(pred[ok] - y[ok]))

                print(
                    f"    confusion (rows true, cols pred) "
                    f"order {labs}: mean zone error "
                    f"{err:.2f} zones\n    {cm.tolist()}"
                )

    print(
        "\np small AND bal.acc well above chance = "
        "zone/presence information is present. "
        "time-only should be near chance."
    )
    print(
        "person_zone is the positive control; interpret "
        "obj_zone only in light of it."
    )


if __name__ == "__main__":
    main()
