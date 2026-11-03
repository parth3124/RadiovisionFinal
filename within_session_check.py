#!/usr/bin/env python3
"""
within_session_check.py - is there ANY person-zone information inside a single session? (ceiling check, no torch)

    python within_session_check.py --data dataset_room
    python within_session_check.py --data dataset_room --norm none

For each session separately: stratified K-fold over its 18 person recordings (6 per zone); the live-subcarrier mask,
standardization and classifier are fitted on the training folds only. A permutation test (zone labels shuffled
within the session, whole procedure refit) gives the probability of the observed accuracy under 'no relationship'.
Valid because the recording order inside each session was shuffled (zone is not confounded with time).
Reads together with zone_temporal_compare.py:
  within-session high + cross-session at chance  -> zone information exists but is session-specific (does not transfer)
  within-session at chance as well               -> this link/geometry does not encode a person's zone at all
Uses the same loader, preprocessing and feature definitions as zone_temporal_compare.py (imported, not copied).
"""
import argparse
import numpy as np
import zone_temporal_compare as Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_room")
    ap.add_argument("--norm", choices=["packet", "none"], default="packet")
    ap.add_argument("--models", nargs="+", default=["A", "B1", "B2", "B3"], choices=["A", "B1", "B2", "B3"])
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(a.seed)
    recs, skipped, other, _ = Z.load_all(a.data)
    print(f"{len(recs)} person recordings ({len(skipped)} skipped), norm={a.norm}, {a.perms} permutations, chance 33.3%")
    sessions = sorted({r.session for r in recs})
    print(f"\n{'session':<8}{'model':<6}{'accuracy':>9}{'null mean':>11}{'null 95th':>11}{'p':>8}   recall per zone (Z1 Z2 Z3)")
    for s in sessions:
        rs = [r for r in recs if r.session == s]; y = np.array([r.zone for r in rs])
        k = int(min(a.folds, np.bincount(y).min()))
        if k < 2:
            print(f"{s}: too few recordings per zone"); continue
        folds = []
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=a.seed).split(np.zeros(len(y)), y):
            trr = [rs[i] for i in tr]; live = Z.live_mask(trr)
            Xtr_raw = Z.build_sequences(trr, live, a.norm); mu, sd = Z.fit_stats(Xtr_raw)
            Xtr = Z.standardize(Xtr_raw, mu, sd)
            Xte = Z.standardize(Z.build_sequences([rs[i] for i in te], live, a.norm), mu, sd)
            folds.append((tr, te, {m: (Z.make_features(m, Xtr), Z.make_features(m, Xte)) for m in a.models}))

        def cv_acc(m, labels):
            correct = 0
            pred_all = np.zeros(len(labels), int)
            for tr, te, F in folds:
                pred = Z.fit_predict_lr(F[m][0], labels[tr], F[m][1])
                pred_all[te] = pred; correct += int((pred == labels[te]).sum())
            return correct / len(labels), pred_all

        for m in a.models:
            obs, pred = cv_acc(m, y)
            null = np.array([cv_acc(m, rng.permutation(y))[0] for _ in range(a.perms)])
            p = (1 + (null >= obs).sum()) / (1 + a.perms)
            rec = [float(np.mean(pred[y == z] == z)) for z in range(3)]
            print(f"{s:<8}{m:<6}{100 * obs:>8.1f}%{100 * null.mean():>10.1f}%{100 * np.percentile(null, 95):>10.1f}%{p:>8.3f}   "
                  + " ".join(f"{100 * r:3.0f}%" for r in rec))
    print("\nSmall p (and accuracy well above the null 95th percentile) = zone information exists inside that session.")
    print("Each session has only 18 recordings, so a single accuracy is uncertain by roughly +-11 points; rely on p and on agreement across sessions.")


if __name__ == "__main__":
    main()
