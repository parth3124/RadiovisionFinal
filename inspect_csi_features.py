import numpy as np
from collections import defaultdict
from zone_eval import load

DATA = "dataset_room"
recs, names = load(DATA)

# Select live subcarriers using r01 only, to avoid using r02
# when deciding which features to inspect.
train = [r for r in recs if r["meta"]["session"] == "r01"]
live = {}

for name in names:
    rm = np.mean([r["links"][name].mean(0) for r in train], axis=0)
    live[name] = rm > 0.05 * np.median(rm)
    print(f"Link {name}: {live[name].sum()} live subcarriers")

# Extract per-recording features using the same log-amplitude and
# per-packet gain normalization as zone_eval.py.
features = []
for r in recs:
    parts = []
    for name in names:
        A = r["links"][name]
        Lg = np.log1p(A)
        Lg = Lg - Lg[:, live[name]].mean(axis=1, keepdims=True)
        parts.append(Lg[:, live[name]].mean(axis=0))
    features.append(np.concatenate(parts))

features = np.asarray(features)

def summarize(title, indices):
    X = features[indices]
    if len(X) == 0:
        return None
    mean = X.mean(axis=0)
    return mean

def dist(a, b):
    # Euclidean distance between mean feature vectors,
    # divided by the typical within-group feature variation.
    pooled = np.vstack([a, b])
    scale = np.sqrt(np.mean(np.var(pooled, axis=0))) + 1e-12
    return float(np.sqrt(np.mean((a.mean(0) - b.mean(0))**2)) / scale)

def group(session, condition, zone=None):
    idx = []
    for i, r in enumerate(recs):
        m = r["meta"]
        if m["session"] != session or m["condition"] != condition:
            continue
        if zone is not None and m.get("zone") != zone:
            continue
        idx.append(i)
    return np.asarray(idx, dtype=int)

print("\nRECORDING COUNTS")
for session in ("r01", "r02"):
    for condition in ("empty", "obj", "person"):
        zones = (None,) if condition == "empty" else ("Z1", "Z2", "Z3")
        for zone in zones:
            idx = group(session, condition, zone)
            label = condition if zone is None else f"{condition}_{zone}"
            print(f"{session:3s} {label:12s}: {len(idx)}")

print("\nWITHIN-SESSION ZONE SEPARATION")
print("RMS distance between class means / pooled feature scale; larger means more separated.")
for session in ("r01", "r02"):
    for condition in ("obj", "person"):
        means = {}
        for zone in ("Z1", "Z2", "Z3"):
            idx = group(session, condition, zone)
            means[zone] = features[idx].mean(axis=0)
        values = [
            np.sqrt(np.mean((means[a] - means[b])**2))
            for a, b in (("Z1", "Z2"), ("Z1", "Z3"), ("Z2", "Z3"))
        ]
        scale_idx = np.concatenate([group(session, condition, z) for z in ("Z1", "Z2", "Z3")])
        scale = np.sqrt(np.mean(np.var(features[scale_idx], axis=0))) + 1e-12
        print(f"{session} {condition:6s}: "
              f"Z1-Z2={values[0]/scale:.3f}, "
              f"Z1-Z3={values[1]/scale:.3f}, "
              f"Z2-Z3={values[2]/scale:.3f}")

print("\nSAME-ZONE SESSION SHIFT")
print("RMS difference between session means / pooled within-session feature scale.")
for condition in ("obj", "person"):
    for zone in ("Z1", "Z2", "Z3"):
        i1 = group("r01", condition, zone)
        i2 = group("r02", condition, zone)
        a, b = features[i1], features[i2]
        pooled = np.vstack([a, b])
        scale = np.sqrt(np.mean(np.var(pooled, axis=0))) + 1e-12
        shift = np.sqrt(np.mean((a.mean(0) - b.mean(0))**2)) / scale
        print(f"{condition:6s} {zone}: shift={shift:.3f}")

print("\nOBJECT/PERSON CHANGE FROM EMPTY ROOM")
print("RMS difference between condition mean and empty mean / pooled feature scale.")
for session in ("r01", "r02"):
    empty = features[group(session, "empty")]
    for condition in ("obj", "person"):
        idx = np.concatenate([group(session, condition, z) for z in ("Z1", "Z2", "Z3")])
        X = features[idx]
        pooled = np.vstack([empty, X])
        scale = np.sqrt(np.mean(np.var(pooled, axis=0))) + 1e-12
        change = np.sqrt(np.mean((X.mean(0) - empty.mean(0))**2)) / scale
        print(f"{session} {condition:6s}: change={change:.3f}")

print("\nNote: descriptive distances only; they are not statistical significance tests.")
