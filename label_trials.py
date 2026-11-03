import json
import numpy as np

DATA_FILE = "merged/s01.npz"
META_FILE = "merged/s01_meta.json"
OUT_FILE = "merged/s01_labeled.npz"

TRIM_SECONDS = 0.5

# Load merged CSI data and metadata
d = np.load(DATA_FILE)
with open(META_FILE) as f:
    meta = json.load(f)

tx_t = d["tx_t"]
present = d["present"]

# -1 means unlabelled; 0=A, 1=B, 2=C
label_map = {"A": 0, "B": 1, "C": 2}
label_names = np.array(["A", "B", "C"])

y = np.full(len(tx_t), -1, dtype=np.int8)
trial_id = np.full(len(tx_t), -1, dtype=np.int16)

# Only Laptop A contains the manually entered zone labels
events = [
    e for e in meta["events"]
    if e.get("dir") == "sessions/s01_laptopA"
]

trials = []
active = None
discarded = []

for e in events:
    if e["kind"] == "trial_start":
        if active is not None:
            discarded.append(active)
            print("Discarding unclosed trial:", active["label"])

        label = e.get("label", "").strip()
        if label == "As A":
            label = "A"

        active = {
            "label": label,
            "start": e["tx_t"]
        }

    elif e["kind"] == "trial_end":
        if active is None:
            print("WARNING: trial_end without a start")
            continue

        active["end"] = e["tx_t"]

        if active["label"] in label_map:
            trials.append(active.copy())
        else:
            print("Ignoring unknown label:", active["label"])

        active = None

if active is not None:
    discarded.append(active)
    print("Discarding final unclosed trial:", active["label"])

print("\nCompleted labelled trials:", len(trials))
print("Unclosed/discarded trials:", len(discarded))

# Assign labels to packets, trimming movement near trial boundaries
print("\nTRIAL-BY-TRIAL COUNTS")
print("Trial  Zone  Duration  Packets  All-4-RX  At-least-2-RX")

for i, trial in enumerate(trials):
    start = trial["start"] + TRIM_SECONDS
    end = trial["end"] - TRIM_SECONDS

    if end <= start:
        print("Skipping too-short trial:", i + 1)
        continue

    mask = (tx_t >= start) & (tx_t <= end)

    y[mask] = label_map[trial["label"]]
    trial_id[mask] = i

    n = int(mask.sum())
    all4 = int(present[mask].all(axis=1).sum())
    at_least2 = int((present[mask].sum(axis=1) >= 2).sum())

    print(
        f"{i+1:5d}  {trial['label']:>4s}  "
        f"{end-start:7.2f}s  {n:7d}  "
        f"{all4:8d}  {at_least2:14d}"
    )

# Overall coverage per zone
print("\nZONE SUMMARY")
for name, value in label_map.items():
    mask = y == value
    n = int(mask.sum())

    if n == 0:
        print(f"Zone {name}: NO PACKETS")
        continue

    all4 = int(present[mask].all(axis=1).sum())
    at_least2 = int((present[mask].sum(axis=1) >= 2).sum())

    print(
        f"Zone {name}: {n} packets | "
        f"all 4 RX: {all4} ({100*all4/n:.1f}%) | "
        f"at least 2 RX: {at_least2} ({100*at_least2/n:.1f}%)"
    )

# Save a new dataset; preserve all original arrays
np.savez_compressed(
    OUT_FILE,
    **{key: d[key] for key in d.files},
    y=y,
    trial_id=trial_id,
    label_names=label_names
)

print("\nSaved:", OUT_FILE)
print("Unlabelled packets:", int((y == -1).sum()))
