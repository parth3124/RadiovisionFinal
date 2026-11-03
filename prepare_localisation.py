import numpy as np
from pathlib import Path

DATA_DIR = Path("dataset_robot/walk_free")

X = []
Y = []
names = []

for folder in sorted(DATA_DIR.glob("grid01_*")):

    csi_file = folder / "csi_rx1.npz"
    track_file = folder / "track.npz"

    if not csi_file.exists() or not track_file.exists():
        continue

    csi_data = np.load(csi_file)
    track_data = np.load(track_file)

    # Keep only complete CSI packets
    valid_csi = csi_data["csi_len"] == 384
    csi = csi_data["csi"][valid_csi]
    csi_time = csi_data["host_t"][valid_csi]

    # Convert interleaved I/Q values into amplitudes
    csi = csi.astype(np.float32)
    amplitude = np.sqrt(
        csi[:, 0::2] ** 2 + csi[:, 1::2] ** 2
    )

    # Camera ground-truth data
    camera_time = track_data["t"]
    x = track_data["x"]
    y = track_data["y"]
    visible = track_data["visible"].astype(bool)

    valid_camera = (
        visible
        & np.isfinite(camera_time)
        & np.isfinite(x)
        & np.isfinite(y)
    )

    camera_time = camera_time[valid_camera]
    x = x[valid_camera]
    y = y[valid_camera]

    # Sort camera samples by timestamp
    order = np.argsort(camera_time)
    camera_time = camera_time[order]
    x = x[order]
    y = y[order]

    # Interpolate camera position at CSI timestamps
    inside = (
        (csi_time >= camera_time[0])
        & (csi_time <= camera_time[-1])
    )

    amplitude = amplitude[inside]
    csi_time = csi_time[inside]

    target_x = np.interp(csi_time, camera_time, x)
    target_y = np.interp(csi_time, camera_time, y)

    # One feature vector per recording
    feature = np.mean(amplitude, axis=0)
    target = [np.mean(target_x), np.mean(target_y)]

    X.append(feature)
    Y.append(target)
    names.append(folder.name)

    print(
        folder.name,
        "packets:", len(amplitude),
        "position:", np.round(target, 3)
    )

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

np.savez(
    "localisation_dataset.npz",
    X=X,
    Y=Y,
    names=np.array(names)
)

print("\nDataset created")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Number of recordings:", len(names))
