import numpy as np
from pathlib import Path
from scipy.signal import welch

DATA_DIR = Path("dataset_robot/walk_free")
WINDOW_SEC = 1.0
MIN_PACKETS = 40
MAX_GAP = 0.15


def safe_stats(a):
    """Statistics along the packet/time axis."""
    return np.concatenate([
        np.mean(a, axis=0),
        np.std(a, axis=0),
        np.median(a, axis=0),
        np.percentile(a, 25, axis=0),
        np.percentile(a, 75, axis=0)
    ])


def circular_stats(phase):
    """Circular phase statistics."""
    return np.concatenate([
        np.mean(np.sin(phase), axis=0),
        np.mean(np.cos(phase), axis=0),
        np.std(np.sin(phase), axis=0),
        np.std(np.cos(phase), axis=0)
    ])


def make_features(csi, rssi, timestamps):
    # Firmware format: imaginary, real, imaginary, real...
    imag = csi[:, 0::2].astype(np.float32)
    real = csi[:, 1::2].astype(np.float32)

    H = real + 1j * imag

    amplitude = np.abs(H)
    phase = np.angle(H)

    # Relative phase between adjacent subcarriers
    relative_phase = np.angle(H[:, 1:] * np.conj(H[:, :-1]))

    # Amplitude features
    amp_features = safe_stats(amplitude)

    # Complex CSI features
    complex_features = np.concatenate([
        safe_stats(real),
        safe_stats(imag)
    ])

    # Phase features
    phase_features = np.concatenate([
        circular_stats(phase),
        circular_stats(relative_phase)
    ])

    # RSSI features
    rssi_features = np.array([
        np.mean(rssi),
        np.std(rssi),
        np.min(rssi),
        np.max(rssi),
        np.median(rssi)
    ], dtype=np.float32)

    # Temporal features
    diff_amp = np.diff(amplitude, axis=0)
    temporal_features = np.concatenate([
        safe_stats(diff_amp),
        np.mean(np.abs(diff_amp), axis=0),
        np.percentile(np.abs(diff_amp), 90, axis=0)
    ])

    # Temporal Doppler spectrum
    dt = np.median(np.diff(timestamps))
    fs = 1.0 / dt

    doppler = []

    for k in range(amplitude.shape[1]):
        signal = amplitude[:, k] - np.mean(amplitude[:, k])

        if len(signal) < 16:
            continue

        frequencies, power = welch(
            signal,
            fs=fs,
            nperseg=min(128, len(signal))
        )

        # Exclude DC component
        power = power[1:]
        frequencies = frequencies[1:]

        if len(power) == 0:
            continue

        total_power = np.sum(power) + 1e-9
        dominant = frequencies[np.argmax(power)]

        doppler.extend([
            dominant,
            np.sum(power * frequencies) / total_power,
            np.sum(power),
            np.max(power) / total_power
        ])

    doppler_features = np.asarray(doppler, dtype=np.float32)

    # Subcarrier relationships
    corr = np.corrcoef(amplitude, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Keep upper triangle excluding diagonal
    upper = corr[np.triu_indices_from(corr, k=1)]

    correlation_features = np.concatenate([
        upper,
        np.mean(np.abs(np.diff(amplitude, axis=1)), axis=0)
    ])

    return {
        "amplitude": amp_features,
        "complex": complex_features,
        "phase": phase_features,
        "rssi": rssi_features,
        "temporal": temporal_features,
        "doppler": doppler_features,
        "correlation": correlation_features
    }


all_features = {
    "amplitude": [],
    "complex": [],
    "phase": [],
    "rssi": [],
    "temporal": [],
    "doppler": [],
    "correlation": []
}

targets = []
groups = []
window_names = []

recordings = sorted(DATA_DIR.glob("grid01_*"))

for folder in recordings:
    csi_file = folder / "csi_rx1.npz"
    track_file = folder / "track.npz"

    if not csi_file.exists() or not track_file.exists():
        continue

    csi_data = np.load(csi_file)
    track_data = np.load(track_file)

    valid = csi_data["csi_len"] == 384

    csi = csi_data["csi"][valid]
    timestamps = csi_data["host_t"][valid]
    rssi = csi_data["rssi"][valid].astype(np.float32)

    camera_t = track_data["t"]
    camera_x = track_data["x"]
    camera_y = track_data["y"]

    camera_valid = (
        track_data["visible"].astype(bool)
        & np.isfinite(camera_t)
        & np.isfinite(camera_x)
        & np.isfinite(camera_y)
    )

    camera_t = camera_t[camera_valid]
    camera_x = camera_x[camera_valid]
    camera_y = camera_y[camera_valid]

    order = np.argsort(camera_t)
    camera_t = camera_t[order]
    camera_x = camera_x[order]
    camera_y = camera_y[order]

    # Do not interpolate across large camera gaps.
    window_start = max(timestamps[0], camera_t[0])
    window_end = min(timestamps[-1], camera_t[-1])

    start = window_start

    while start + WINDOW_SEC <= window_end:
        end = start + WINDOW_SEC

        csi_mask = (timestamps >= start) & (timestamps < end)
        cam_mask = (camera_t >= start) & (camera_t <= end)

        if np.sum(csi_mask) < MIN_PACKETS or np.sum(cam_mask) < 10:
            start = end
            continue

        t = timestamps[csi_mask]
        gap = np.max(np.diff(t)) if len(t) > 1 else 999

        if gap > MAX_GAP:
            start = end
            continue

        # Use camera samples inside this window for ground truth.
        x = camera_x[cam_mask]
        y = camera_y[cam_mask]

        features = make_features(
            csi[csi_mask],
            rssi[csi_mask],
            t
        )

        for key in all_features:
            all_features[key].append(features[key])

        targets.append([np.mean(x), np.mean(y)])
        groups.append(folder.name)
        window_names.append(f"{folder.name}_{start:.3f}")

        start = end

    print(folder.name, "windows collected:",
          sum(g == folder.name for g in groups))

if not targets:
    raise RuntimeError("No valid windows were extracted.")

# All feature families must have consistent dimensions.
for key, values in all_features.items():
    all_features[key] = np.stack(values).astype(np.float32)

np.savez_compressed(
    "localisation_features.npz",
    **all_features,
    Y=np.asarray(targets, dtype=np.float32),
    groups=np.asarray(groups),
    names=np.asarray(window_names)
)

print("\nFEATURE EXTRACTION COMPLETE")
print("Total windows:", len(targets))
print("Unique recordings:", len(set(groups)))
print("Target shape:", np.asarray(targets).shape)

for key, values in all_features.items():
    print(key, values.shape)
