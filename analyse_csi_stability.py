import numpy as np
from pathlib import Path

for folder in sorted(Path("dataset_robot/walk_free").glob("grid01_*")):

    d = np.load(folder / "csi_rx1.npz")

    valid = d["csi_len"] == 384
    csi = d["csi"][valid].astype(np.float32)

    amp = np.sqrt(
        csi[:, 0::2] ** 2 + csi[:, 1::2] ** 2
    )

    mean = np.mean(amp, axis=0)
    std = np.std(amp, axis=0)

    # Average coefficient of variation across subcarriers
    cv = np.mean(std / (mean + 1e-6))

    print(folder.name, "CV:", round(float(cv), 3))
