
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from zone_eval import load


DATA = "dataset_room"
OUT = Path("csi_profile_plots")


def build_live_masks(recs, names):
    """Select live subcarriers using r01 only, matching inspect_csi_features.py."""
    train = [r for r in recs if r["meta"]["session"] == "r01"]

    if not train:
        raise RuntimeError("No r01 recordings found.")

    live = {}

    for name in names:
        rm = np.mean(
            [r["links"][name].mean(axis=0) for r in train],
            axis=0,
        )
        mask = rm > 0.05 * np.median(rm)
        live[name] = mask

        print(
            f"Link {name}: {mask.sum()} live subcarriers "
            f"out of {len(mask)}"
        )

    return live


def make_profile(amplitude, mask):
    """Match the preprocessing in inspect_csi_features.py."""
    amplitude = np.asarray(amplitude, dtype=float)

    if amplitude.ndim != 2:
        raise ValueError(
            f"Expected [packets, subcarriers], got {amplitude.shape}"
        )

    if amplitude.shape[1] != len(mask):
        raise ValueError(
            f"Mask width {len(mask)} does not match "
            f"CSI width {amplitude.shape[1]}"
        )

    if not np.any(mask):
        raise ValueError("No live subcarriers selected.")

    log_amp = np.log1p(np.maximum(amplitude, 0))

    # Same packet-wise normalization as inspect_csi_features.py.
    packet_mean = log_amp[:, mask].mean(axis=1, keepdims=True)
    log_amp = log_amp - packet_mean

    # Return one value per live subcarrier.
    return log_amp[:, mask].mean(axis=0)


def get_grouped_profiles(recs, names, live):
    """Compute one profile per recording, grouped by session and class."""
    grouped = {}

    for rec in recs:
        meta = rec["meta"]
        session = meta["session"]
        condition = meta["condition"]
        zone = meta.get("zone")

        label = condition if condition == "empty" else f"{condition}_{zone}"

        for name in names:
            profile = make_profile(rec["links"][name], live[name])
            key = (name, session, label)
            grouped.setdefault(key, []).append(profile)

    return grouped


def mean_and_std(profiles):
    matrix = np.stack(profiles)
    mean = matrix.mean(axis=0)

    if len(profiles) > 1:
        std = matrix.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)

    return mean, std


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name))


def plot_zone_profiles(grouped, names, live):
    """Plot object/person profiles and the empty-room profile."""
    for name in names:
        # Original subcarrier indices, with dead bins excluded.
        x = np.flatnonzero(live[name])

        for condition in ("obj", "person"):
            fig, axes = plt.subplots(
                1, 2, figsize=(15, 5), sharey=True
            )

            for ax, session in zip(axes, ("r01", "r02")):
                for zone in ("Z1", "Z2", "Z3"):
                    key = (name, session, f"{condition}_{zone}")
                    profiles = grouped.get(key, [])

                    if not profiles:
                        continue

                    mean, std = mean_and_std(profiles)
                    ax.plot(
                        x, mean,
                        label=f"{zone} (n={len(profiles)})"
                    )
                    ax.fill_between(
                        x, mean - std, mean + std, alpha=0.15
                    )

                empty = grouped.get((name, session, "empty"), [])

                if empty:
                    mean, _ = mean_and_std(empty)
                    ax.plot(
                        x, mean,
                        "k--", label=f"Empty (n={len(empty)})"
                    )

                ax.set_title(session)
                ax.set_xlabel("Original subcarrier index")
                ax.grid(True, alpha=0.25)
                ax.legend()

            axes[0].set_ylabel(
                "Mean log amplitude after live-bin centering"
            )
            fig.suptitle(
                f"{condition.capitalize()} CSI profiles — link {name}"
            )
            fig.tight_layout()

            path = OUT / (
                f"{safe_name(name)}_{condition}_zone_profiles.png"
            )
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")


def plot_empty_differences(grouped, names, live):
    """Plot each condition's profile minus the empty-room profile."""
    for name in names:
        x = np.flatnonzero(live[name])

        for session in ("r01", "r02"):
            empty = grouped.get((name, session, "empty"), [])

            if not empty:
                print(f"No empty-room recordings for {session}, link {name}")
                continue

            empty_mean, _ = mean_and_std(empty)

            fig, ax = plt.subplots(figsize=(12, 5))

            labels = (
                "obj_Z1", "obj_Z2", "obj_Z3",
                "person_Z1", "person_Z2", "person_Z3",
            )

            plotted = False

            for label in labels:
                profiles = grouped.get((name, session, label), [])

                if not profiles:
                    continue

                condition_mean, _ = mean_and_std(profiles)

                ax.plot(
                    x,
                    condition_mean - empty_mean,
                    label=label,
                )
                plotted = True

            if plotted:
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.set_title(
                    f"{session}: condition minus empty-room profile "
                    f"— link {name}"
                )
                ax.set_xlabel("Original subcarrier index")
                ax.set_ylabel(
                    "Difference in centered log amplitude"
                )
                ax.grid(True, alpha=0.25)
                ax.legend(ncol=2)
                fig.tight_layout()

                path = OUT / (
                    f"{safe_name(name)}_{session}_empty_differences.png"
                )
                fig.savefig(path, dpi=160, bbox_inches="tight")
                print(f"Saved: {path}")

            plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    recs, names = load(DATA)

    print(f"Loaded {len(recs)} recordings.")
    print(f"Links: {names}")

    live = build_live_masks(recs, names)
    grouped = get_grouped_profiles(recs, names, live)

    plot_zone_profiles(grouped, names, live)
    plot_empty_differences(grouped, names, live)

    print("\nDone. These plots use the same live-bin mask and")
    print("packet-wise normalization as inspect_csi_features.py.")


if __name__ == "__main__":
    main()

