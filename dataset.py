
import numpy as np


class Recording:
    """
    Utilities for working with timestamped CSI recordings.
    """

    @staticmethod
    def _sync_times(host_t, ts_us):
        """
        Reconstruct a monotonic time axis using host arrival timestamps
        and radio timestamps.

        host_t:
            Host monotonic timestamps in seconds.

        ts_us:
            Radio timestamps in microseconds.

        Returns:
            Corrected timestamps in seconds.

        This implementation unwraps the radio clock and uses its
        relative timing while preserving the host recording's time origin.
        """

        host_t = np.asarray(host_t, dtype=np.float64)
        ts_us = np.asarray(ts_us, dtype=np.float64)

        if host_t.ndim != 1 or ts_us.ndim != 1:
            raise ValueError("Timestamp arrays must be one-dimensional")

        if len(host_t) != len(ts_us):
            raise ValueError("host_t and ts_us must have equal lengths")

        if len(host_t) == 0:
            return np.array([], dtype=np.float64)

        if len(host_t) == 1:
            return host_t.copy()

        if not np.all(np.isfinite(host_t)):
            raise ValueError("Invalid host timestamps")

        if not np.all(np.isfinite(ts_us)):
            raise ValueError("Invalid radio timestamps")

        # Unwrap a 32-bit microsecond radio timer.
        modulus = 2 ** 32
        unwrapped = ts_us.copy()

        offset = 0.0

        for i in range(1, len(unwrapped)):
            difference = unwrapped[i] - unwrapped[i - 1]

            if difference < -modulus / 2:
                offset += modulus

            unwrapped[i] += offset

        # Convert radio microseconds to seconds.
        radio_t = unwrapped * 1e-6

        # Use the radio clock's relative intervals.
        radio_relative = radio_t - radio_t[0]

        # Estimate the scale between the radio clock and host clock.
        radio_diffs = np.diff(radio_relative)
        host_diffs = np.diff(host_t)

        valid = (
            np.isfinite(radio_diffs)
            & np.isfinite(host_diffs)
            & (radio_diffs > 0)
            & (host_diffs > 0)
        )

        if np.any(valid):
            scale = np.median(
                host_diffs[valid] / radio_diffs[valid]
            )
        else:
            scale = 1.0

        # Reject implausible scale estimates.
        if not 0.9 <= scale <= 1.1:
            scale = 1.0

        corrected = host_t[0] + radio_relative * scale

        # Preserve monotonicity.
        corrected = np.maximum.accumulate(corrected)

        return corrected
