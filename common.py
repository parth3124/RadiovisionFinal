
import numpy as np

# Maximum permitted gap when reconstructing a regular CSI time grid.
MAX_GAP_SEC = 0.20


def csi_amplitude(csi):
    """
    Convert interleaved int8 CSI I/Q values into amplitudes.

    Expected input:
        (..., 2 * num_subcarriers)

    Output:
        (..., num_subcarriers)

    If the input is already complex, calculate its magnitude.
    """

    csi = np.asarray(csi)

    if np.iscomplexobj(csi):
        return np.abs(csi).astype(np.float32)

    if csi.ndim == 0:
        raise ValueError("CSI input must have at least one dimension")

    if csi.shape[-1] % 2 != 0:
        raise ValueError(
            f"Expected an even final dimension for I/Q pairs, got {csi.shape}"
        )

    # Convert interleaved I, Q, I, Q values.
    i_values = csi[..., 0::2].astype(np.float32)
    q_values = csi[..., 1::2].astype(np.float32)

    amplitude = np.sqrt(i_values ** 2 + q_values ** 2)

    return amplitude.astype(np.float32)


def validate_csi(csi):
    """Check that CSI contains finite numerical values."""

    csi = np.asarray(csi)

    if csi.size == 0:
        raise ValueError("CSI array is empty")

    if not np.issubdtype(csi.dtype, np.number):
        raise TypeError("CSI must contain numerical values")

    if not np.all(np.isfinite(csi)):
        raise ValueError("CSI contains NaN or infinite values")

    return True


def normalize_amplitude(amplitude, eps=1e-8):
    """
    Normalize each packet by its mean amplitude.

    Useful for reducing overall signal-strength variation.
    """

    amplitude = np.asarray(amplitude, dtype=np.float32)

    mean = np.mean(amplitude, axis=-1, keepdims=True)

    return amplitude / (mean + eps)


def check_packet_times(host_t):
    """Check whether host timestamps are valid and increasing."""

    host_t = np.asarray(host_t, dtype=np.float64)

    if host_t.ndim != 1:
        raise ValueError("host_t must be one-dimensional")

    if len(host_t) < 2:
        raise ValueError("At least two timestamps are required")

    if not np.all(np.isfinite(host_t)):
        raise ValueError("Invalid timestamp values")

    if np.any(np.diff(host_t) <= 0):
        raise ValueError("Timestamps must be strictly increasing")

    return True
