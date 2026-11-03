"""
csi_complex.py - complex CSI reconstruction and phase sanitisation (library; read-only on the data).

Record format (from the audit): csi_len 384 = three 128-value blocks of 64 positions, each position = 2 int8 values.
Convention (ESP-IDF docs): first value = imaginary, second = real  ->  c = real + 1j*imag.
(Swapping the two mirrors the phase; every diagnostic here is insensitive to that sign. Not verifiable from the data.)
Valid (non-null) positions per block of the 384-type, measured in all 132 recordings by audit_recordings.py:
    block 0: 6..31, 33..58      block 1: 2..58      block 2: 6..62
Only 384-type packets are used (the 128/256-types have different layouts and levels).

Sanitisation methods (applied per packet and per block):
    raw        none
    cpo        remove the common phase offset: multiply by exp(-j*angle(sum_k c_k))
    cpo_slope  additionally remove a linear phase slope across subcarriers (sampling-time offset), slope estimated as the
               circular mean of adjacent-subcarrier phase steps, then remove the common phase
    ref        phase relative to one reference subcarrier (centre of the valid range): c_k * conj(c_ref)
Warning: cpo/cpo_slope also remove the part of a real channel change that looks like a common phase or slope change.
"""
import numpy as np

BLOCK = 128
VALID_384 = {0: np.r_[6:32, 33:59], 1: np.r_[2:59], 2: np.r_[6:63]}
METHODS = ("raw", "cpo", "cpo_slope", "ref")


def to_complex(csi, length=384):
    """csi int8 [n, >=length] -> list of complex arrays [n, 64], one per 128-value block"""
    x = np.asarray(csi[:, :length], dtype=np.float32)
    out = []
    for b in range(length // BLOCK):
        blk = x[:, b * BLOCK:(b + 1) * BLOCK]
        out.append(blk[:, 1::2] + 1j * blk[:, 0::2])
    return out


def select_type(a, length=384):
    """packets of one csi_len, in radio-time order -> (complex blocks, ts_us, rssi, host_t)"""
    m = a["csi_len"].astype(int) == length
    order = np.argsort(a["ts_us"].astype(np.int64)[m], kind="stable")
    csi = a["csi"][m][order]
    return to_complex(csi, length), a["ts_us"][m][order].astype(np.int64), a["rssi"][m][order].astype(float), a["host_t"][m][order]


def common_phase(c, valid):
    return np.angle(c[:, valid].sum(1))


def slope(c, valid):
    """per-packet circular mean of adjacent-subcarrier phase steps (rad per subcarrier index)"""
    k = np.where(np.isin(valid[:-1], valid) & np.isin(valid[:-1] + 1, valid))[0]
    k = valid[:-1][np.isin(valid[:-1] + 1, valid)]
    return np.angle((c[:, k + 1] * np.conj(c[:, k])).sum(1)), k


def sanitize(c, valid, method):
    if method == "raw":
        return c
    if method == "ref":
        ref = valid[len(valid) // 2]
        return c * np.conj(c[:, [ref]])
    out = c
    if method == "cpo_slope":
        s, _ = slope(c, valid); idx = np.arange(c.shape[1]) - valid.mean()
        out = c * np.exp(-1j * s[:, None] * idx[None, :])
    return out * np.exp(-1j * common_phase(out, valid))[:, None]


def temporal_R(c, valid):
    """per-subcarrier circular concentration of phase over packets: 1 = perfectly stable, ~0 = random"""
    z = c[:, valid]; u = z / np.maximum(np.abs(z), 1e-9)
    return np.abs(u.mean(0))


def jump_fraction(c, valid, ts_us, max_dt_ms=40.0):
    """median over subcarriers of the fraction of consecutive-packet phase steps with |step| > pi/2"""
    dt = np.diff(ts_us) * 1e-3; ok = (dt > 0) & (dt <= max_dt_ms)
    ang = np.angle(c[:, valid]); d = np.angle(np.exp(1j * np.diff(ang, axis=0)))[ok]
    return float(np.median((np.abs(d) > np.pi / 2).mean(0))) if len(d) else float("nan")


def circ_sd(angles):
    R = abs(np.exp(1j * np.asarray(angles)).mean()); return float(np.sqrt(-2 * np.log(max(R, 1e-12))))