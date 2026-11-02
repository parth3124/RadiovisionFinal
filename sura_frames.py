"""Frame format shared by the recorder, merger and simulator (mirrors include/sura_proto.h)."""
import struct

HDR = struct.Struct("<BBBBHBbIIII")      # 24 bytes
HDR_SIZE = HDR.size
FT_CSI, FT_STATS = 1, 2
FL_FIRST_WORD_INVALID, FL_UNMATCHED = 0x10, 0x20
STATS = struct.Struct("<IIII")
MAGIC = b"\xa5\x5a"


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def pack_frame(ftype, rx_id, flags, rssi, tx_boot, tx_seq, tx_us, rx_ts, body: bytes) -> bytes:
    h = HDR.pack(0xA5, 0x5A, ftype, rx_id, len(body), flags, rssi, tx_boot, tx_seq, tx_us, rx_ts)
    return h + body + struct.pack("<H", crc16(h + body))


class FrameParser:
    """Resynchronising parser: feed() raw serial bytes, get (hdr_tuple, body, raw_frame) back."""
    def __init__(self):
        self.buf = bytearray()
        self.crc_fail = 0

    def feed(self, data: bytes):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                del self.buf[:-1]
                break
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < HDR_SIZE:
                break
            h = HDR.unpack_from(self.buf)
            n = h[4]
            if n > 700:
                del self.buf[:1]
                continue
            total = HDR_SIZE + n + 2
            if len(self.buf) < total:
                break
            raw = bytes(self.buf[:total])
            if crc16(raw[:-2]) != struct.unpack("<H", raw[-2:])[0]:
                self.crc_fail += 1
                del self.buf[:1]
                continue
            out.append((h, raw[HDR_SIZE:-2], raw))
            del self.buf[:total]
        return out
