#!/usr/bin/env python3
"""Record one continuous session from N receiver serial ports.

Run one copy per laptop. Receivers on different laptops need NO clock sync: they are
aligned afterwards by the TX packet counter (tx_boot, tx_seq) embedded in every frame.

    python record_session.py --ports /dev/ttyACM0 /dev/ttyACM1 --out sessions/s01_laptopA

Console commands while recording (Enter after each):
    m           sync marker (do the visible flash/clap at the same moment)
    s <label>   trial start     e   trial end     n <text>   note     q   quit
Each marker/event stores host monotonic time + the latest TX seq seen on this laptop.
Everything is written raw (host_t, frame bytes) so a crash loses nothing.
"""
import argparse, json, os, struct, sys, threading, time
import serial
from sura_frames import FrameParser, FT_CSI, FT_STATS, STATS

lock = threading.Lock()
latest = {"boot": None, "seq": None}
status = {}


def reader(port, outdir, stop):
    name = os.path.basename(port)
    f = open(os.path.join(outdir, f"raw_{name}.sbin"), "ab", buffering=1 << 16)
    ser = serial.Serial(port, 115200, timeout=0.05)
    ser.reset_input_buffer()
    p = FrameParser()
    st = status[name] = dict(rx_id=None, csi=0, last_stats=None, crc_fail=0, t0=time.monotonic(), n_win=0)
    while not stop.is_set():
        data = ser.read(ser.in_waiting or 1)
        if not data:
            continue
        t = time.monotonic()
        for h, body, raw in p.feed(data):
            f.write(struct.pack("<dH", t, len(raw)) + raw)   # host_t | frame_len | frame
            st["rx_id"] = h[3]
            if h[2] == FT_CSI:
                st["csi"] += 1
                st["n_win"] += 1
                with lock:
                    latest["boot"], latest["seq"] = h[7], h[8]
            elif h[2] == FT_STATS:
                st["last_stats"] = STATS.unpack(body)
        st["crc_fail"] = p.crc_fail
    f.close(); ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stop = threading.Event()
    ev = open(os.path.join(a.out, "events.jsonl"), "a", buffering=1)

    def log(kind, **kw):
        with lock:
            rec = dict(host_t=time.monotonic(), wall=time.time(), kind=kind,
                       latest_tx_boot=latest["boot"], latest_tx_seq=latest["seq"], **kw)
        ev.write(json.dumps(rec) + "\n")
        print(f"  [{kind}] seq~{rec['latest_tx_seq']} {kw}")

    log("session_start", ports=a.ports)
    for p in a.ports:
        threading.Thread(target=reader, args=(p, a.out, stop), daemon=True).start()

    def printer():
        while not stop.is_set():
            time.sleep(2)
            parts = []
            for n, s in status.items():
                dt = max(time.monotonic() - s["t0"], 1e-3); s["t0"] = time.monotonic()
                rate = s["n_win"] / dt; s["n_win"] = 0
                ls = s["last_stats"]
                parts.append(f"{n}(rx{s['rx_id']}) {rate:5.1f}Hz crc_fail={s['crc_fail']}"
                             + (f" matched={ls[0]} unmatched={ls[1]} orphan={ls[2]} drops={ls[3]}" if ls else ""))
            print(" | ".join(parts))
    threading.Thread(target=printer, daemon=True).start()

    print(__doc__.split("Console commands")[1])
    try:
        for line in sys.stdin:
            c = line.strip().split(maxsplit=1)
            if not c: continue
            if c[0] == "m": log("marker")
            elif c[0] == "s": log("trial_start", label=c[1] if len(c) > 1 else "")
            elif c[0] == "e": log("trial_end")
            elif c[0] == "n": log("note", text=c[1] if len(c) > 1 else "")
            elif c[0] == "q": break
    except KeyboardInterrupt:
        pass
    log("session_end"); stop.set(); time.sleep(0.3)


if __name__ == "__main__":
    main()
