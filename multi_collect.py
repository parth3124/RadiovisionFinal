"""
multi_collect.py - record CSI from several RX boards at once (one serial port each). Existing firmware and
common.SerialCSIReader are reused unchanged; every CSI line is stamped with the Mac's clock on arrival, so all
links share one time base.

1) Placement / health check (live, ~1 line per second; wave your arms near a board and watch its motion value):
    python multi_collect.py --check --ports /dev/cu.usbmodemA /dev/cu.usbmodemB ... --names NW NE SE SW
2) Protocol-driven recording (labels come from the protocol file, so they cannot be mislabelled):
    python multi_collect.py --ports ... --names ... --session r01 --protocol protocol_r01.txt --out dataset_room
       --voice   macOS spoken instructions ("still zone five, starting in eight")   --auto 4  hands-free, 4 s between recordings
       --start N resume at protocol line N

Each recording is saved as  <out>/<label>/<session>_<NN>_<label>/
    csi_<name>.npz   per link: host_t, seq, ts_us, rssi, csi (raw int8), csi_len   (same layout as before)
    meta.json        label, presence, motion, zone, session, links, per-link counts / rates
"""
import argparse, json, os, subprocess, time
import numpy as np
import common as C

MOVING = ("walk", "wave", "squat", "jump")
HUMAN = MOVING + ("person", "still")


def parse_label(label):
    if label == "empty":
        return "empty", None
    cond, sep, z = label.partition("_Z")
    return (cond, "Z" + z) if sep else (label, None)


def speak(on, text):
    if on:
        subprocess.Popen(["say", text])


def live_motion(r, sec=1.0, lag=5):
    """relative CSI change over the last `sec` seconds (per-packet gain removed); None if not enough data"""
    with r.lock:
        if len(r.live_t) < 10:
            return None
        t = np.fromiter(r.live_t, float, len(r.live_t)); A = np.stack(r.live_amp)
    A = A[t > t[-1] - sec]
    if len(A) <= lag + 2:
        return None
    A = A / np.maximum(A.mean(1, keepdims=True), 1e-3)
    return float(np.abs(A[lag:] - A[:-lag]).mean())


def start_readers(ports, baud):
    rs = [C.SerialCSIReader(p, baud) for p in ports]
    for r in rs:
        r.start()
    return rs


def wait_all(rs, names, timeout=25):
    t0 = time.monotonic()
    while True:
        if all(r.n_sub is not None for r in rs):
            return
        if any(r.error for r in rs) or time.monotonic() - t0 > timeout:
            for n, r in zip(names, rs):
                print(f"  {n}: " + (f"SERIAL ERROR {r.error}" if r.error else
                      ("OK, CSI flowing" if r.n_sub else f"NO CSI (last non-CSI line: {r.last_text!r})")))
            raise SystemExit("not every link delivers CSI. Check ports, that TX is on, and that no Serial Monitor is open.")
        time.sleep(0.2)


def check(rs, names, seconds):
    print("link health, one line per second (Ctrl+C to stop).  motion = relative CSI change; try waving near a board.")
    t0 = time.monotonic()
    try:
        while not seconds or time.monotonic() - t0 < seconds:
            time.sleep(1.0)
            parts = []
            for n, r in zip(names, rs):
                m = live_motion(r)
                bar = "-" if m is None else "#" * min(20, int(m * 200))
                parts.append(f"{n}: {r.rate:4.0f}Hz bad={r.n_bad} mot={('--' if m is None else f'{m:.3f}')} {bar:<8}")
            print(" | ".join(parts))
    except KeyboardInterrupt:
        pass


def parse_protocol(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            n, lab = line.split(".", 1); out.append((int(n), lab.strip()))
    return out


HINTS = {"empty": ("REMOVE the object, LEAVE the room, stay out until the end", "empty room"),
         "walk_free": ("WALK around the room continuously, covering all zones", "walk"),
         "wave": ("stand in the middle and WAVE both arms widely", "wave"),
         "squat": ("stand in the middle and SQUAT and stand up repeatedly", "squat"),
         "jump": ("stand in the middle and JUMP repeatedly", "jump")}


def describe(label):
    cond, zone = parse_label(label)
    if zone and cond == "obj":
        return (f"PICK UP the object and place it fresh on the mark of zone {zone[1:]}, then LEAVE the room and stay out",
                f"object, zone {zone[1:]}")
    if zone:
        return f"walk to zone {zone[1:]} and stand STILL on the mark, facing the same way", f"person, zone {zone[1:]}"
    return HINTS.get(label, (label, label))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-seconds", type=float, default=0)
    ap.add_argument("--session")
    ap.add_argument("--protocol")
    ap.add_argument("--out", default="dataset_room")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--auto", type=float, default=0.0, help="seconds between recordings instead of waiting for Enter")
    ap.add_argument("--voice", action="store_true")
    a = ap.parse_args()
    names = a.names or [f"rx{i + 1}" for i in range(len(a.ports))]
    if len(names) != len(a.ports):
        ap.error("--names must have one name per port")
    rs = start_readers(a.ports, a.baud)
    print("waiting for CSI on every link...")
    wait_all(rs, names)
    print("all links delivering CSI, subcarriers:", {n: r.n_sub for n, r in zip(names, rs)})
    if a.check:
        check(rs, names, a.check_seconds)
    else:
        if not (a.session and a.protocol):
            ap.error("--session and --protocol are required for recording")
        run(a, rs, names)
    for r in rs:
        r.stop()


def run(a, rs, names):
    items = [it for it in parse_protocol(a.protocol) if it[0] >= a.start]
    print(f"{len(items)} recordings to go (from #{a.start}); {a.duration:.0f} s each, countdown {a.countdown:.0f} s")
    saved = 0
    try:
        for n, label in items:
            text, spoken = describe(label)
            print(f"\n#{n}  {label}:  {text}")
            if a.auto > 0:
                speak(a.voice, f"next: {spoken}"); time.sleep(a.auto)
            else:
                input("   press Enter to start the countdown > ")
                speak(a.voice, spoken)
            t_end = time.monotonic() + a.countdown
            while time.monotonic() < t_end:
                print(f"\r   starting in {t_end - time.monotonic():3.0f} s   ", end="", flush=True); time.sleep(0.25)
            print("\r   RECORDING...                    ")
            speak(a.voice, "go")
            for r in rs:
                r.start_recording()
            t0 = time.monotonic(); time.sleep(a.duration); dur = time.monotonic() - t0
            rows = [r.stop_recording() for r in rs]
            speak(a.voice, "stop")
            d = os.path.join(a.out, label, f"{a.session}_{n:02d}_{label}")
            os.makedirs(d, exist_ok=True)
            counts = {}
            for name, rr in zip(names, rows):
                if rr:
                    C.save_csi_rows(rr, os.path.join(d, f"csi_{name}.npz"))
                counts[name] = len(rr)
            med = float(np.median(list(counts.values()))) or 1.0
            weak = [k for k, v in counts.items() if v < 0.6 * med or v < 0.5 * C.CSI_FS * dur * 0.58]
            cond, zone = parse_label(label)
            meta = {"label": label, "condition": cond, "zone": zone, "object_present": int(cond == "obj"),
                    "human_present": int(cond.startswith(HUMAN)), "motion": int(cond.startswith(MOVING)),
                    "session": a.session, "protocol_line": n,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "duration_s": round(dur, 2), "links": names,
                    "ports": a.ports, "csi_samples": counts, "csi_rate_hz": {k: round(v / dur, 1) for k, v in counts.items()},
                    "weak_links": weak, "csi_malformed_total": {nm: r.n_bad for nm, r in zip(names, rs)},
                    "time_base": "time.monotonic() on the Mac, shared by all links"}
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
            saved += 1
            print("   saved " + d + "   samples " + ", ".join(f"{k} {v}" for k, v in counts.items())
                  + (f"   <- WEAK LINK(S): {weak}, consider redoing" if weak else ""))
    except KeyboardInterrupt:
        print("\nstopped; resume with --start", n)
    print(f"\n{saved} recordings saved to {a.out}/")


if __name__ == "__main__":
    main()