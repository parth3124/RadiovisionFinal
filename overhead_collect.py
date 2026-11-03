#!/usr/bin/env python3
"""
overhead_collect.py - record CSI (1..N RX boards) and the overhead-camera ground-truth position TOGETHER, on one clock.

    python overhead_collect.py --calib calib.json --cam 0 --ports /dev/cu.usbmodem1101 --names link --session o01 --out dataset_over
    python overhead_collect.py --calib calib.json --cam 0 --ports /dev/cu.usbmodem* --auto-names --session o01 --out dataset_over   (new multi-RX firmware)

Window keys:  1 = walk_free (walk continuously, covering the whole area)   2 = stop_and_go (stand still ~3 s, move to another spot, repeat)
              3 = empty (nobody in the room)       SPACE = start a take (countdown, then --duration s)  / stop it early      q = quit
Every frame the cap is detected, converted to floor metres and stamped with time.monotonic() right after the frame is read - the same
clock that stamps every CSI line. A take is saved to  <out>/<label>/<session>_<NN>_<label>/ :
    csi_<link>.npz   per link (same layout as before)      track.npz   t, px, py, x, y, visible (one row per camera frame)      meta.json
Suggested session (about 15 min): 6 x walk_free, 6 x stop_and_go, 3 x empty, in SHUFFLED order (use --plan to get one).
"""
import argparse, hashlib, json, os, shutil, time
import numpy as np
import cv2
import common as C
import overhead_common as OC
from multi_collect import start_readers, wait_all, detect_names, speak

LABELS = {ord("1"): "walk_free", ord("2"): "stop_and_go", ord("3"): "empty"}
HINT = {"walk_free": ("WALK around continuously, covering the whole area", "walk around"),
        "stop_and_go": ("STAND STILL about 3 s, move to another spot, repeat; use every part of the area", "stop and go"),
        "empty": ("LEAVE the room, nobody inside until the take ends", "empty room, please leave")}


def make_plan(n_walk, n_stop, n_empty, seed):
    import random
    rng = random.Random(seed)
    labs = ["walk_free"] * n_walk + ["stop_and_go"] * n_stop + ["empty"] * n_empty
    for _ in range(1000):
        rng.shuffle(labs)
        if all(labs[i] != labs[i - 1] for i in range(1, len(labs))):
            break
    return labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib.json")
    ap.add_argument("--cam", type=int)
    ap.add_argument("--ports", nargs="+")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--auto-names", dest="auto_names", action="store_true")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--session")
    ap.add_argument("--out", default="dataset_over")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--countdown", type=float, default=5.0)
    ap.add_argument("--countdown-empty", dest="countdown_empty", type=float, default=15.0, help="time to leave the room")
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print a shuffled take order for a session and exit")
    ap.add_argument("--plan-walk", type=int, default=6); ap.add_argument("--plan-stop", type=int, default=6)
    ap.add_argument("--plan-empty", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.plan:
        for i, l in enumerate(make_plan(a.plan_walk, a.plan_stop, a.plan_empty, a.seed), 1):
            print(f"{i:2d}. {l}")
        return
    if not (a.ports and a.session):
        ap.error("--ports and --session are required")
    calib = OC.load_calib(a.calib)
    cam_idx = calib["cam"] if a.cam is None else a.cam
    calib_sha = hashlib.sha1(open(a.calib, "rb").read()).hexdigest()[:10]
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(os.path.join(a.out, "calib.json")):
        shutil.copy(a.calib, os.path.join(a.out, "calib.json"))
    names = a.names or [f"rx{i + 1}" for i in range(len(a.ports))]
    if len(names) != len(a.ports):
        ap.error("--names must have one name per port")
    rs = start_readers(a.ports, a.baud)
    print("waiting for CSI on every link...")
    wait_all(rs, names)
    if a.auto_names:
        names = detect_names(rs, a.ports)
    print("links:", dict(zip(a.ports, names)))
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calib["frame"][0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib["frame"][1])
    if not cap.read()[0]:
        raise SystemExit(f"camera {cam_idx} not readable")
    area, zones = tuple(calib["area"]), tuple(calib["zones"])
    label, state, take, n_saved = "walk_free", "idle", None, 0
    existing = [d for l in os.listdir(a.out) if os.path.isdir(os.path.join(a.out, l)) for d in os.listdir(os.path.join(a.out, l))
                if d.startswith(a.session + "_")]
    take_no = len(existing)
    print("keys: 1 walk_free, 2 stop_and_go, 3 empty, SPACE start/stop, q quit")

    def save(take):
        nonlocal n_saved
        rows = [r.stop_recording() for r in rs]
        dur = take["t_stop"] - take["t_start"]
        counts = {n: len(rr) for n, rr in zip(names, rows)}
        d = os.path.join(a.out, take["label"], f"{a.session}_{take['no']:02d}_{take['label']}")
        os.makedirs(d, exist_ok=True)
        for n, rr in zip(names, rows):
            if rr:
                C.save_csi_rows(rr, os.path.join(d, f"csi_{n}.npz"))
        tr = np.array(take["track"], np.float64).reshape(-1, 6)
        np.savez_compressed(os.path.join(d, "track.npz"), t=tr[:, 0], px=tr[:, 1], py=tr[:, 2], x=tr[:, 3], y=tr[:, 4],
                            visible=tr[:, 5] > 0.5)
        vis = float(np.mean(tr[:, 5])) if len(tr) else 0.0
        meta = {"kind": "overhead", "label": take["label"], "condition": take["label"], "zone": None, "session": a.session,
                "take": take["no"], "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "duration_s": round(dur, 2), "links": names,
                "csi_samples": counts, "csi_rate_hz": {k: round(v / dur, 1) for k, v in counts.items()},
                "camera_frames": len(tr), "camera_fps": round(len(tr) / dur, 1), "visible_fraction": round(vis, 3),
                "calibration_sha": calib_sha, "area": list(area), "zones": list(zones),
                "time_base": "time.monotonic() on the Mac, shared by CSI and camera"}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
        warn = []
        if take["label"] != "empty" and vis < 0.8:
            warn.append(f"cap visible in only {100 * vis:.0f}% of frames")
        if take["label"] == "empty" and vis > 0.05:
            warn.append(f"cap visible in {100 * vis:.0f}% of an EMPTY take (someone in view?)")
        if min(counts.values()) < 0.5 * C.CSI_FS * 0.58 * dur:
            warn.append(f"low CSI count {counts}")
        n_saved += 1
        print(f"  saved {d}: {dur:.0f} s, CSI {counts}, camera {len(tr)} frames ({meta['camera_fps']} fps), cap visible {100 * vis:.0f}%"
              + (f"   <- WARNING: {'; '.join(warn)}" if warn else ""))

    t_state = 0.0
    while True:
        ok, frame = cap.read()
        now = time.monotonic()
        if not ok:
            print("camera stopped"); break
        det, mask = OC.detect_marker(frame, calib["hsv_lo"], calib["hsv_hi"], calib.get("min_area_px", 60))
        row = (now, np.nan, np.nan, np.nan, np.nan, 0.0)
        view = frame.copy()
        OC.draw_area(view, {"H": calib["H"], "area": area, "zones": zones})
        txt = "cap not seen"
        if det:
            x, y = OC.pix_to_world(calib["H"], det[0], det[1]); z = OC.zone_index(x, y, area, zones)
            row = (now, det[0], det[1], x, y, 1.0)
            cv2.circle(view, (int(det[0]), int(det[1])), 8, (0, 0, 255), 2)
            txt = f"x={x:5.2f} y={y:5.2f} m  zone {z + 1 if z >= 0 else '-'}"
        if state == "countdown" and now >= t_state:
            take = {"label": label, "no": take_no, "t_start": now, "track": []}; take_no += 1
            for r in rs:
                r.start_recording()
            state = "recording"; speak(a.voice, "go"); print(f"RECORDING {label} (take {take['no']})")
        if state == "recording":
            take["track"].append(row)
            if now - take["t_start"] >= a.duration:
                take["t_stop"] = now; save(take); take = None; state = "idle"; speak(a.voice, "stop")
        C.put_text(cv2, view, f"label: {label}   {txt}", (8, 22), 0.55, (0, 255, 255))
        if state == "recording":
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 0, 255), 6)
            C.put_text(cv2, view, f"RECORDING {now - take['t_start']:4.0f}/{a.duration:.0f} s  CSI {[r.n_recorded() for r in rs]}", (8, 48), 0.55, (0, 0, 255))
        elif state == "countdown":
            C.put_text(cv2, view, f"GET READY: {HINT[label][0]}  ({t_state - now:3.0f})", (8, 48), 0.5, (0, 255, 255))
        else:
            C.put_text(cv2, view, "idle: 1/2/3 label, SPACE record, q quit   " + " ".join(f"{n}:{r.rate:.0f}Hz" for n, r in zip(names, rs)), (8, 48), 0.45, (0, 255, 0))
        cv2.imshow("overhead collector", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if state == "idle" and key in LABELS:
            label = LABELS[key]; print("label ->", label)
        if key == ord(" "):
            if state == "idle":
                state = "countdown"; t_state = now + (a.countdown_empty if label == "empty" else a.countdown)
                speak(a.voice, HINT[label][1])
            elif state == "recording":
                take["t_stop"] = now; save(take); take = None; state = "idle"
    if state == "recording" and take:
        take["t_stop"] = time.monotonic(); save(take)
    for r in rs:
        r.stop()
    cap.release(); cv2.destroyAllWindows()
    print(f"done, {n_saved} takes saved to {a.out}/")


if __name__ == "__main__":
    main()
