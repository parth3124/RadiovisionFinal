#!/usr/bin/env python3
"""Record the overhead camera with a host-clock timestamp for every frame.

Uses time.monotonic(), the same clock as record_session.py, so run it on the SAME laptop.
    python record_camera.py --device 0 --out sessions/s01_laptopA --fps 30

Writes camera.mp4 and camera_frames.csv (frame_idx, host_t). The timestamp is taken when
the frame is delivered, which lags exposure by roughly 1-3 frames on USB webcams; the
visible sync flash/clap at each marker measures and removes that lag in merge_session.py.
Ctrl-C to stop.
"""
import argparse, csv, os, time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cap = cv2.VideoCapture(a.device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
cap.set(cv2.CAP_PROP_FPS, a.fps)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vw = cv2.VideoWriter(os.path.join(a.out, "camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
csvf = open(os.path.join(a.out, "camera_frames.csv"), "w", newline="")
cw = csv.writer(csvf); cw.writerow(["frame_idx", "host_t"])
i = 0
t0 = None
try:
    while True:
        ok, frame = cap.read()
        t = time.monotonic()
        if not ok:
            continue
        t0 = t if t0 is None else t0
        vw.write(frame); cw.writerow([i, f"{t:.6f}"]); i += 1
        if i % 300 == 0:
            print(f"{i} frames, mean fps {i/(t-t0):.1f}")
            csvf.flush()
except KeyboardInterrupt:
    pass
vw.release(); csvf.close(); cap.release()
print(f"saved {i} frames. NOTE: the video is written at nominal {a.fps} fps; use camera_frames.csv for true times.")
