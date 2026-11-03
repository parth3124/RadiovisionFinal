"""
check_session.py - verify a newly recorded session against its shuffled schedule.

    python check_session.py --data dataset_v2 --session s06 --schedule schedule_s06.txt
Reads only meta.json. Reports: count per gesture, saved order vs schedule, recordings with low CSI rate,
missing pose, or short duration.
"""
import argparse, glob, json, os
import datetime as dt

CLASSES = ["idle", "hands_crossed", "hands_up", "arm_side", "arm_forward"]
LET = "ICUSF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v2")
    ap.add_argument("--session", required=True)
    ap.add_argument("--schedule")
    a = ap.parse_args()
    recs = []
    for p in glob.glob(os.path.join(a.data, "*", "*", "meta.json")):
        m = json.load(open(p))
        if m.get("session") == a.session:
            m["_dir"] = os.path.dirname(p); recs.append(m)
    recs.sort(key=lambda m: m.get("created", ""))
    print(f"session {a.session}: {len(recs)} recordings")
    for c in CLASSES:
        print(f"  {c:<14}{sum(m['gesture'] == c for m in recs):>3}")
    order = [CLASSES.index(m["gesture"]) for m in recs]
    print("\nsaved order:    " + "".join(LET[c] for c in order))
    if a.schedule:
        sched = []
        for line in open(a.schedule):
            if "key" in line:
                sched.append(int(line.split("key")[1].split()[0]) - 1)
        print("schedule order: " + "".join(LET[c] for c in sched))
        if order == sched:
            print("-> saved order MATCHES the schedule exactly")
        else:
            diff = [i + 1 for i in range(min(len(order), len(sched))) if order[i] != sched[i]]
            print(f"-> differs from the schedule (first differing positions: {diff[:10]}); "
                  f"{len(order)} saved vs {len(sched)} scheduled. Fine if you redid recordings, otherwise check labels.")
    runs = sum(order[i] == order[i - 1] for i in range(1, len(order)))
    print(f"adjacent identical gestures: {runs}")
    print("\nquality warnings:")
    bad = 0
    for m in recs:
        w = []
        if m.get("csi_rate_hz", 99) < 50: w.append(f"CSI rate {m['csi_rate_hz']} Hz")
        if m.get("camera_frames") and m.get("pose_frames", 0) < 0.8 * m["camera_frames"]:
            w.append(f"pose in only {m['pose_frames']}/{m['camera_frames']} frames")
        if m.get("duration_s", 99) < 18: w.append(f"short ({m['duration_s']} s)")
        if w:
            bad += 1; print(f"  {os.path.basename(m['_dir'])}: " + ", ".join(w))
    print("  none" if not bad else f"  {bad} recording(s) flagged")


if __name__ == "__main__":
    main()
    