"""
make_protocol.py - write a SHUFFLED recording protocol for one room session (no class blocks, no immediate repeats).

    python make_protocol.py --session r01 --seed 1                      # stage 1: empty / still-in-zone / walking
    python make_protocol.py --session r01 --seed 1 --gestures wave squat jump --gesture-reps 6   # + stage 2

STEP 1 (1 TX + 1 RX, 3 zones along the TX-RX line):
    python make_protocol.py --session r01 --seed 1                  # default: 8 empty + 6x3 obj + 6x3 person
Labels (one per recording, performed for --duration seconds in multi_collect.py):
  empty         no object in any zone, nobody near the link (leave before the countdown ends and stay out)
  obj_Z<k>      the OBJECT sits in zone k, nobody near the link
  person_Z<k>   a PERSON stands still in zone k (positive control: a person is a much bigger scatterer)
  still_Z<k>    (later, 9 zones) person standing still in zone k of the 3x3 grid
  walk_free     walk around the room continuously, random path, covering all zones
  wave|squat|jump|...   large dynamic gesture, performed in the middle of the room
"""
import argparse, random

HINTS = {"empty": "REMOVE the object, LEAVE the room and stay out of it",
         "walk_free": "WALK around the room continuously, covering all zones",
         "wave": "stand in the middle and WAVE both arms widely and continuously",
         "squat": "stand in the middle and SQUAT down and stand up, repeatedly",
         "jump": "stand in the middle and JUMP repeatedly"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--zones", type=int, default=3)
    ap.add_argument("--empty-reps", type=int, default=8)
    ap.add_argument("--obj-reps", type=int, default=6, help="recordings per zone with the object")
    ap.add_argument("--person-reps", type=int, default=6, help="recordings per zone with a person")
    ap.add_argument("--still-reps", type=int, default=0, help="(later) recordings per zone, person still, 3x3 grid")
    ap.add_argument("--walk-reps", type=int, default=0)
    ap.add_argument("--gestures", nargs="*", default=[])
    ap.add_argument("--gesture-reps", type=int, default=6)
    ap.add_argument("--duration", type=float, default=10.0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    labels = ["empty"] * a.empty_reps + ["walk_free"] * a.walk_reps
    labels += [f"obj_Z{z}" for z in range(1, a.zones + 1) for _ in range(a.obj_reps)]
    labels += [f"person_Z{z}" for z in range(1, a.zones + 1) for _ in range(a.person_reps)]
    labels += [f"still_Z{z}" for z in range(1, a.zones + 1) for _ in range(a.still_reps)]
    labels += [g for g in a.gestures for _ in range(a.gesture_reps)]
    while True:
        rng.shuffle(labels)
        for _ in range(5000):
            bad = [i for i in range(1, len(labels)) if labels[i] == labels[i - 1]]
            if not bad:
                break
            i = bad[0]; j = rng.randrange(len(labels)); labels[i], labels[j] = labels[j], labels[i]
        if not any(labels[i] == labels[i - 1] for i in range(1, len(labels))):
            break
    lines = [f"session {a.session}: {len(labels)} recordings of {a.duration:.0f} s, shuffled"]
    lines += [f"{n:3d}. {l}" for n, l in enumerate(labels, 1)]
    open(f"protocol_{a.session}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]) + ("\n..." if len(labels) > 11 else ""))
    mins = len(labels) * (a.duration + 8 + 4) / 60
    print(f"\nsaved protocol_{a.session}.txt  (about {mins:.0f} min with 8 s countdown and short gaps)")


if __name__ == "__main__":
    main()