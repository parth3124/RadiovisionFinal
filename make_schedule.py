import argparse, random

CLASSES = ["idle", "hands_crossed", "hands_up", "arm_side", "arm_forward"]


def make(reps, rng):
    while True:
        seq = [c for c in range(5) for _ in range(reps)]
        rng.shuffle(seq)
        for _ in range(2000):
            bad = [i for i in range(1, len(seq)) if seq[i] == seq[i - 1]]
            if not bad:
                return seq
            i = bad[0]; j = rng.randrange(len(seq)); seq[i], seq[j] = seq[j], seq[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--session", default="s06")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    seq = make(a.reps, random.Random(a.seed))
    lines = [f"session {a.session}: {len(seq)} recordings, {a.reps} per gesture, shuffled"]
    for n, c in enumerate(seq, 1):
        lines.append(f"{n:3d}.  key {c + 1}   {CLASSES[c]}")
    txt = "\n".join(lines)
    print(txt)
    open(f"schedule_{a.session}.txt", "w").write(txt + "\n")
    print(f"\nsaved schedule_{a.session}.txt")


if __name__ == "__main__":
    main()
