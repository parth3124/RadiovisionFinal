"""
check_room_session.py - verify a room session against its protocol.
    python check_room_session.py --data dataset_room --session r01 --protocol protocol_r01.txt
Reports counts per label, saved order vs protocol, immediate repeats, weak links / low CSI rates.
"""
import argparse, glob, json, os
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_room")
    ap.add_argument("--session", required=True)
    ap.add_argument("--protocol")
    a = ap.parse_args()
    ms = []
    for p in glob.glob(os.path.join(a.data, "*", "*", "meta.json")):
        m = json.load(open(p))
        if m.get("session") == a.session:
            m["_dir"] = os.path.dirname(p); ms.append(m)
    ms.sort(key=lambda m: m["created"])
    print(f"session {a.session}: {len(ms)} recordings")
    for lab, n in sorted(Counter(m["label"] for m in ms).items()):
        print(f"  {lab:<14}{n:>3}")
    saved = [m["label"] for m in ms]
    if a.protocol:
        prot = []
        for line in open(a.protocol):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                prot.append(line.split(".", 1)[1].strip())
        print("\nsaved order MATCHES the protocol exactly" if saved == prot else
              f"\nsaved order differs from the protocol ({len(saved)} saved vs {len(prot)} planned); fine only if you redid or skipped recordings")
    print(f"immediate repeats: {sum(saved[i] == saved[i - 1] for i in range(1, len(saved)))}")
    bad = [(os.path.basename(m['_dir']), m["weak_links"], m["csi_rate_hz"]) for m in ms
           if m.get("weak_links") or min(m["csi_rate_hz"].values()) < 45]
    print("\nweak-link / low-rate recordings: " + ("none" if not bad else ""))
    for b in bad:
        print("  ", b)


if __name__ == "__main__":
    main()