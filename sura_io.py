"""
sura_io.py - read-only access to the original SURA recordings (never writes into the dataset).

Recording layout (written by multi_collect.py):   <root>/<label>/<session>_<NN>_<label>/{meta.json, csi_<link>.npz}
Also supports a single loose pair:  --npz FILE [--meta FILE].
Arrays in csi_<link>.npz: host_t (float64, Mac monotonic s), seq (uint32), ts_us (uint32, ESP32 receive clock),
rssi (int8), csi (int8 [n, max_len], zero padded), csi_len (int16 [n]).
"""
import glob, json, os
import numpy as np

BLOCK = 128          # int8 values per 64-subcarrier block (64 x [first byte, second byte])
REQUIRED = ("host_t", "seq", "ts_us", "rssi", "csi", "csi_len")


def discover(root):
    """-> list of (npz_path, meta_path) for every csi_*.npz that sits next to a meta.json"""
    out = []
    mps = sorted(glob.glob(os.path.join(root, "*", "*", "meta.json")))
    if not mps:                                   # other folder depth: search recursively
        mps = sorted(glob.glob(os.path.join(root, "**", "meta.json"), recursive=True))
    for mp in mps:
        for f in sorted(glob.glob(os.path.join(os.path.dirname(mp), "csi_*.npz"))):
            out.append((f, mp))
    return out


def load(npz_path, meta_path=None):
    """-> dict with arrays (copies), meta (dict or {}), link name, problems (list of strings)"""
    problems = []
    with np.load(npz_path) as z:
        missing = [k for k in REQUIRED if k not in z.files]
        arr = {k: z[k].copy() for k in z.files}
    if missing:
        problems.append(f"missing arrays {missing}")
    meta = {}
    if meta_path and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    elif meta_path:
        problems.append("meta.json not found")
    base = os.path.basename(npz_path)
    link = base[4:-4] if base.startswith("csi_") else base
    # loose files may be renamed (csi_link_5fad50.npz): fall back to the single link named in meta.json
    mk = link if link in meta.get("csi_samples", {}) else (meta["links"][0] if len(meta.get("links", [])) == 1 else link)
    return dict(path=npz_path, meta_path=meta_path, link=link, meta_key=mk, meta=meta, a=arr, problems=problems,
                session=str(meta.get("session", "?")), label=str(meta.get("label", "?")),
                condition=str(meta.get("condition", "?")), zone=meta.get("zone"))
