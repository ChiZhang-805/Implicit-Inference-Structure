import ast
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_duration(duration):
    """Parse many duration formats into (start_sec, end_sec)."""
    if duration is None:
        return None
    data = duration
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return None
        try:
            data = ast.literal_eval(s)
        except Exception:
            # fallback "a,b"
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 2:
                    a, b = _to_float(parts[0]), _to_float(parts[1])
                    if a is not None and b is not None:
                        return (min(a, b), max(a, b))
            return None

    def flatten(v):
        if isinstance(v, (list, tuple)):
            out = []
            for e in v:
                out.extend(flatten(e))
            return out
        f = _to_float(v)
        return [] if f is None else [f]

    vals = flatten(data)
    if len(vals) < 2:
        return None
    a, b = vals[0], vals[1]
    if math.isfinite(a) and math.isfinite(b):
        return (min(a, b), max(a, b))
    return None


def is_inside_mask(t, mask_range, eps=1e-3):
    if mask_range is None:
        return False
    s, e = mask_range
    return (t + eps) >= s and (t - eps) <= e


def non_evidence_segments(total_duration, mask_range):
    td = max(float(total_duration or 0.0), 0.0)
    if td <= 0:
        return [(0.0, 0.0)]
    if mask_range is None:
        return [(0.0, td)]
    s, e = max(0.0, mask_range[0]), min(td, mask_range[1])
    eps = min(1e-3, td / 1000.0)
    segs = []
    if s > 0:
        segs.append((0.0, max(0.0, s - eps)))
    if e < td:
        segs.append((min(td, e + eps), td))
    if segs:
        return [(a, b) for a, b in segs if b >= a]
    # mask covers full video: no non-evidence interval exists.
    return []


def infer_temporal_query_type(question):
    q = (question or "").strip().lower()
    if any(w in q for w in ["after", "following", "later", "next", " end"]):
        return "after"
    if any(w in q for w in ["before", "prior", "earlier", "start", "beginning"]):
        return "before"
    if q.startswith("why") or "reason" in q or "purpose" in q:
        return "why"
    if q.startswith("how"):
        return "how"
    if q.startswith("what"):
        return "what"
    return "balanced"


def _timestamps(vr):
    n = len(vr)
    if n == 0:
        return []
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    fps = fps if fps > 1e-6 else 30.0
    return [i / fps for i in range(n)]


def _sample_from_indices(indices, k, rng):
    if not indices:
        return []
    if len(indices) >= k:
        step = len(indices) / float(k)
        out = []
        for i in range(k):
            lo = int(i * step)
            hi = int((i + 1) * step)
            hi = max(lo + 1, min(hi, len(indices)))
            out.append(indices[rng.randrange(lo, hi)])
        return out
    out = indices[:]
    while len(out) < k:
        out.append(out[-1])
    return out


def sample_tcr_frame_indices(vr, num_frames, question, mask_duration=None, all_duration=None, mode="tcr", seed=None):
    rng = random.Random(seed)
    ts = _timestamps(vr)
    n = len(ts)
    if n == 0:
        return [0] * num_frames, [0.0] * num_frames
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    total_duration = (n - 1) / max(fps, 1e-6)
    if all_duration is not None:
        parsed_all = parse_duration(all_duration)
        if parsed_all is not None:
            total_duration = max(total_duration, parsed_all[1])

    mask_range = parse_duration(mask_duration)
    valid = [i for i, t in enumerate(ts) if not is_inside_mask(t, mask_range)]
    if not valid:
        # impossible strict mask; fallback to all but keep diagnostics by assertion disabled.
        valid = list(range(n))

    qtype = infer_temporal_query_type(question) if mode == "tcr" else "balanced"
    before, after = [], []
    if mask_range is not None:
        s, e = mask_range
        before = [i for i in valid if ts[i] < s]
        after = [i for i in valid if ts[i] > e]

    global_valid = valid
    if qtype == "after":
        alloc = [int(num_frames * 0.5), int(num_frames * 0.25)]
        alloc.append(num_frames - sum(alloc))
        sel = _sample_from_indices(after or global_valid, alloc[0], rng)
        sel += _sample_from_indices(before or global_valid, alloc[1], rng)
        sel += _sample_from_indices(global_valid, alloc[2], rng)
    elif qtype == "before":
        alloc = [int(num_frames * 0.5), int(num_frames * 0.25)]
        alloc.append(num_frames - sum(alloc))
        sel = _sample_from_indices(before or global_valid, alloc[0], rng)
        sel += _sample_from_indices(after or global_valid, alloc[1], rng)
        sel += _sample_from_indices(global_valid, alloc[2], rng)
    elif qtype in {"why", "how", "what"}:
        b = num_frames // 3
        a = num_frames // 3
        g = num_frames - b - a
        sel = _sample_from_indices(before or global_valid, b, rng)
        sel += _sample_from_indices(after or global_valid, a, rng)
        sel += _sample_from_indices(global_valid, g, rng)
    else:
        sel = _sample_from_indices(global_valid, num_frames, rng)

    if len(sel) < num_frames:
        sel = _sample_from_indices(sel or global_valid, num_frames, rng)
    sel = sel[:num_frames]
    secs = [round(ts[i], 3) for i in sel]

    if mask_range is not None and valid:
        assert all(not is_inside_mask(t, mask_range) for t in secs), "duration leak in selected frames"
    return sel, secs


def sample_tcr_multi_views(vr, num_frames, question, mask_duration=None, all_duration=None, seed=None):
    idx_g, sec_g = sample_tcr_frame_indices(vr, num_frames, question, mask_duration, all_duration, mode="balanced", seed=seed)
    idx_b, sec_b = sample_tcr_frame_indices(vr, num_frames, "before and after boundary", mask_duration, all_duration, mode="tcr", seed=None if seed is None else seed + 1)
    idx_q, sec_q = sample_tcr_frame_indices(vr, num_frames, question, mask_duration, all_duration, mode="tcr", seed=None if seed is None else seed + 2)
    return {
        "global": {"indices": idx_g, "seconds": sec_g},
        "boundary": {"indices": idx_b, "seconds": sec_b},
        "query": {"indices": idx_q, "seconds": sec_q},
    }
