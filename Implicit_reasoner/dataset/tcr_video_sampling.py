import ast
import math
import random
from typing import Optional, Tuple


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_duration(duration) -> Optional[Tuple[float, float]]:
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
            parts = [p.strip() for p in s.replace("-", ",").split(",") if p.strip()]
            if len(parts) >= 2:
                a, b = _to_float(parts[0]), _to_float(parts[1])
                if a is not None and b is not None:
                    return (min(a, b), max(a, b))
            return None

    def flatten(obj):
        if isinstance(obj, (list, tuple)):
            out = []
            for item in obj:
                out.extend(flatten(item))
            return out
        f = _to_float(obj)
        return [] if f is None else [f]

    vals = flatten(data)
    if len(vals) < 2:
        return None
    a, b = vals[0], vals[1]
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return (min(a, b), max(a, b))


def is_inside_mask(t, mask_range, eps=1e-3):
    if mask_range is None:
        return False
    s, e = mask_range
    return (t + eps) >= s and (t - eps) <= e


def _timestamps(vr):
    n = len(vr)
    if n <= 0:
        return []
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    fps = fps if fps > 1e-6 else 30.0
    return [i / fps for i in range(n)]


def _deterministic_pick(indices, k):
    if not indices:
        return []
    if len(indices) >= k:
        if k == 1:
            return [indices[len(indices) // 2]]
        step = (len(indices) - 1) / float(k - 1)
        return [indices[min(len(indices) - 1, int(round(i * step)))] for i in range(k)]
    out = list(indices)
    while len(out) < k:
        out.append(indices[len(out) % len(indices)])
    return out


def sample_tcr_frame_indices(vr, num_frames, question, mask_duration=None, all_duration=None, mode="tcr", seed=None):
    del question, all_duration, mode  # reserved for future adaptive policies
    if num_frames <= 0:
        return [], []
    ts = _timestamps(vr)
    if not ts:
        raise RuntimeError("VideoReader has no frames; cannot sample TCR frames.")

    mask = parse_duration(mask_duration)
    legal_indices = [i for i, t in enumerate(ts) if not is_inside_mask(t, mask)]
    if not legal_indices:
        raise RuntimeError(f"No legal non-evidence frame exists for mask_duration={mask_duration}")

    if seed is not None:
        rnd = random.Random(seed)
        shuffled = list(legal_indices)
        rnd.shuffle(shuffled)
        chosen = _deterministic_pick(sorted(shuffled), num_frames)
    else:
        chosen = _deterministic_pick(legal_indices, num_frames)

    seconds = [round(ts[i], 3) for i in chosen]
    assert_no_duration_leak(seconds, mask_duration)
    return chosen, seconds


def sample_tcr_multi_views(vr, num_frames, question, mask_duration=None, all_duration=None, seed=None):
    base_seed = seed if seed is not None else 0
    idx_g, sec_g = sample_tcr_frame_indices(vr, num_frames, question, mask_duration, all_duration, mode="global", seed=base_seed)
    idx_b, sec_b = sample_tcr_frame_indices(vr, num_frames, "boundary", mask_duration, all_duration, mode="boundary", seed=base_seed + 1)
    idx_q, sec_q = sample_tcr_frame_indices(vr, num_frames, question, mask_duration, all_duration, mode="query", seed=base_seed + 2)
    return {
        "global": {"indices": idx_g, "seconds": sec_g},
        "boundary": {"indices": idx_b, "seconds": sec_b},
        "query": {"indices": idx_q, "seconds": sec_q},
    }


def assert_no_duration_leak(seconds, duration):
    mask = parse_duration(duration)
    if mask is None:
        return True
    leaked = [float(s) for s in seconds if is_inside_mask(float(s), mask)]
    if leaked:
        raise RuntimeError(f"Duration leak detected. mask={mask}, leaked_seconds={leaked}")
    return True
