import argparse
import json

from dataset.tcr_video_sampling import parse_duration, assert_no_duration_leak


def build_ann_index(anns):
    idx = {}
    for i, a in enumerate(anns):
        qid = str(a.get("question_id", ""))
        vid = str(a.get("video_id", ""))
        if qid:
            idx[(qid, vid)] = a
            idx[(qid, "")] = a
        idx[(str(i), vid)] = a
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--annotation_json", required=True)
    args = ap.parse_args()

    results = json.load(open(args.result_json))
    anns = json.load(open(args.annotation_json))
    ann_idx = build_ann_index(anns)

    violations = 0
    for i, r in enumerate(results):
        views = r.get("frame_seconds_by_view", {})
        if not views:
            continue
        qid = str(r.get("question_id", ""))
        vid = str(r.get("video_id", ""))
        ann = ann_idx.get((qid, vid)) or ann_idx.get((qid, "")) or ann_idx.get((str(i), vid))
        if ann is None:
            continue
        duration = ann.get("duration")
        mask = parse_duration(duration)
        if mask is None:
            continue
        for view, seconds in views.items():
            try:
                assert_no_duration_leak(seconds, duration)
            except Exception as e:
                violations += 1
                print(f"violation question_id={qid} video_id={vid} view={view} duration={mask} seconds={seconds} err={e}")

    print(f"violations: {violations}")
    if violations > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
