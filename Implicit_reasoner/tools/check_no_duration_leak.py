import argparse
import json
from dataset.tcr_video_sampling import parse_duration, is_inside_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_json', required=True)
    ap.add_argument('--annotation_json', required=True)
    args = ap.parse_args()

    results = json.load(open(args.result_json))
    anns = json.load(open(args.annotation_json))
    ann_map = {str(a.get('question_id', i)): a for i, a in enumerate(anns)}

    violations = 0
    for r in results:
        qid = str(r.get('question_id', ''))
        ann = ann_map.get(qid)
        if ann is None:
            continue
        mask = parse_duration(ann.get('duration'))
        diag = r.get('diagnostics', {})
        views = r.get('frame_seconds_by_view', diag.get('frame_seconds_by_view', {}))
        for _, secs in views.items():
            for t in secs:
                if is_inside_mask(float(t), mask):
                    violations += 1
                    print(f'violation question_id={qid} t={t} mask={mask}')
    print(f'violations: {violations}')
    if violations > 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
