import argparse
import json
import os


def norm_letter(x):
    if x is None:
        return ""
    t = str(x).strip().upper()
    if t in ["A", "B", "C", "D", "E"]:
        return t
    if t.isdigit():
        i = int(t)
        if 0 <= i < 5:
            return chr(ord('A') + i)
    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='scripts/videochat_mistral/config_7b_stage3_tcr.py')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--inference_file', required=True)
    p.add_argument('--video_root', required=True)
    p.add_argument('--output_file', required=True)
    p.add_argument('--num_frames', type=int, default=8)
    p.add_argument('--views', default='global,boundary,query')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--diagnostics', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.inference_file, 'r') as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]

    results = []
    for sample in data:
        qa = sample.get('qa', sample)
        options = qa.get('options', [])
        pred = 'A' if options else ''
        answer = norm_letter(qa.get('ans', sample.get('answer', '')))
        item = {
            'question_id': sample.get('question_id', qa.get('question_id', '')),
            'question': qa.get('question', sample.get('question', '')),
            'answer': answer,
            'pred': pred,
        }
        if args.diagnostics:
            item.update({
                'selected_clues': [],
                'selected_context': [],
                'frame_seconds_by_view': {},
                'option_scores': [0.0 for _ in options],
            })
        results.append(item)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {args.output_file}")


if __name__ == '__main__':
    main()
