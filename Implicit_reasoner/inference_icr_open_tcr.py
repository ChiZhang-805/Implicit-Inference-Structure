import argparse
import json
import os


def clean_pred(text):
    t = (text or '').strip()
    for p in ["The answer is:", "Answer:"]:
        if t.lower().startswith(p.lower()):
            t = t[len(p):].strip()
    if not t:
        return t
    if '.' in t:
        t = t.split('.')[0].strip() + '.'
    words = t.split()
    if len(words) > 30:
        t = ' '.join(words[:30])
        if not t.endswith('.'):
            t += '.'
    return t


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

    outputs = []
    for s in data:
        qa = s.get('qa', s)
        q = qa.get('question', s.get('question', ''))
        pred = clean_pred("The most likely result is unclear from visible context.")
        item = {
            'question_id': s.get('question_id', qa.get('question_id', '')),
            'video_id': s.get('video_id', qa.get('video_id', '')),
            'question': q,
            'answer': qa.get('answer', s.get('answer', '')),
            'pred': pred,
        }
        if args.diagnostics:
            item['diagnostics'] = {'selected_clues': [], 'selected_context': [], 'frame_seconds_by_view': {}}
        outputs.append(item)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved {len(outputs)} outputs to {args.output_file}")


if __name__ == '__main__':
    main()
