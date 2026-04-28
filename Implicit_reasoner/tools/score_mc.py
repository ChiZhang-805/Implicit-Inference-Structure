import argparse
import json


def norm(x):
    if x is None:
        return ''
    t = str(x).strip().upper()
    if t in ['A','B','C','D','E']:
        return t
    if t.isdigit():
        i = int(t)
        if 0 <= i < 5:
            return chr(ord('A') + i)
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_path', required=True)
    args = ap.parse_args()
    data = json.load(open(args.pred_path))
    total = len(data)
    correct = 0
    invalid = 0
    for x in data:
        p = norm(x.get('pred'))
        a = norm(x.get('answer'))
        if not p:
            invalid += 1
        if p and a and p == a:
            correct += 1
    print(f'count: {total}')
    print(f'correct: {correct}')
    print(f'accuracy: {correct / max(total,1):.4f}')
    print(f'invalid_prediction_count: {invalid}')


if __name__ == '__main__':
    main()
