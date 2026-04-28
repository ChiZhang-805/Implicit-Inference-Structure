import argparse
import json


def norm(x):
    if x is None:
        return ""
    t = str(x).strip().upper()
    if t in ["A", "B", "C", "D", "E"]:
        return t
    if t.isdigit():
        i = int(t)
        if 0 <= i < 5:
            return chr(ord("A") + i)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True)
    args = ap.parse_args()
    data = json.load(open(args.pred_path))
    total, correct, invalid = len(data), 0, 0
    for item in data:
        p = norm(item.get("pred"))
        a = norm(item.get("answer"))
        if not p:
            invalid += 1
        if p and a and p == a:
            correct += 1
    acc = correct / total if total else 0.0
    print(f"total: {total}")
    print(f"correct: {correct}")
    print(f"invalid: {invalid}")
    print(f"accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
