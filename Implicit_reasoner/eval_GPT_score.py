import argparse
import ast
import json
from tqdm import tqdm
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate open-ended predictions with GPT judge")
    parser.add_argument("--pred_path", default="YYY.json")
    parser.add_argument("--output_json", default="gpt_score.json")
    parser.add_argument("--model", default="gpt-3.5-turbo-1106")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_base", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.pred_path, 'r') as f:
        pred_contents = json.load(f)
    if args.limit:
        pred_contents = pred_contents[:args.limit]

    client = OpenAI(api_key=args.api_key or None, base_url=args.api_base or None)

    response_list = []
    for sample in tqdm(pred_contents):
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "Evaluate whether predicted answer meaningfully matches correct answer. Return only dict with keys pred(yes/no), score(0-5 integer)."},
                {"role": "user", "content": f"Question: {question}\nCorrect Answer: {answer}\nPredicted Answer: {pred}"},
            ],
        )
        response_message = ast.literal_eval(completion.choices[0].message.content.strip())
        response_message['question'] = question
        response_message['YN'] = response_message['pred']
        response_message['pred'] = pred
        response_message['answer'] = answer
        response_message['question_id'] = sample['question_id']
        response_list.append(response_message)

    with open(args.output_json, 'w') as f:
        json.dump(response_list, f, indent=2)

    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for result in response_list:
        count += 1
        score_sum += int(result['score'])
        if str(result.get('YN', '')).lower() == 'yes':
            yes_count += 1
        else:
            no_count += 1

    average_score = score_sum / max(count, 1)
    accuracy = yes_count / max(yes_count + no_count, 1)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()
