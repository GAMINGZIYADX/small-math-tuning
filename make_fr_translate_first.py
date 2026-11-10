"""Translate-first French data: French question -> faithful English restatement + the model's own English solution.

Why: on MGSM-fr, 77 of our 106 misses are questions the model solves in English. It misreads the French
while restating the problem (e.g. "3 fois par semaine" -> "6 days a week"). Training the restatement
step explicitly, with the original GSM8K English as the target, attacks exactly that.

Usage: python make_fr_translate_first.py          (needs fr_train.jsonl and fr_translations.jsonl from make_fr_data.py)
Output: fr_train_tf.jsonl (fields: question_fr, solution)  ->  python train_sft_fr.py --data fr_train_tf.jsonl --out out_fr_tf
"""
import argparse, json
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000, help="must match the --n used for make_fr_data.py")
a = ap.parse_args()

ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(a.n))
fr = [json.loads(l)["fr"] for l in open("fr_translations.jsonl")][:a.n]
en_of = {f: e for f, e in zip(fr, ds["question"])}

rows = [json.loads(l) for l in open("fr_train.jsonl")]
n = 0
with open("fr_train_tf.jsonl", "w") as f:
    for r in rows:
        en = en_of.get(r["question_fr"])
        if en is None:
            continue
        target = f"Question in English: {en.strip()}\n\n{r['solution']}"
        f.write(json.dumps({"question_fr": r["question_fr"], "solution": target}, ensure_ascii=False) + "\n")
        n += 1
print(f"{n}/{len(rows)} translate-first examples in fr_train_tf.jsonl")
