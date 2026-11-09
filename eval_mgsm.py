"""MGSM eval: the same 250 grade-school problems in several languages.
Usage: python eval_mgsm.py --lang fr        (and --lang en to compare)
       python eval_mgsm.py --lang fr --model out_grpo/merged
Greedy decoding. Saves outputs to mgsm_<lang>_<model>.jsonl."""
import argparse, json, re, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--lang", default="fr")
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
SYS = "Please reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def to_num(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\[a-z]+|[$€%{}\s]", "", s)
    s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s) else s.replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
ds = load_dataset("juletxara/mgsm", a.lang, split="test")
name = a.model.strip("/").split("/")[-1]
f = open(f"mgsm_{a.lang}_{name}.jsonl", "w")
correct = 0
for i in range(0, len(ds), a.bs):
    b = ds[i:i + a.bs]
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True) for q in b["question"]]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=1024, do_sample=False)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for q, t, gold in zip(b["question"], texts, b["answer_number"]):
        p = to_num(last_boxed(t))
        ok = p is not None and abs(p - float(gold)) < 1e-4
        correct += ok
        f.write(json.dumps({"question": q, "gold": gold, "pred": p, "ok": ok, "output": t}, ensure_ascii=False) + "\n")
    n = min(i + a.bs, len(ds))
    print(f"{n}/{len(ds)}  acc so far: {correct / n:.3f}", flush=True)
print(f"\nFINAL MGSM-{a.lang} accuracy ({a.model}): {correct / len(ds):.3f}")
