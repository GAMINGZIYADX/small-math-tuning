"""Greedy GSM8K eval. Usage: python eval_gsm8k.py <model_path_or_hf_id> [n_questions]
Run on the base model first, then on out/merged, and compare."""
import re, sys, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SYS = "Please reason step by step, and put your final answer after 'The answer is:'."

tok = AutoTokenizer.from_pretrained(path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(N))

def to_num(s):
    try: return float(s.replace(",", "").replace("$", "").rstrip("."))
    except Exception: return None

def extract(text):
    m = re.search(r"answer is:?\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m: m = re.search(r"\\boxed\{\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        return to_num(nums[-1]) if nums else None
    return to_num(m.group(1))

correct, BS = 0, 32
for i in range(0, N, BS):
    batch = ds[i:i + BS]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True) for q in batch["question"]]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=False)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for t, gold in zip(texts, batch["answer"]):
        g = to_num(gold.split("####")[-1].strip())
        p = extract(t)
        correct += int(p is not None and g is not None and abs(p - g) < 1e-4)
    print(f"{min(i + BS, N)}/{N}  acc so far: {correct / min(i + BS, N):.3f}", flush=True)

print(f"\nFINAL GSM8K accuracy ({path}): {correct / N:.3f}")
