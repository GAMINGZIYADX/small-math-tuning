"""MATH-500 eval (harder than GSM8K). Usage: python eval_math500.py <model> [n_questions]
Saves every question/output to math500_<name>.jsonl so we can see what it gets wrong."""
import json, re, sys, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SYS = "Please reason step by step, and put your final answer within \\boxed{}."

tok = AutoTokenizer.from_pretrained(path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(N))

def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0: return None
    j = text.find("{", i)
    if j < 0: return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{": depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0: return text[j + 1:k]
    return None

def norm(s):
    if s is None: return None
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\$", ""), ("$", ""), ("\\%", ""), ("%", "")]:
        s = s.replace(a, b)
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    if re.fullmatch(r"-?\d+(,\d{3})+(\.\d+)?", s): s = s.replace(",", "")
    return s

def same(p, g):
    p, g = norm(p), norm(g)
    if p is None or g is None: return False
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-6
    except Exception: return False

name = path.strip("/").split("/")[-1]
out_f = open(f"math500_{name}.jsonl", "w")
correct, BS = 0, 32
for i in range(0, N, BS):
    b = ds[i:i + BS]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True) for q in b["problem"]]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=1536, do_sample=False)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for q, t, gold in zip(b["problem"], texts, b["answer"]):
        pred = last_boxed(t)
        ok = same(pred, gold)
        correct += int(ok)
        out_f.write(json.dumps({"problem": q, "gold": gold, "pred": pred, "ok": ok, "output": t}) + "\n")
    out_f.flush()
    print(f"{min(i + BS, N)}/{N}  acc so far: {correct / min(i + BS, N):.3f}", flush=True)

print(f"\nFINAL MATH-500 accuracy ({path}): {correct / N:.3f}")
