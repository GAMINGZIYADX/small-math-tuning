"""French MATH eval: the first 200 MATH-500 problems translated to French (see make_math500_fr.py).
Usage: python eval_math500_fr.py --model out_fr/merged             (French)
       python eval_math500_fr.py --model out_fr/merged --lang en   (same problems in English, for the gap)
Greedy, same grader and system prompt as eval_math500.py. Saves math500fr_<lang>_<model>[_mn<N>].jsonl."""
import argparse, json, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="out_fr/merged")
ap.add_argument("--lang", default="fr", choices=["fr", "en"])
ap.add_argument("--data", default="math500_fr.jsonl")
ap.add_argument("--max_new", type=int, default=2048)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
SYS = "Please reason step by step, and put your final answer within \\boxed{}."


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
    for x, y in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\$", ""), ("$", ""), ("\\%", ""), ("%", "")]:
        s = s.replace(x, y)
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


rows = [json.loads(l) for l in open(a.data)]
tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
name = a.model.strip("/").replace("/", "_") if os.path.isdir(a.model) else a.model.split("/")[-1]
suffix = "" if a.max_new == 2048 else f"_mn{a.max_new}"
f = open(f"math500fr_{a.lang}_{name}{suffix}.jsonl", "w")
correct = cut = 0
for i in range(0, len(rows), a.bs):
    b = rows[i:i + a.bs]
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r[f"problem_{a.lang}"]}],
                                       tokenize=False, add_generation_prompt=True) for r in b]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    for r, t, g in zip(b, texts, gen):
        pred = last_boxed(t)
        ok = same(pred, r["answer"])
        truncated = bool((g != tok.pad_token_id).sum().item() >= a.max_new)
        correct += ok
        cut += truncated
        f.write(json.dumps({"idx": r["idx"], "level": r["level"], "subject": r["subject"], "gold": r["answer"],
                            "pred": pred, "ok": ok, "truncated": truncated, "output": t}, ensure_ascii=False) + "\n")
    f.flush()
    n = min(i + a.bs, len(rows))
    print(f"{n}/{len(rows)}  acc so far: {correct / n:.3f}  truncated: {cut}", flush=True)
print(f"\nFINAL MATH-500-{a.lang} (first {len(rows)}) accuracy ({a.model}, max_new={a.max_new}): "
      f"{correct / len(rows):.3f}  truncated: {cut}/{len(rows)}")
