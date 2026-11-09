"""Majority-vote (self-consistency) eval on MATH-500.

Usage:
  python eval_math500_vote.py --model Qwen/Qwen2.5-Math-1.5B-Instruct --n 200 --k 8

Reports, on the same questions:
  single  = average accuracy of one sampled answer
  vote@k  = accuracy of the most common answer among k samples
  pass@k  = % of questions where at least one sample was right (rough ceiling for RL training)
Resumable: results go to vote_<model>_k<k>.jsonl; rerun the same command to continue after a crash.
If you get CUDA out-of-memory, add --qb 4.
"""
import argparse, collections, json, os, re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--n", type=int, default=200, help="number of MATH-500 questions")
ap.add_argument("--k", type=int, default=8, help="samples per question")
ap.add_argument("--qb", type=int, default=8, help="questions per GPU batch (lower if OOM)")
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--max_new", type=int, default=1536)
a = ap.parse_args()

SYS = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------- answer extraction / matching ----------
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


def norm(s):
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for x, y in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\$", ""), ("$", ""), ("\\%", ""), ("%", "")]:
        s = s.replace(x, y)
    s = s.replace(" ", "").rstrip(".")
    s = re.sub(r"^[a-zA-Z]=", "", s)                      # "x=5" -> "5"
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)   # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", s)  # \frac{9}2
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)        # \frac9{19}
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)          # \sqrt2 -> \sqrt{2}
    s = re.sub(r"_(\d+)$", r"_{\1}", s)                   # 4210_5 -> 4210_{5}
    if re.fullmatch(r"-?\d+(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def to_float(s):
    try:
        return float(s)
    except Exception:
        m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s or "")
        if m:
            return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
        return None


def same(p, g):
    p, g = norm(p), norm(g)
    if p is None or g is None:
        return False
    if p == g:
        return True
    fp, fg = to_float(p), to_float(g)
    return fp is not None and fg is not None and abs(fp - fg) < 1e-6


# ---------- resume ----------
name = a.model.strip("/").split("/")[-1]
out_path = f"vote_{name}_k{a.k}.jsonl"
done = {}
if os.path.exists(out_path):
    for line in open(out_path):
        r = json.loads(line)
        done[r["idx"]] = r
    print(f"Resuming: {len(done)} questions already done in {out_path}")


def report(rows, final=False):
    n = len(rows)
    if n == 0:
        return
    single = sum(r["n_correct"] for r in rows) / (n * a.k)
    vote = sum(r["vote_ok"] for r in rows) / n
    pk = sum(r["n_correct"] > 0 for r in rows) / n
    tag = "FINAL " if final else ""
    print(f"{tag}{n}/{a.n}  single: {single:.3f}  vote@{a.k}: {vote:.3f}  pass@{a.k}: {pk:.3f}", flush=True)


ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(a.n))
todo = [i for i in range(a.n) if i not in done]

if todo:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    f = open(out_path, "a")
    for s in range(0, len(todo), a.qb):
        idxs = todo[s:s + a.qb]
        probs = [ds[i]["problem"] for i in idxs]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True) for q in probs]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True,
                                 temperature=a.temp, top_p=0.95, num_return_sequences=a.k)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for qi, i in enumerate(idxs):
            ex = ds[i]
            preds = [last_boxed(t) for t in texts[qi * a.k:(qi + 1) * a.k]]
            oks = [same(p, ex["answer"]) for p in preds]
            votes = collections.Counter(norm(p) for p in preds if p is not None)
            top = votes.most_common(1)[0][0] if votes else None
            r = {"idx": i, "subject": ex["subject"], "level": ex["level"], "gold": ex["answer"],
                 "preds": preds, "n_correct": sum(oks), "vote": top, "vote_ok": same(top, ex["answer"])}
            done[i] = r
            f.write(json.dumps(r) + "\n")
        f.flush()
        report(list(done.values()))
    f.close()

rows = [done[i] for i in range(a.n)]
print()
report(rows, final=True)

# breakdown so we know where training should focus
print("\nBy subject (vote / pass@k):")
by = collections.defaultdict(list)
for r in rows:
    by[r["subject"]].append(r)
for subj, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    v = sum(r["vote_ok"] for r in rs) / len(rs)
    p = sum(r["n_correct"] > 0 for r in rs) / len(rs)
    print(f"  {subj:24s} n={len(rs):3d}  vote {v:.2f}  pass {p:.2f}")
print("\nBy difficulty level (vote / pass@k):")
by = collections.defaultdict(list)
for r in rows:
    by[r["level"]].append(r)
for lvl in sorted(by):
    rs = by[lvl]
    v = sum(r["vote_ok"] for r in rs) / len(rs)
    p = sum(r["n_correct"] > 0 for r in rs) / len(rs)
    print(f"  level {lvl}  n={len(rs):3d}  vote {v:.2f}  pass {p:.2f}")
