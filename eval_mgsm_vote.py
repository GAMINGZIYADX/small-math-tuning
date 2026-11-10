"""MGSM with majority voting: k sampled answers per question, the most common answer wins.
Usage: python eval_mgsm_vote.py --lang fr --model out_fr2/merged --k 8
       python eval_mgsm_vote.py --data mgsm_dev_fr.jsonl --model out_fr2/merged --k 8
Reports: mean single-sample accuracy, majority vote@k, pass@k. Saves <tag>_<model>_k<k>.jsonl (flushed per batch; rerun resumes).
Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid spilling past 16 GB."""
import argparse, collections, json, os, re, torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lang", default="fr")
ap.add_argument("--data", default=None)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--prompts_per_batch", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
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


def to_num(s):  # identical to eval_mgsm.py
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\[a-z]+|[$€%{}\s]", "", s)
    s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s) else s.replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
ds = Dataset.from_list([json.loads(l) for l in open(a.data)]) if a.data else load_dataset("juletxara/mgsm", a.lang, split="test")
name = a.model.strip("/").replace("/", "_") if os.path.isdir(a.model) else a.model.split("/")[-1]
tag = os.path.basename(a.data).removesuffix(".jsonl") if a.data else f"mgsm_{a.lang}"
out_path = f"{tag}_{name}_k{a.k}.jsonl"
done = [json.loads(l) for l in open(out_path)] if os.path.exists(out_path) else []   # resume
single = sum(r["n_ok"] for r in done) / a.k
vote = sum(r["vote_ok"] for r in done)
passk = sum(r["n_ok"] > 0 for r in done)
if done:
    print(f"Resuming after {len(done)} finished questions", flush=True)
f = open(out_path, "a")
for i in range(len(done), len(ds), a.prompts_per_batch):
    b = ds[i:i + a.prompts_per_batch]
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True) for q in b["question"]]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=a.temp, top_p=a.top_p,
                             num_return_sequences=a.k)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for j, (q, gold) in enumerate(zip(b["question"], b["answer_number"])):
        preds = [to_num(last_boxed(t)) for t in texts[j * a.k:(j + 1) * a.k]]
        oks = [p is not None and abs(p - float(gold)) < 1e-4 for p in preds]
        counts = collections.Counter(p for p in preds if p is not None)
        v = counts.most_common(1)[0][0] if counts else None
        v_ok = v is not None and abs(v - float(gold)) < 1e-4
        single += sum(oks) / a.k
        vote += v_ok
        passk += any(oks)
        f.write(json.dumps({"question": q, "gold": gold, "preds": preds, "vote": v, "vote_ok": v_ok,
                            "n_ok": sum(oks)}, ensure_ascii=False) + "\n")
    f.flush()
    del out
    torch.cuda.empty_cache()   # long sampled answers fragment the cache; on 16 GB that spills to shared memory
    n = min(i + a.prompts_per_batch, len(ds))
    print(f"{n}/{len(ds)}  single {single / n:.3f}  vote@{a.k} {vote / n:.3f}  pass@{a.k} {passk / n:.3f}", flush=True)
N = len(ds)
print(f"\nFINAL {tag} ({a.model}, k={a.k}, T={a.temp}): single {single / N:.3f}  vote@{a.k} {vote / N:.3f}  pass@{a.k} {passk / N:.3f}")
