"""Build French training data: GSM8K *train* questions translated to French, paired with
Qwen-Math's own correct English solution (generated from the original English question).

Usage: python make_fr_data.py --n 3000
Output: fr_train.jsonl  (fields: question_fr, solution)
Needs: pip install sentencepiece sacremoses
Stages are cached (fr_translations.jsonl), so rerunning after a crash skips finished translation.
"""
import argparse, json, os, re, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000, help="how many GSM8K train questions to use")
ap.add_argument("--solver", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--translator", default="Helsinki-NLP/opus-mt-en-fr")
ap.add_argument("--bs", type=int, default=48)
SYS = "Please reason step by step, and put your final answer within \\boxed{}."


def numbers_en(s):
    return set(re.findall(r"\d+", s.replace(",", "")))


def numbers_fr(s):
    s = re.sub(r"(?<=\d)[\s\u202f\u00a0](?=\d{3}\b)", "", s)  # "1 000" -> "1000"
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s)                 # "1.000" -> "1000"
    return set(re.findall(r"\d+", s))


def translation_ok(en, fr):
    """Reject translations that dropped or changed a number, or look broken."""
    return numbers_en(en) <= numbers_fr(fr) and 0.5 < len(fr) / max(1, len(en)) < 2.5


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
    s = re.sub(r"\\text\{[^}]*\}|\\[a-z]+|[$%{}\s]", "", s).replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def main():
    a = ap.parse_args()
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(a.n))
    golds = [float(x.split("####")[-1].strip().replace(",", "")) for x in ds["answer"]]

    # ---- stage 1: translate questions ----
    cache = "fr_translations.jsonl"
    if os.path.exists(cache):
        fr = [json.loads(l)["fr"] for l in open(cache)]
        print(f"Loaded {len(fr)} cached translations")
    else:
        fr = []
    if len(fr) < a.n:
        mt_tok = MarianTokenizer.from_pretrained(a.translator)
        mt = MarianMTModel.from_pretrained(a.translator).cuda().eval().half()
        with open(cache, "a") as f:
            for i in range(len(fr), a.n, 64):
                batch = ds["question"][i:i + 64]
                enc = mt_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad():
                    out = mt.generate(**enc, max_new_tokens=512, num_beams=4)
                for t in mt_tok.batch_decode(out, skip_special_tokens=True):
                    fr.append(t)
                    f.write(json.dumps({"fr": t}, ensure_ascii=False) + "\n")
                print(f"translated {len(fr)}/{a.n}", flush=True)
        del mt
        torch.cuda.empty_cache()

    keep = [i for i in range(a.n) if translation_ok(ds["question"][i], fr[i])]
    print(f"Translations passing number check: {len(keep)}/{a.n}")

    # ---- stage 2: model solves the ENGLISH question; keep correct solutions ----
    tok = AutoTokenizer.from_pretrained(a.solver, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.solver, dtype=torch.bfloat16).cuda().eval()
    n_ok = 0
    with open("fr_train.jsonl", "w") as f:
        for s in range(0, len(keep), a.bs):
            idx = keep[s:s + a.bs]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYS}, {"role": "user", "content": ds["question"][i]}],
                tokenize=False, add_generation_prompt=True) for i in idx]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=768, do_sample=False)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                p = to_num(last_boxed(t))
                if p is not None and abs(p - golds[i]) < 1e-4:
                    n_ok += 1
                    f.write(json.dumps({"question_fr": fr[i], "solution": t.strip()}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"solved {min(s + a.bs, len(keep))}/{len(keep)}  kept {n_ok}", flush=True)
    print(f"\nDone: {n_ok} French training examples in fr_train.jsonl")


if __name__ == "__main__":
    main()
