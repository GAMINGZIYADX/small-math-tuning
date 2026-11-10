"""French training data v2: more questions, two translations per question, plus a French dev set.

Changes vs make_fr_data.py (v1 = 3,000 questions, opus-mt only):
  - all 7,473 GSM8K train questions
  - every question translated twice: opus-mt (literal) and Luth-2-2B (fluent, natural French; Apache-2.0)
  - same targets as v1: the base model's own correct ENGLISH solution (self-distillation)
  - also builds mgsm_dev_fr.jsonl: 250 GSM8K *test* questions that are NOT in MGSM, translated by Luth,
    so configs can be chosen without tuning on MGSM-fr itself

Usage: python make_fr_data2.py
Output: fr_train2.jsonl (question_fr, solution, src), mgsm_dev_fr.jsonl (question, answer_number)
Every stage is cached, so a crash or rerun resumes:
  en_solutions.jsonl, fr_translations.jsonl (opus, shared with v1), fr_translations_luth.jsonl, mgsm_dev_fr.jsonl
"""
import argparse, json, os, re, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
from make_fr_data import translation_ok, last_boxed, to_num, SYS

ap = argparse.ArgumentParser()
ap.add_argument("--solver", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--opus", default="Helsinki-NLP/opus-mt-en-fr")
ap.add_argument("--luth", default="kurakurai/Luth-2-2B")
ap.add_argument("--n_dev", type=int, default=250)
a = ap.parse_args()

LUTH_PROMPT = ("Traduis ce problème de mathématiques en français naturel, comme dans un manuel scolaire français. "
               "Garde exactement tous les nombres et écris-les en chiffres (3, pas trois). Ne résous pas le problème. "
               "Réponds uniquement avec la traduction, sans rien ajouter.\n\n{q}")


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def clean_luth(t):
    t = t.strip()
    t = re.sub(r"^(traduction|voici la traduction)[^:\n]*:\s*", "", t, flags=re.I).strip()
    if len(t) > 1 and t[0] in "\"«“" and t[-1] in "\"»”":
        t = t[1:-1].strip()
    return t


FR_NUM = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
          "onze": 11, "douze": 12, "quinze": 15, "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
          "soixante": 60, "cent": 100, "mille": 1000, "un": 1, "une": 1}


def digits_fr(fr):
    """Number words -> digits, only for the number check (the stored translation is unchanged)."""
    return re.sub(r"\b(" + "|".join(FR_NUM) + r")\b", lambda m: str(FR_NUM[m.group(1).lower()]), fr, flags=re.I)


def luth_ok(en, fr):
    if not translation_ok(en, digits_fr(fr)):
        return False
    if "\\boxed" in fr or re.search(r"\b(réponse|solution)\s*:", fr, re.I):
        return False  # Luth solved it instead of translating
    words = re.findall(r"[a-z]+", fr.lower())
    eng = sum(w in {"the", "and", "how", "many", "what", "is", "of", "does", "much"} for w in words)
    return eng <= 1


def generate(tok, model, prompts, max_new, bs, desc):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        outs += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{desc} {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def main():
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)   # same order as v1
    N = len(ds)
    en_q = ds["question"]
    golds = [float(x.split("####")[-1].strip().replace(",", "")) for x in ds["answer"]]

    # ---- stage 1: base model solves every ENGLISH question (reuse v1's correct solutions) ----
    cache = "en_solutions.jsonl"
    sols = {r["idx"]: r for r in load_jsonl(cache)}
    if not sols and os.path.exists("fr_train.jsonl"):
        fr_v1 = [json.loads(l)["fr"] for l in open("fr_translations.jsonl")]
        idx_of = {f: i for i, f in enumerate(fr_v1)}
        with open(cache, "w") as f:
            for r in load_jsonl("fr_train.jsonl"):
                i = idx_of.get(r["question_fr"])
                if i is not None:
                    sols[i] = {"idx": i, "ok": True, "solution": r["solution"]}
                    f.write(json.dumps(sols[i], ensure_ascii=False) + "\n")
        print(f"Seeded {len(sols)} correct English solutions from v1")
    todo = [i for i in range(N) if i not in sols]
    if todo:
        tok = AutoTokenizer.from_pretrained(a.solver, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(a.solver, dtype=torch.bfloat16).cuda().eval()
        with open(cache, "a") as f:
            for s in range(0, len(todo), 384):
                chunk = todo[s:s + 384]
                prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": en_q[i]}],
                                                   tokenize=False, add_generation_prompt=True) for i in chunk]
                texts = generate(tok, model, prompts, 768, 48, f"solve [{s + len(chunk)}/{len(todo)}]")
                for i, t in zip(chunk, texts):
                    p = to_num(last_boxed(t))
                    sols[i] = {"idx": i, "ok": p is not None and abs(p - golds[i]) < 1e-4, "solution": t.strip()}
                    f.write(json.dumps(sols[i], ensure_ascii=False) + "\n")
                f.flush()
        del model
        torch.cuda.empty_cache()
    print(f"English solutions: {sum(r['ok'] for r in sols.values())}/{N} correct")

    # ---- stage 2: opus-mt translations (extends v1's cache, same order) ----
    opus = [json.loads(l)["fr"] for l in open("fr_translations.jsonl")] if os.path.exists("fr_translations.jsonl") else []
    if len(opus) < N:
        mt_tok = MarianTokenizer.from_pretrained(a.opus)
        mt = MarianMTModel.from_pretrained(a.opus).cuda().eval().half()
        with open("fr_translations.jsonl", "a") as f:
            for i in range(len(opus), N, 64):
                enc = mt_tok(en_q[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad():
                    out = mt.generate(**enc, max_new_tokens=512, num_beams=4)
                for t in mt_tok.batch_decode(out, skip_special_tokens=True):
                    opus.append(t)
                    f.write(json.dumps({"fr": t}, ensure_ascii=False) + "\n")
                print(f"opus {len(opus)}/{N}", flush=True)
        del mt
        torch.cuda.empty_cache()

    # ---- stage 3: Luth translations of train questions + dev set ----
    luth = {r["idx"]: r["fr"] for r in load_jsonl("fr_translations_luth.jsonl")}
    mgsm_en = {re.sub(r"\s+", " ", q).strip() for q in load_dataset("juletxara/mgsm", "en", split="test")["question"]}
    test = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0)
    dev = [(q, float(ans.split("####")[-1].strip().replace(",", ""))) for q, ans in zip(test["question"], test["answer"])
           if re.sub(r"\s+", " ", q).strip() not in mgsm_en][:a.n_dev + 50]
    need_train = [i for i in range(N) if sols[i]["ok"] and i not in luth]
    need_dev = not os.path.exists("mgsm_dev_fr.jsonl")
    if need_train or need_dev:
        ltok = AutoTokenizer.from_pretrained(a.luth, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(a.luth, dtype=torch.bfloat16).cuda().eval()
        mk = lambda q: ltok.apply_chat_template([{"role": "user", "content": LUTH_PROMPT.format(q=q)}],
                                                tokenize=False, add_generation_prompt=True)
        if need_dev:
            outs = generate(ltok, lm, [mk(q) for q, _ in dev], 384, 64, "luth dev")
            kept = [(clean_luth(t), g) for (q, g), t in zip(dev, outs) if luth_ok(q, clean_luth(t))][:a.n_dev]
            with open("mgsm_dev_fr.jsonl", "w") as f:
                for q, g in kept:
                    f.write(json.dumps({"question": q, "answer_number": g}, ensure_ascii=False) + "\n")
            print(f"dev set: {len(kept)} French questions in mgsm_dev_fr.jsonl")
        with open("fr_translations_luth.jsonl", "a") as f:
            for s in range(0, len(need_train), 512):
                chunk = need_train[s:s + 512]
                outs = generate(ltok, lm, [mk(en_q[i]) for i in chunk], 384, 64, f"luth train [{s + len(chunk)}/{len(need_train)}]")
                for i, t in zip(chunk, outs):
                    luth[i] = clean_luth(t)
                    f.write(json.dumps({"idx": i, "fr": luth[i]}, ensure_ascii=False) + "\n")
                f.flush()
        del lm
        torch.cuda.empty_cache()

    # ---- stage 4: build training set ----
    n_opus = n_luth = 0
    with open("fr_train2.jsonl", "w") as f:
        for i in range(N):
            if not sols[i]["ok"]:
                continue
            if translation_ok(en_q[i], opus[i]):
                f.write(json.dumps({"question_fr": opus[i], "solution": sols[i]["solution"], "src": "opus"}, ensure_ascii=False) + "\n")
                n_opus += 1
            fr = luth.get(i)
            if fr and fr != opus[i] and luth_ok(en_q[i], fr):
                f.write(json.dumps({"question_fr": fr, "solution": sols[i]["solution"], "src": "luth"}, ensure_ascii=False) + "\n")
                n_luth += 1
    print(f"\nDone: fr_train2.jsonl = {n_opus} opus + {n_luth} Luth = {n_opus + n_luth} examples")


if __name__ == "__main__":
    main()
