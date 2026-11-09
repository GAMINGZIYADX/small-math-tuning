"""GRPO (RL) fine-tuning with LoRA for math.
The model samples G answers per problem; correct answers get reward 1, wrong get 0.
Answers better than their group's average are reinforced, worse ones are pushed down.

Train on MATH train split (no overlap with MATH-500), levels 4-5 by default.

  python grpo.py --steps 50                                   # first chunk (~1 min/step)
  python grpo.py --steps 50 --start 200 --resume out_grpo/adapter   # continue (command printed at end)

Then evaluate (delete the old vote file first, or it will "resume" stale results):
  rm -f vote_merged_k8.jsonl
  python eval_math500_vote.py --model out_grpo/merged --n 200 --k 8
"""
import argparse, json, os, re, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--p", type=int, default=4, help="problems per step")
ap.add_argument("--g", type=int, default=8, help="samples per problem (group size)")
ap.add_argument("--start", type=int, default=0, help="offset into the shuffled problem pool")
ap.add_argument("--resume", default=None, help="adapter dir to continue from")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--r", type=int, default=32, help="LoRA rank")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--mb", type=int, default=2, help="sequences per backward pass (lower if OOM)")
ap.add_argument("--levels", default="4,5")
ap.add_argument("--subjects", default="all")
ap.add_argument("--save_every", type=int, default=10)
ap.add_argument("--out", default="out_grpo")
SYS = "Please reason step by step, and put your final answer within \\boxed{}."


# ---------- answer extraction / matching (same as eval_math500_vote.py) ----------
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
    s = re.sub(r"^[a-zA-Z]=", "", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)
    s = re.sub(r"_(\d+)$", r"_{\1}", s)
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


# ---------- core GRPO pieces ----------
def completion_mask(comp, eos_t):
    """1 for completion tokens up to and including the first EOS, 0 after."""
    is_eos = torch.isin(comp, eos_t).long()
    return ((is_eos.cumsum(1) - is_eos) == 0).long()


def group_advantages(rewards, g):
    r = rewards.view(-1, g)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    return adv.view(-1)


def policy_backward(model, out, P, cmask, prompt_attn, adv, mb, max_new):
    """Policy-gradient loss, backward in micro-batches. Returns number of sequences used."""
    attn = torch.cat([prompt_attn, cmask], 1)
    pos = (attn.cumsum(1) - 1).clamp(min=0)
    keep = (adv != 0).nonzero().view(-1).tolist()
    for j in range(0, len(keep), mb):
        idx = keep[j:j + mb]
        L = int(cmask[idx].sum(1).max())
        ids, am, ps = out[idx, :P + L], attn[idx, :P + L], pos[idx, :P + L]
        logits = model(input_ids=ids, attention_mask=am, position_ids=ps,
                       use_cache=False, logits_to_keep=L + 1).logits[:, :-1]
        tgt = ids[:, P:P + L]
        lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = cmask[idx, :L].float()
        loss = -((lp * m).sum(1) / max_new * adv[idx]).sum() / len(keep)
        loss.backward()
    return len(keep)


def main():
    a = ap.parse_args()
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel

    SUBS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
    subs = SUBS if a.subjects == "all" else a.subjects.split(",")
    levels = {f"Level {x.strip()}" for x in a.levels.split(",")}
    ds = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in subs])
    ds = ds.filter(lambda e: e["level"] in levels)
    ds = ds.map(lambda e: {"gold": last_boxed(e["solution"])}).filter(lambda e: e["gold"] is not None)
    ds = ds.shuffle(seed=0)
    print(f"Problem pool: {len(ds)} (levels {sorted(levels)}), starting at {a.start}")

    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.resume:
        model = PeftModel.from_pretrained(base, a.resume, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    eos = base.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    eos_t = torch.tensor(sorted(set(eos + [tok.eos_token_id, tok.pad_token_id]) - {None}), device="cuda")

    os.makedirs(a.out, exist_ok=True)
    log = open(f"{a.out}/log.jsonl", "a")
    step = 0
    for step in range(a.steps):
        i0 = a.start + step * a.p
        if i0 >= len(ds):
            print("Ran out of problems.")
            break
        t0 = time.time()
        batch = ds[i0:i0 + a.p]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True) for q in batch["problem"]]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        P = enc["input_ids"].shape[1]

        model.eval()
        with torch.no_grad():  # top_k=0 / rep_penalty=1 keep sampling faithful to the model's own probabilities
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=a.temp,
                                 top_p=1.0, top_k=0, repetition_penalty=1.0,
                                 num_return_sequences=a.g, pad_token_id=tok.pad_token_id)
        comp = out[:, P:]
        cmask = completion_mask(comp, eos_t)
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        golds = [gd for gd in batch["gold"] for _ in range(a.g)]
        rewards = torch.tensor([1.0 if same(last_boxed(t), gd) else 0.0 for t, gd in zip(texts, golds)],
                               device="cuda")
        adv = group_advantages(rewards, a.g)
        t_gen = time.time() - t0

        model.train()
        opt.zero_grad(set_to_none=True)
        used = policy_backward(model, out, P, cmask,
                               enc["attention_mask"].repeat_interleave(a.g, 0), adv, a.mb, a.max_new)
        if used:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        groups = rewards.view(-1, a.g)
        stats = {"step": step + 1, "reward": round(rewards.mean().item(), 3),
                 "groups_learning": int(((groups.sum(1) > 0) & (groups.sum(1) < a.g)).sum()),
                 "avg_len": int(cmask.sum(1).float().mean()), "truncated": int((cmask.sum(1) >= a.max_new).sum()),
                 "gen_s": round(t_gen), "total_s": round(time.time() - t0)}
        log.write(json.dumps(stats) + "\n"); log.flush()
        print(f"step {stats['step']:3d}/{a.steps}  reward {stats['reward']:.2f}  "
              f"learning-groups {stats['groups_learning']}/{a.p}  len {stats['avg_len']}  "
              f"trunc {stats['truncated']}  {stats['total_s']}s", flush=True)
        if (step + 1) % a.save_every == 0:
            model.save_pretrained(f"{a.out}/adapter")

    model.save_pretrained(f"{a.out}/adapter")
    print("Merging and saving full model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{a.out}/merged")
    tok.save_pretrained(f"{a.out}/merged")
    nxt = a.start + (step + 1) * a.p
    print(f"\nDone. Continue with:\n  python grpo.py --steps {a.steps} --start {nxt} --resume {a.out}/adapter"
          f"\nEvaluate with:\n  rm -f vote_merged_k8.jsonl && python eval_math500_vote.py --model {a.out}/merged --n 200 --k 8")


if __name__ == "__main__":
    main()
