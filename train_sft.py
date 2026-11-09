"""Incremental LoRA SFT on MetaMathQA.
  python train_sft.py --n 3000                      # first chunk (fresh adapter)
  python train_sft.py --n 3000 --start 3000 --resume out/adapter   # next chunk, continues
Each run saves out/adapter and out/merged. Then: python eval_gsm8k.py out/merged 200
"""
import argparse, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--n", type=int, default=3000, help="examples in this chunk")
ap.add_argument("--start", type=int, default=0, help="offset into the shuffled dataset")
ap.add_argument("--resume", default=None, help="path to an existing adapter to continue from")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--maxlen", type=int, default=512)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
a = ap.parse_args()

OUT = "out"
SYS = "Please reason step by step, and put your final answer after 'The answer is:'."
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float32

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).cuda()
if a.resume:
    model = PeftModel.from_pretrained(model, a.resume, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=0)
ds = ds.select(range(a.start, a.start + a.n))

def encode(ex):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": ex["query"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(ex["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:a.maxlen]
    labels = ([-100] * len(p_ids) + r_ids)[:a.maxlen]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

ds = ds.map(encode, remove_columns=ds.column_names, num_proc=1)
steps = max(1, a.n // (a.bs * a.accum))

args = TrainingArguments(
    output_dir=f"{OUT}/ckpt", per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
    num_train_epochs=1, learning_rate=a.lr, lr_scheduler_type="cosine",
    warmup_steps=max(1, steps // 10), logging_steps=5, save_strategy="no",
    bf16=use_bf16, fp16=not use_bf16, dataloader_num_workers=0, report_to="none")

Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()

model.save_pretrained(f"{OUT}/adapter")
merged = model.merge_and_unload()
merged.save_pretrained(f"{OUT}/merged")
tok.save_pretrained(f"{OUT}/merged")
print(f"Done. Next chunk: python train_sft.py --n {a.n} --start {a.start + a.n} --resume {OUT}/adapter")
