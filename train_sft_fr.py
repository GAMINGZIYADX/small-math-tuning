"""LoRA SFT: French question -> the model's own correct English solution.
Usage: python train_sft_fr.py            (reads fr_train.jsonl, writes out_fr/merged)
Then:  python eval_mgsm.py --lang fr --model out_fr/merged
       python eval_mgsm.py --lang en --model out_fr/merged   (check English didn't get worse)
If CUDA out-of-memory: --bs 2 --accum 8
"""
import argparse, json, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--data", default="fr_train.jsonl")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--r", type=int, default=32)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--maxlen", type=int, default=1024)
ap.add_argument("--out", default="out_fr")
a = ap.parse_args()
SYS = "Please reason step by step, and put your final answer within \\boxed{}."

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(
    r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

rows = [json.loads(l) for l in open(a.data)]
print(f"{len(rows)} training examples")


def encode(ex):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": ex["question_fr"]}],
        tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(ex["solution"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + r)[:a.maxlen]
    labels = ([-100] * len(p) + r)[:a.maxlen]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


ds = Dataset.from_list(rows).shuffle(seed=0)
ds = ds.map(encode, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir=f"{a.out}/ckpt", per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
    num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=10,
    logging_steps=10, save_strategy="no", bf16=True, dataloader_num_workers=0, report_to="none")
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()

model.save_pretrained(f"{a.out}/adapter")
merged = model.merge_and_unload()
merged.save_pretrained(f"{a.out}/merged")
tok.save_pretrained(f"{a.out}/merged")
print(f"Done. Evaluate: python eval_mgsm.py --lang fr --model {a.out}/merged")
