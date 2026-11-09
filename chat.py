"""Chat with a model. Usage: python chat.py <model_path_or_hf_id>
Example: python chat.py out/merged   or   python chat.py Qwen/Qwen2.5-Math-1.5B-Instruct
Type 'exit' to quit."""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = sys.argv[1]
SYS = "Please reason step by step, and put your final answer within \\boxed{}."
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()

while True:
    q = input("\nYou: ").strip()
    if q.lower() in ("exit", "quit", ""):
        break
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=2048, do_sample=False)
    print("\nModel:", tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
