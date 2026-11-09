"""Chat with majority voting: each question is solved k times, the most common answer wins.
Usage: python chat_vote.py [model] [--k 8]
Default model: Qwen/Qwen2.5-Math-1.5B-Instruct. Type 'exit' to quit."""
import argparse, collections, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--max_new", type=int, default=2048)
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


def norm(s):
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s.strip())
    for x, y in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), (" ", "")]:
        s = s.replace(x, y)
    s = re.sub(r"^[a-zA-Z]=", "", s.rstrip("."))
    return s


tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
print(f"Loaded {a.model}. Each question is solved {a.k} times (takes ~30-60s).")


def read_question():
    """Multi-line input. Paste/type the question, then press Enter on TWO empty lines to send.
    Single empty lines inside the question are fine."""
    print("\nYou (paste question, then press Enter twice on empty lines to send; 'exit' to quit):")
    lines, blank = [], False
    while True:
        line = input()
        if line.strip().lower() in ("exit", "quit") and not lines:
            return None
        if line.strip() == "":
            if lines and blank:
                return "\n".join(lines)
            blank = True
            continue
        blank = False
        lines.append(line)

while True:
    q = read_question()
    if q is None:
        break
    t0 = time.time()
    prompt = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.7,
                             top_p=0.95, num_return_sequences=a.k)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    answers = [last_boxed(t) for t in texts]
    votes = collections.Counter(norm(x) for x in answers if x is not None)
    if not votes:
        print("\nModel: no final answer in any attempt (try a clearer question).")
        continue
    best, n = votes.most_common(1)[0]
    # show the shortest full solution that reached the winning answer
    sol = min((t for t, x in zip(texts, answers) if x is not None and norm(x) == best), key=len)
    print("\nModel:", sol)
    print(f"\n--- Votes: " + ", ".join(f"{v} x{c}" for v, c in votes.most_common()) +
          f"  | no answer: {answers.count(None)}")
    conf = "high" if n >= 0.75 * a.k else "medium" if n >= 0.5 * a.k else "LOW (double-check this)"
    print(f"--- Final answer: {best}   agreement {n}/{a.k} -> confidence {conf}   ({time.time() - t0:.0f}s)")
