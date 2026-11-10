"""Translate MATH-500 problems to French with LaTeX protected by placeholders.

Usage: python make_math500_fr.py --n 200
Output: math500_fr.jsonl (fields: idx, unique_id, subject, level, answer, problem_en, problem_fr)

All LaTeX is masked before translation (opus-mt-en-fr), the prose is translated line by line, and the
LaTeX is put back. A problem is kept only if
  - every placeholder comes back exactly once, so every math span is restored verbatim,
  - every number in the English prose (outside math) survives in the French prose,
  - the French has the same counts of $, \\frac, \\[, braces and newlines as the English,
  - the prose length ratio is sane.
Rejected problems are listed in math500_fr_rejected.jsonl.
"""
import argparse, json, re, torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--translator", default="Helsinki-NLP/opus-mt-en-fr")
ap.add_argument("--device", default="cuda")
ap.add_argument("--style", default="bracket", choices=["bracket", "eq"], help="placeholder style")

# Order matters: big blocks first, then display math, then inline math, then leftovers.
PATTERNS = [
    r"\\\$",                                   # escaped dollar FIRST: "$\$$1500" contains "$$"
    r"\[asy\].*?\[/asy\]",
    r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}",
    r"\$\$.*?\$\$",
    r"\\\[.*?\\\]",
    r"\\\(.*?\\\)",
    r"(?<!\\)\$.+?(?<!\\)\$",                  # inline $...$
    r"\\[a-zA-Z]+\*?(\{[^{}]*\})*",            # stray commands in prose, e.g. \textbf{...}
]
STYLE = "bracket"
PH = None


def set_style(style):
    global STYLE, PH
    STYLE = style
    PH = re.compile(r"\[(\d+)\]") if style == "bracket" else re.compile(r"EQ([A-Z]+)")


def ph_name(i):
    if STYLE == "bracket":
        return f"[{i}]"
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return "EQ" + s


def mask(text):
    spans = []
    for pat in PATTERNS:
        def sub(m):
            spans.append(m.group(0))
            return " " + ph_name(len(spans) - 1) + " "   # spaces: MATH glues text to math, e.g. "\\]is"
        text = re.sub(pat, sub, text, flags=re.S)
    # Placeholders nested inside a span must not keep their padding, or the LaTeX gains spaces.
    spans = [re.sub(" (" + PH.pattern + ") ", r"\1", sp) for sp in spans]
    return text, spans


def unmask(text, spans):
    # Placeholders can be nested (e.g. a \$ inside inline math), so restore until stable.
    for _ in range(5):
        new = PH.sub(lambda m: spans[name_idx(m.group(0))] if name_idx(m.group(0)) < len(spans) else m.group(0), text)
        if new == text:
            break
        text = new
    return text


def name_idx(name):
    if STYLE == "bracket":
        return int(name[1:-1])
    n = 0
    for c in name[2:]:
        n = n * 26 + (ord(c) - 64)
    return n - 1


def tidy(masked, spans):
    """Runs on the MASKED French (math still hidden, so LaTeX is never touched): undo the padding
    spaces and drop French elision before math ("l'$a$" -> "$a$")."""
    def elide(m):
        i = name_idx(m.group(2))
        sp = spans[i] if i < len(spans) else ""
        is_math = sp.startswith(("$", "\\[", "\\(")) and not sp.startswith("\\$")
        return m.group(2) if is_math else m.group(0)
    masked = re.sub(r"\b([lL]['’]\s*)(" + PH.pattern + ")", elide, masked)
    masked = re.sub(r"[ \t]{2,}", " ", masked)
    masked = re.sub(r"[ \t]+\n|\n[ \t]+", "\n", masked)
    masked = re.sub(r" ([.,)])", r"\1", masked)
    return masked.strip()


def numbers_en(s):
    return set(re.findall(r"\d+", s.replace(",", "")))


def numbers_fr(s):
    s = re.sub(r"(?<=\d)[\s\u202f\u00a0](?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s)
    return set(re.findall(r"\d+", s))


PROSE_IN_MATH = re.compile(r"\b[a-zA-Z]{3,}(?:[ ,']+[a-zA-Z]{2,}){3,}")


def latex_signature(s):
    return {k: s.count(k) for k in ["$", "\\frac", "\\[", "\\]", "{", "}", "\\begin", "[asy]", "\n"]}


def check(en, fr, masked_en, masked_fr, spans):
    top = [m.group(0) for m in PH.finditer(masked_en)]
    for name in top:
        if len(re.findall(re.escape(name) + (r"(?![A-Z])" if STYLE == "eq" else ""), masked_fr)) != 1:
            return f"placeholder {name} not restored exactly once"
    if {m.group(0) for m in PH.finditer(masked_fr)} - set(top):
        return "invented placeholder"
    for sp in spans:
        if sp not in fr and not PH.search(sp):
            return f"math span lost: {sp[:40]}"
    for sp in spans:
        if sp.startswith("$") and PROSE_IN_MATH.search(re.sub(r"\\text\{[^}]*\}|\\[a-zA-Z]+", " ", sp)):
            return f"English prose inside a math span: {sp[:60]}"
    if latex_signature(en) != latex_signature(fr):
        return f"latex signature differs: {latex_signature(en)} vs {latex_signature(fr)}"
    prose_en, prose_fr = PH.sub("", masked_en), PH.sub("", masked_fr)
    if not numbers_en(prose_en) <= numbers_fr(prose_fr):
        return f"number lost: {numbers_en(prose_en) - numbers_fr(prose_fr)}"
    if len(prose_en.strip()) > 20 and not 0.6 < len(prose_fr) / len(prose_en) < 2.5:
        return f"length ratio {len(prose_fr) / len(prose_en):.2f}"
    return None


def main():
    a = ap.parse_args()
    set_style(a.style)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(a.n))
    tok = MarianTokenizer.from_pretrained(a.translator)
    mt = MarianMTModel.from_pretrained(a.translator).to(a.device).eval()

    # Collect every line that has prose to translate, translate them all in batches.
    masked, spans_all, todo = [], [], []
    for i, p in enumerate(ds["problem"]):
        m, spans = mask(p)
        masked.append(m)
        spans_all.append(spans)
        for j, line in enumerate(m.split("\n")):
            if re.search(r"[A-Za-z]{2,}", PH.sub("", line)):
                todo.append((i, j, line))
    print(f"{len(todo)} prose lines to translate")
    out_lines = {}
    for s in range(0, len(todo), 32):
        batch = todo[s:s + 32]
        enc = tok([x[2] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(a.device)
        with torch.no_grad():
            out = mt.generate(**enc, num_beams=4, max_new_tokens=400)
        for (i, j, _), t in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
            out_lines[(i, j)] = t
        print(f"translated {min(s + 32, len(todo))}/{len(todo)}", flush=True)

    kept = rejected = 0
    with open("math500_fr.jsonl", "w") as f, open("math500_fr_rejected.jsonl", "w") as fr_rej:
        for i, ex in enumerate(ds):
            lines = masked[i].split("\n")
            masked_fr = "\n".join(out_lines.get((i, j), line) for j, line in enumerate(lines))
            masked_fr = tidy(masked_fr, spans_all[i])
            fr = unmask(masked_fr, spans_all[i])
            why = check(ex["problem"], fr, masked[i], masked_fr, spans_all[i])
            row = {"idx": i, "unique_id": ex["unique_id"], "subject": ex["subject"], "level": ex["level"],
                   "answer": ex["answer"], "problem_en": ex["problem"], "problem_fr": fr}
            if why:
                rejected += 1
                fr_rej.write(json.dumps({**row, "reason": why}, ensure_ascii=False) + "\n")
            else:
                kept += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nKept {kept}/{a.n} translations in math500_fr.jsonl, rejected {rejected} (math500_fr_rejected.jsonl)")


if __name__ == "__main__":
    main()
