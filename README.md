# Small-model math tuning on a consumer GPU

Experiments in improving small open LLMs (0.5B–1.5B) at math reasoning, on one RTX 5060 Ti (16 GB) via WSL2. Every change is measured against a baseline on standard benchmarks, and negative results are kept in.

**Model:** the French fine-tune is on Hugging Face at [GAMINGZI/qwen2.5-math-1.5b-french](https://huggingface.co/GAMINGZI/qwen2.5-math-1.5b-french).

**Stack:** PyTorch 2.11 (CUDA 12.8), Transformers 5.x, PEFT (LoRA), Hugging Face Datasets · Ubuntu on WSL2

## Results

### 1. SFT on a 0.5B model: no gain
| Model | GSM8K (200 q, greedy) |
|---|---|
| Qwen2.5-0.5B-Instruct | 43.5% |
| + LoRA SFT on 2k MetaMathQA examples | 41.5% |

Training loss barely moved (0.25 → 0.22). The instruct model had already seen this kind of data, so there was nothing new to learn. Switched to a math-specialized base.

### 2. Qwen2.5-Math-1.5B-Instruct baseline
| Benchmark | Score |
|---|---|
| GSM8K (200 q, greedy) | 84.0% |
| MATH-500 (500 q, greedy) | 72.6% |

Failure analysis on MATH-500: mostly real errors (arithmetic slips, wrong setups), with only ~1% answer-format mismatches. Weakest areas: Intermediate Algebra, Geometry, Precalculus, and Level 5 problems.

### 3. Majority voting and GRPO (MATH-500, first 200 questions, k=8 samples)
| Model | single sample | majority vote@8 | pass@8 |
|---|---|---|---|
| Base | 74.6% | **78.5%** | 88.5% |
| GRPO, 50 steps, lr 1e-5 | 74.2% | 78.5% | 88.0% |
| GRPO, +50 steps, lr 5e-5 | 73.9% | 77.5% | 87.0% |

- **Majority voting** gives about +4 points over a single answer with no training.
- **GRPO** (my own LoRA implementation, binary correctness reward, group-normalized advantages, trained on MATH train levels 4–5) **did not improve** the model at this scale (100 steps ≈ 400 problems). Run 1 barely changed the weights: the mean LoRA-B norm was 0.02, and the eval outputs were near-identical to the base model under the same seed, which pointed to a learning rate that was too low. Run 2 changed the weights but stayed within noise (about ±3 points at n=200). Qwen already RL-trained this model with far more compute.

### 4. French math (MGSM, 250 grade-school problems)
| Model | MGSM-en | MGSM-fr |
|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | 86.0% | **38.8%** |
| **+ French SFT (below)** | 85.6% | **57.6%** (+18.8) |

The same problems drop by 47 points in French. Failures are genuine misreadings of the French question, not parsing issues (only 8/153 had no final answer).

**Approach (cross-lingual self-distillation):** translate GSM8K *train* questions to French (opus-mt-en-fr, filtered so every number survives translation). The model solves the **English** originals, and only correct solutions are kept. Then LoRA-SFT on *French question → the model's own correct English solution*. This teaches French comprehension without changing the reasoning style, and MGSM's test problems come from GSM8K's test split, so there is no train/test overlap.

**Data:** 3,000 GSM8K train questions → 2,805 passed the number-preservation check → 2,684 kept after self-solving (96% solve rate). **Training:** LoRA r=32 on all attention and MLP projections, 1 epoch, lr 1e-4, about 25 minutes on the RTX 5060 Ti.

**Result:** French accuracy rose from 38.8% to 57.6% (+18.8 points, well beyond the ~±3 noise at n=250), with English unchanged (86.0% → 85.6%). That closes about 40% of the French–English gap, and it is the only training run in this project that produced a real gain.

Weights: [GAMINGZI/qwen2.5-math-1.5b-french](https://huggingface.co/GAMINGZI/qwen2.5-math-1.5b-french).

### 5. Comparison with existing French-capable small models (MGSM, same script)
| Model | Size | MGSM-en | MGSM-fr | Answers in |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | 1.5B | 86.0% | 38.8% | English |
| **Ours (+ French SFT)** | 1.5B | **85.6%** | 57.6% | English |
| Qwen/Qwen3.5-2B (non-thinking) | 2B | – | 62.0% | French |
| kurakurai/Luth-2-2B | 2B | 80.4% | **72.8%** | French |
| kurakurai/Luth-2-2B, 2048-token budget | 2B | – | **77.2%** | French |

Same prompt, greedy decoding, 1024-token budget for all models. Both Qwen3.5-based models run with their default chat template, which prefills an empty `<think></think>` (non-thinking mode). Qwen3.5-2B with thinking enabled would likely score higher but needs a much larger token budget.

**Honest read:** Luth-2-2B (a French fine-tune of Qwen3.5-2B) beats our model by 15 points in French, well beyond noise. Its French–English gap is only 7.6 points, against 28 for ours. Stock Qwen3.5-2B is 4.4 points ahead of us in French, which is borderline at n=250 (±3 each). Our model is still the best of the three in English (85.6% vs 80.4% for Luth), and it is smaller. Luth's misses are mostly long rambles cut off at the token limit: 38 answers hit the 1024-token cap. With 2048 tokens it rises to 77.2% (11 wrong answers become right, none flip the other way), but 20 answers are still cut off. Our model only hit the 1024 cap twice, so a bigger budget would not help it, and the 2048 number is the fair one for Luth.

### 6. Harder French eval: MATH-500 in French (first 200 problems)
GSM-level problems are not enough to judge French math. I translated the first 200 MATH-500 problems with opus-mt-en-fr (`make_math500_fr.py`). All LaTeX (`$...$`, `\[...\]`, environments, `[asy]` blocks, `\$`) is replaced with placeholders before translation and restored afterwards. A problem is kept only if every LaTeX span comes back byte-identical, every number in the prose survives, and no English prose ended up inside a math span. An independent audit of all 200 problems found every LaTeX span identical, `\frac` 70/70 and `$` 1202/1202, and no leftover English. The data is in `math500_fr.jsonl`.

| Model | MATH-fr | MATH-en (same 200) | Gap | Cut off (fr) |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B-Instruct (base) | 62.5% | 74.6%* | 12.1 | 2 |
| **Ours (+ French SFT)** | **65.0%** | 74.5% | 9.5 | 1 |
| kurakurai/Luth-2-2B | 62.0% | 64.0% | 2.0 | 65 |

Greedy decoding, 2048 tokens. *The base model's English figure is the single-sample score from section 3 (sampled, k=8, same 200 problems), not a greedy run.

**Honest read:** on harder math our 1.5B model is level with Luth-2-2B (65.0 vs 62.0, within ±3 noise). The French SFT does **not** measurably help here (+2.5 over base, within noise). MATH problems are mostly LaTeX, so the base model loses only about 12 points in French, against 47 on MGSM. The French gain is in word-problem comprehension. English MATH is unchanged (74.5% vs 74.6%). Luth's French gap is tiny, but it is capped by verbosity: a third of its answers don't finish within 2048 tokens, in either language.

### 7. Negative result: "translate-first" SFT
**Failure analysis:** 77 of our 106 MGSM-fr misses are questions the model solves in English. It misreads the French while restating it ("3 fois par semaine" → "6 days a week"). **Idea:** train French question → *the original English question* + solution, so the restatement step learns to be faithful (`make_fr_translate_first.py`, same 2,684 examples, same LoRA setup).

| Model | MGSM-fr |
|---|---|
| Ours (+ French SFT) | **57.6%** |
| + translate-first | 42.8% (−14.8) |

It made things much worse. The model always writes the English restatement, but a 1.5B math model cannot translate: a house renovation became a used-clothing store, vacuum-cleaner sales became a cake recipe. Once it has written the wrong question it solves that one, while the original model kept looking at the French question during its reasoning. Lesson: don't force an explicit intermediate step that the model is bad at.

### 8. French v2: more and better data → MGSM-fr 67.2%
Goal: at least 70% on MGSM-fr. To avoid tuning on the test set, I built a **French dev set** of 250 GSM8K test questions that are *not* in MGSM, translated by Luth-2-2B (`mgsm_dev_fr.jsonl`). Versions are chosen on the dev set, and MGSM-fr stays the reported test score. The dev set tracks MGSM closely: base 36.0 vs 38.8, v1 57.2 vs 57.6.

**What changed vs v1** (`make_fr_data2.py`, same training script and LoRA setup):
- all 7,473 GSM8K train questions instead of 3,000;
- each question translated **twice**: opus-mt (literal) and Luth-2-2B (natural French). Luth gets a prompt to keep every number as digits and not to solve. The number check now understands French number words;
- 13,644 examples (6,720 opus-mt + 6,924 Luth), about 5× v1. Targets are still the base model's own correct English solutions (7,154 of 7,473 correct).

| Model | MGSM-fr | French dev | MGSM-en |
|---|---|---|---|
| Base | 38.8% | 36.0% | 86.0% |
| v1 (2.7k, opus-mt) | 57.6% | 57.2% | 85.6% |
| **v2 (13.6k, opus-mt + Luth)** | **67.2%** | **65.6%** | 82.8% |

**Honest read:** +9.6 on MGSM-fr and +8.4 on dev, both well beyond the ±3 noise, so this is a real gain. English dipped 2.8 points, which is borderline. v2 fixed 39 of v1's French misses and broke 15. It still misses 47 French questions that it solves in English, so there is headroom left. Training took 50 minutes. v2 replaced v1 on Hugging Face, and v1 stays in the repo history.

## What I learned
- Measure a baseline first. Two of three training attempts looked plausible in the training logs, and the evals showed no gain.
- Rising training reward is not evidence: it rose 58% → 69% in GRPO run 1 while the benchmark didn't move.
- Check that training actually changed the model (adapter weight norms, identical outputs under a fixed seed).
- Tune where the model is weak (French: +18.8 points), not where it's already been heavily optimized (English MATH: no gain from SFT or RL).
- Self-distillation (training on the model's own filtered outputs) improves a weak skill without damaging the strong one.
- Data was the bottleneck, not code: 5× more examples with a second, natural-French translation took MGSM-fr from 57.6% to 67.2%.
- Keep a dev set separate from the benchmark before chasing a target number.
- Forcing an explicit intermediate step the model is bad at (translation) makes errors worse: the model commits to its own wrong restatement.

## Files
| File | Purpose |
|---|---|
| `eval_gsm8k.py` | GSM8K greedy eval |
| `eval_math500.py` | MATH-500 greedy eval, saves per-question outputs |
| `eval_math500_vote.py` | single / majority-vote / pass@k eval with subject and level breakdown, resumable |
| `eval_mgsm.py` | MGSM eval in any language (en, fr, …), `--max_new` for the token budget, `--data` for the dev set |
| `make_math500_fr.py` | translates MATH-500 to French with LaTeX protected by placeholders, plus integrity checks |
| `eval_math500_fr.py` | MATH eval on the French set (`--lang en` runs the same problems in English) |
| `train_sft.py` | incremental LoRA SFT on MetaMathQA |
| `grpo.py` | GRPO (RL) with LoRA, written from scratch |
| `make_fr_data.py` | builds French training data (translation + self-solved filtering) |
| `train_sft_fr.py` | LoRA SFT on the French data |
| `make_fr_translate_first.py` | translate-first variant of the French data (negative result, section 7) |
| `make_fr_data2.py` | v2 French data: all GSM8K train questions, opus-mt + Luth-2-2B translations, plus the French dev set |
| `eval_mgsm_vote.py` | MGSM / dev-set eval with majority voting (k samples), single / vote@k / pass@k |
| `chat.py` / `chat_vote.py` | interactive chat, plain or with majority voting |

## Run
```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell GPUs need cu128+
pip install transformers peft datasets accelerate sentencepiece sacremoses
python eval_math500_vote.py --n 200 --k 8
```
