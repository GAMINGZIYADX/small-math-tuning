# Small-model math tuning on a consumer GPU

Experiments in improving small open LLMs (0.5B–1.5B) at math reasoning, on one RTX 5060 Ti (16 GB) via WSL2. Every change is measured against a baseline on standard benchmarks, and negative results are kept in.

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

### 5. Comparison with existing French-capable small models (MGSM, same script)
| Model | Size | MGSM-en | MGSM-fr | Answers in |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | 1.5B | 86.0% | 38.8% | English |
| **Ours (+ French SFT)** | 1.5B | **85.6%** | 57.6% | English |
| Qwen/Qwen3.5-2B (non-thinking) | 2B | – | 62.0% | French |
| kurakurai/Luth-2-2B | 2B | 80.4% | **72.8%** | French |

Same prompt, greedy decoding, 1024-token budget for all models. Both Qwen3.5-based models run with their default chat template, which prefills an empty `<think></think>` (non-thinking mode). Qwen3.5-2B with thinking enabled would likely score higher but needs a much larger token budget.

**Honest read:** Luth-2-2B (a French fine-tune of Qwen3.5-2B) beats our model by 15 points in French, well beyond noise. Its French–English gap is only 7.6 points, against 28 for ours. Stock Qwen3.5-2B is 4.4 points ahead of us in French, which is borderline at n=250 (±3 each). Our model is still the best of the three in English (85.6% vs 80.4% for Luth), and it is smaller. Luth's misses are mostly long rambles cut off at the token limit (26 of 37 unparsed answers).

## What I learned
- Measure a baseline first. Two of three training attempts looked plausible in the training logs, and the evals showed no gain.
- Rising training reward is not evidence: it rose 58% → 69% in GRPO run 1 while the benchmark didn't move.
- Check that training actually changed the model (adapter weight norms, identical outputs under a fixed seed).
- Tune where the model is weak (French: +18.8 points), not where it's already been heavily optimized (English MATH: no gain from SFT or RL).
- Self-distillation (training on the model's own filtered outputs) improves a weak skill without damaging the strong one.

## Files
| File | Purpose |
|---|---|
| `eval_gsm8k.py` | GSM8K greedy eval |
| `eval_math500.py` | MATH-500 greedy eval, saves per-question outputs |
| `eval_math500_vote.py` | single / majority-vote / pass@k eval with subject and level breakdown, resumable |
| `eval_mgsm.py` | MGSM eval in any language (en, fr, …) |
| `train_sft.py` | incremental LoRA SFT on MetaMathQA |
| `grpo.py` | GRPO (RL) with LoRA, written from scratch |
| `make_fr_data.py` | builds French training data (translation + self-solved filtering) |
| `train_sft_fr.py` | LoRA SFT on the French data |
| `chat.py` / `chat_vote.py` | interactive chat, plain or with majority voting |

## Run
```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell GPUs need cu128+
pip install transformers peft datasets accelerate sentencepiece sacremoses
python eval_math500_vote.py --n 200 --k 8
```
