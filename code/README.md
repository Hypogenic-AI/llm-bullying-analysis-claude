# Cloned Research Repositories

This directory contains cloned repositories relevant to our research on LLM-generated text detection and adversarial attacks against detectors. The collection covers both attack methods (evasion) and detection methods, as well as benchmarking infrastructure.

---

## Overview Table

| Repo | Role | Paper Venue |
|------|------|-------------|
| SICO | Attack (prompt optimization) | arXiv 2023 |
| CoPA | Attack (contrastive paraphrase, training-free) | EMNLP 2025 |
| AdvPara | Attack (adversarial paraphrase, training-free) | arXiv 2025 |
| Binoculars | Detector (zero-shot, cross-perplexity) | arXiv 2024 |
| Fast-DetectGPT | Detector (zero-shot, conditional probability curvature) | ICLR 2024 |
| HMGC | Attack (adversarial word substitution) | COLING 2024 |
| RAID | Benchmark (dataset + evaluation framework) | ACL 2024 |

---

## 1. SICO — Substitution-based In-Context example Optimization

**URL:** https://github.com/ColinLu50/Evade-GPT-Detector

**Purpose:** Automatically optimizes few-shot prompts that guide LLMs (ChatGPT, Vicuna) to generate human-like text that evades AI-generated text detectors. Two modes: SICO-gen (generation task) and SICO-para (paraphrase task).

**Key Files:**
- `SICO_train.py` — main training loop; optimizes in-context examples against a proxy detector
- `SICO_test_gen.py` — uses trained prompt to generate evading texts
- `run_test_detection.py` — evaluates generated texts against multiple detectors
- `detectors.py` — wrapper implementations for GPTzero, OpenAI detector, DetectGPT, Log-Rank, GPT-2 detector, ChatGPT detector
- `sico/` — core SICO algorithm, LLM API wrappers (`LLM_api.py`)
- `datasets/` — SQuAD, ELI5, Yelp splits (eval, test, incontext TSVs)
- `environment.yml` — Conda environment spec

**Dependencies:** OpenAI API (ChatGPT), optional Vicuna (FastChat local server), GPTzero API key, HuggingFace models (auto-downloaded)

**Entry Points:**
```bash
# Training
python SICO_train.py --dataset squad --llm chatgpt --detector chatdetect --task essay --incontext-size 8 --eval-size 32 --train-iter 6

# Text generation with trained prompt
python SICO_test_gen.py --dataset squad --llm chatgpt --detector chatdetect --task essay --incontext-size 8 --eval-size 32 --train-iter 6 --test-size 100

# Evaluate against a detector
python run_test_detection.py --dataset squad --method SICO-squad-essay-chatgpt-chatdetect --detector detectgpt
```

**Relevance:** Demonstrates prompt-based evasion of detectors. The optimized prompts are directly applicable to studying how LLM-generated bullying content could evade automated moderation systems.

---

## 2. CoPA — Contrastive Paraphrase Attacks

**URL:** https://github.com/ffhibnese/CoPA_Contrastive_Paraphrase_Attacks

**Purpose:** Training-free paraphrase attack that uses contrastive decoding: generates human-like text via an LLM while subtracting a machine-like auxiliary distribution at inference time, causing output to evade statistical detectors.

**Key Files:**
- `paraphrase.sh` — runs paraphrasing pipeline (e.g., `bash paraphrase.sh xsum`)
- `evaluate.sh` / `evaluate.py` — evaluates paraphrased results against detectors
- `detectors/` — implementations of Fast-DetectGPT, DetectGPT, GPT-2 detector, Ghostbuster, RoBERTa detector, and others
- `scripts/` — additional experiment scripts
- `paraphrase-at-scale/` — SIM (semantic similarity) model integration
- `transformers/` — local fork of HuggingFace Transformers (modified for contrastive decoding)
- `data/` — raw and result data (Xsum and others)
- `requirement.txt` — pip dependencies

**Dependencies:** Python 3.11.7, PyTorch, HuggingFace Transformers (local fork), Qwen2.5-72B-Instruct (paraphraser model, from HuggingFace), SIM model (Google Drive download), `pip install -r requirement.txt && pip install -e transformers`

**Entry Points:**
```bash
# Paraphrase the Xsum dataset
bash paraphrase.sh xsum

# Evaluate against Fast-DetectGPT
bash evaluate.sh data/results/xsum_qwen2.5-72b_gpt-3.5-turbo_lambda_0.5_alpha_1e-05_temperature_1.0_max_tries_10_paraphrase.raw_data.json fdgpt
```

**Relevance:** State-of-the-art training-free evasion attack (EMNLP 2025). Directly applicable to testing whether LLM-generated bullying text can be paraphrased to bypass detection systems.

---

## 3. AdvPara — Adversarial Paraphrasing

**URL:** https://github.com/chengez/Adversarial-Paraphrasing

**Purpose:** Training-free, universal adversarial paraphrase attack. Uses an instruction-following LLM guided by a real detector's feedback to iteratively produce text that evades detection. Shown to be highly transferable across detectors (exploits shared human-text distribution across detectors).

**Key Files:**
- `utils.py` — core Adversarial Paraphrasing algorithm implementation
- `paraphrase_and_detect.py` — main script to run paraphrasing and detection together
- `detect_existing_paraphrased_text.py` — run detectors on already-paraphrased outputs
- `parseNsave_paraphrased_output.py` — convert log outputs to HuggingFace dataset format
- `quality_judge_utils.py` — GPT-4o based quality evaluation prompt templates
- `zs_detectors/` — zero-shot detector implementations (`detector.py`, `models/`)
- `MAGE/` — MAGE supervised detector integration
- `scripts/` — SLURM and local bash scripts (including `transfer_test.sbatch`, `create_wm_mage.sbatch`, `detect_existing_paraphrased_text.sbatch`)
- `kgw_wm/`, `uni_wm/` — pre-generated watermarked datasets (KGW and Unigram)
- `outputs/` — saved paraphrased text outputs and detection scores
- `requirements.txt` — pip dependencies

**Dependencies:** Python >= 3.10, PyTorch, HuggingFace Transformers, OpenAI API (for quality evaluation with GPT-4o), `pip install -r requirements.txt`

**Entry Points:**
```bash
# Run adversarial paraphrasing (local GPU)
bash scripts/transfer_test.sbatch > log_file.log

# Test other detectors on existing outputs
python parseNsave_paraphrased_output.py
bash scripts/detect_existing_paraphrased_text.sbatch
```

**Relevance:** Demonstrates that guiding paraphrase with one strong detector transfers evasion to other detectors. Critical for understanding transferable attacks in the context of bullying-content moderation pipelines.

---

## 4. Binoculars — Zero-Shot LLM-Generated Text Detector

**URL:** https://github.com/ahans30/Binoculars

**Purpose:** State-of-the-art zero-shot detector that computes a cross-perplexity score using two related language models (Falcon-7B and Falcon-7B-Instruct by default). No training data required; achieves strong detection with a fixed global threshold.

**Key Files:**
- `binoculars/detector.py` — core `Binoculars` class with `compute_score()` and `predict()` methods
- `binoculars/metrics.py` — perplexity/cross-entropy metric computation
- `binoculars/utils.py` — utility functions
- `binoculars/__init__.py` — package entry point
- `main.py` — script for running detection experiments
- `app.py` — Gradio demo app
- `requirements.txt` — pip dependencies
- `setup.py` — installable package

**Dependencies:** Python 3.9, PyTorch, HuggingFace Transformers, Falcon-7B + Falcon-7B-Instruct (auto-downloaded), `pip install -e .`

**Entry Points:**
```python
from binoculars import Binoculars
bino = Binoculars()
score = bino.compute_score("some text")  # float
label = bino.predict("some text")         # 'Most likely AI-Generated' or human
```
```bash
python app.py  # Gradio demo
python main.py  # experiments
```

**Relevance:** Primary detector to test evasion attacks against. Also referenced in Fast-DetectGPT, CoPA, and AdvPara evaluations. Zero-shot nature makes it suitable as a reference detector in our pipeline.

---

## 5. Fast-DetectGPT — Efficient Zero-Shot Detection via Conditional Probability Curvature

**URL:** https://github.com/baoguangsheng/fast-detect-gpt

**Purpose:** ICLR 2024 method that improves DetectGPT by replacing perturbation-based sampling with conditional probability curvature estimation. Achieves 340x speedup and higher AUROC than DetectGPT. Supports both white-box (known source model) and black-box (surrogate model) detection.

**Key Files:**
- `scripts/fast_detect_gpt.py` — core Fast-DetectGPT algorithm
- `scripts/local_infer.py` — interactive local demo
- `scripts/detect_gpt.py` — original DetectGPT baseline
- `scripts/detect_llm.py` — general detection script covering multiple methods
- `scripts/baselines.py` — supervised detector baselines
- `scripts/data_builder.py` — dataset loading utilities
- `scripts/model.py` — LLM model loading
- `scripts/metrics.py` — AUROC and other metrics
- `main.sh` — reproduce main 5-model experiments
- `gpt3to4.sh` — experiments for GPT-3/ChatGPT/GPT-4
- `setup.sh` — environment setup
- `requirements.txt` — pip dependencies
- `exp_main/`, `exp_gpt3to4/` — pre-run experiment outputs and GPT data

**Dependencies:** Python 3.8, PyTorch 1.10.0, HuggingFace Transformers, gpt-neo-2.7B or gpt-j-6B or Llama3-8B as scoring/sampling models, Tesla A100 (80GB) recommended, `bash setup.sh`

**Entry Points:**
```bash
bash setup.sh
python scripts/local_infer.py                                    # interactive demo
python scripts/local_infer.py --sampling_model_name gpt-j-6B    # more accurate
bash main.sh       # reproduce main experiments
bash gpt3to4.sh    # GPT-3/ChatGPT/GPT-4 experiments
```

**Relevance:** A commonly-used baseline detector. CoPA and AdvPara both evaluate against it. Used to measure how robust evasion attacks are against conditional-probability-based detectors.

---

## 6. HMGC — Humanizing Machine-Generated Content

**URL:** https://github.com/zhouying20/HMGC

**Purpose:** COLING 2024 method that evades AI-text detectors via adversarial word substitution. Trains a surrogate classifier by distilling labels from a target detector, then uses a multi-process FLINT-style attack to substitute words adversarially.

**Key Files:**
- `train_detector.py` — train a surrogate detector on distilled labels from the target victim
- `attack/multi_flint_attack.py` — main multi-process adversarial attack entry point
- `attack/` — attack implementations and recipes
- `evaluation/eval_accuracy.py` — evaluate evasion accuracy across detectors
- `utils/` — shared utilities
- `requirements.txt` — pip dependencies

**Dependencies:** Python, PyTorch, HuggingFace Transformers, dataset from Google Drive (linked in README), `pip install -r requirements.txt`

**Entry Points:**
```bash
# Evaluate accuracy on pre-generated outputs
python evaluation/eval_accuracy.py \
    --detector hc3 \
    --tests ./output/hc3/**/*.jsonl \
    --output_file /tmp/hc3_evaluation.csv

# Train surrogate model
python train_detector.py

# Run multi-process adversarial attack
# (follow attack/multi_flint_attack.py API)
```

**Relevance:** Word-level adversarial attack complementing paraphrase-level attacks. Useful for studying fine-grained evasion strategies against detectors that might be deployed in bullying detection pipelines.

---

## 7. RAID — Robust AI Detection Benchmark

**URL:** https://github.com/liamdugan/raid

**Purpose:** ACL 2024 benchmark: the largest and most comprehensive dataset for evaluating AI-generated text detectors. Contains 10M+ documents spanning 11 LLMs, 11 genres, 4 decoding strategies, and 12 adversarial attacks. Provides a standardized evaluation framework and public leaderboard at https://raid-bench.xyz.

**Key Files:**
- `detect_cli.py` — CLI to run any detector on the RAID dataset
- `evaluate_cli.py` — CLI to evaluate detector predictions at a target FPR
- `detectors/detector.py` — registry of all implemented detectors (Binoculars, Fast-DetectGPT, GLTR, GPTZero, RoBERTa, RADAR, Winston AI, Originality AI, etc.)
- `detectors/models/` — individual detector model implementations
- `generation/adversarial/attack.py` — adversarial attack implementations (homoglyph, paraphrase, synonym, whitespace, zero-width space, etc.)
- `generation/generate.py` — text generation pipeline
- `leaderboard/` — submission templates and existing results
- `requirements.txt` — pip dependencies

**Dependencies:** Python 3.9.7, PyTorch, HuggingFace, OpenAI API (optional for some detectors), Cohere API (optional), `pip install -r requirements.txt` or `pip install raid-bench`

**Entry Points:**
```bash
# Install as package
pip install raid-bench

# CLI detection and evaluation
python detect_cli.py -m gltr -d train.csv -o predictions.json
python evaluate_cli.py -r predictions.json -d train.csv -o results.json -t 0.05

# Programmatic usage
from raid import run_detection, run_evaluation
from raid.utils import load_data
train_df = load_data(split="train")
predictions = run_detection(my_detector, train_df)
evaluation_result = run_evaluation(predictions, train_df)

# Adversarial attacks
from generation.adversarial.attack import get_attack
a = get_attack("paraphrase")
result = a.attack("some text")
```

**Relevance:** Gold-standard evaluation framework. We can use RAID to benchmark how well detectors perform on AI-generated bullying content and how adversarial attacks degrade detection. The 12 attack types cover a broad range of evasion strategies relevant to our research.

---

## Research Context and Cross-Repo Relationships

These repositories collectively address two sides of the AI-generated text detection problem:

**Attack side (evasion):**
- SICO: prompt-level optimization for generation tasks
- CoPA: contrastive decoding for paraphrase attacks (training-free, EMNLP 2025)
- AdvPara: detector-guided adversarial paraphrasing (training-free, highly transferable)
- HMGC: word-level adversarial substitution (surrogate-model based)
- RAID also includes 12 attack implementations

**Detection side:**
- Binoculars: cross-perplexity zero-shot detector (SOTA baseline)
- Fast-DetectGPT: conditional probability curvature detector (ICLR 2024)
- RAID includes 10+ detector implementations

**Evaluation infrastructure:**
- RAID: standardized dataset and metrics for comparing detectors and attacks

For LLM-bullying research, the natural pipeline is:
1. Generate bullying content via LLMs
2. Apply evasion attacks (SICO, CoPA, AdvPara, HMGC) to humanize the content
3. Evaluate detection under attack using detectors (Binoculars, Fast-DetectGPT)
4. Use RAID as a standardized evaluation harness and comparison baseline
