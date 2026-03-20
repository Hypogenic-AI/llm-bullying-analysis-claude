# Datasets

This directory contains datasets used to evaluate whether iteratively rejecting LLM output
as "too AI-sounding" improves evasion of AI detectors compared to single-step prompting.
Each dataset provides human-written and AI-generated text pairs for training or evaluating detectors.

---

## HC3 — Human ChatGPT Comparison Corpus

**Paper:** [HC3: A Human-ChatGPT Comparison Corpus](https://arxiv.org/abs/2301.07597)
**Source:** [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
**License:** CC BY-SA 4.0
**Local path:** `datasets/hc3/`

### Description

HC3 pairs the same questions with both human-written and ChatGPT-generated answers across
five domains: Reddit ELI5, Open QA, Wiki CSAI, Medicine, and Finance.

### Size and Splits

| File             | Entries | Size   | Domain        |
|------------------|--------:|-------:|---------------|
| `all.jsonl`      | 24,322  | 70.3 MB | All domains   |
| `reddit_eli5.jsonl` | 17,112 | 52.9 MB | Reddit ELI5  |
| `finance.jsonl`  | 3,933   | 9.4 MB  | Finance       |
| `medicine.jsonl` | 1,248   | 2.6 MB  | Medicine      |
| `open_qa.jsonl`  | 1,187   | 2.8 MB  | Open QA       |
| `wiki_csai.jsonl`| 842     | 2.1 MB  | Wikipedia CSAI|

Each entry contains one `question`, one or more `human_answers` (list of strings), and one or
more `chatgpt_answers` (list of strings), plus `index` and `source` metadata.

### Download

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download

files = ["all.jsonl", "finance.jsonl", "medicine.jsonl",
         "open_qa.jsonl", "reddit_eli5.jsonl", "wiki_csai.jsonl"]
for fname in files:
    hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename=fname,
        repo_type="dataset",
        local_dir="datasets/hc3"
    )
EOF
```

### Loading Code

```python
import json

def load_hc3(path="datasets/hc3/all.jsonl"):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

data = load_hc3()
# data[0] => {"question": "...", "human_answers": [...], "chatgpt_answers": [...], ...}
```

---

## RAID — Robust AI-generated text Detection benchmark

**Paper:** [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://arxiv.org/abs/2405.07940)
**Source:** [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid)
**License:** MIT
**Local path:** `datasets/raid/` (no files committed — too large; see download instructions)

### Description

RAID is the largest and most comprehensive dataset for evaluating AI-generated text detectors.
It contains over 10 million documents spanning 11 LLMs, 11 genres, 4 decoding strategies, and
12 adversarial attacks. The `model` field is `"human"` for human-written text; all other values
indicate AI-generated text.

### Size and Splits

| Split       | Labels | Domains                                            | Size (no adv.) | Size (with adv.) |
|-------------|--------|----------------------------------------------------|---------------:|-----------------:|
| `train.csv` | Yes    | News, Books, Abstracts, Reviews, Reddit, Recipes, Wiki, Poetry | 802 MB | 11.8 GB |
| `test.csv`  | No     | Same as train                                      | 81 MB          | 1.22 GB          |
| `extra.csv` | Yes    | Code, Czech, German                                | 275 MB         | 3.71 GB          |

**Note:** The test split contains only `id` and `generation` columns (no labels or metadata).
Use the train split for labeled experiments.

### Data Schema (train/extra splits)

| Field               | Description |
|---------------------|-------------|
| `id`                | UUID identifying the source content |
| `adv_source_id`     | UUID of adversarial source (if applicable) |
| `source_id`         | UUID of the human-written source text |
| `model`             | Generator: `human`, `chatgpt`, `gpt4`, `gpt3`, `gpt2`, `llama-chat`, `mistral`, `mistral-chat`, `mpt`, `mpt-chat`, `cohere`, `cohere-chat` |
| `decoding`          | `greedy` or `sampling` |
| `repetition_penalty`| `yes` or `no` |
| `attack`            | Adversarial attack applied, or `none` |
| `domain`            | Genre: `abstracts`, `books`, `code`, `czech`, `german`, `news`, `poetry`, `recipes`, `reddit`, `reviews`, `wiki` |
| `title`             | Title of the source article |
| `prompt`            | Prompt used for generation |
| `generation`        | The generated or human-written text |

### Download

The full dataset is very large. Download only what you need:

```bash
# Using datasets library (streaming — avoids downloading everything)
python - <<'EOF'
from datasets import load_dataset

# Stream the training set (no full download)
raid = load_dataset("liamdugan/raid", "raid", streaming=True)
train_stream = raid["train"]

# Iterate a sample
for i, example in enumerate(train_stream):
    print(example["model"], example["domain"])
    if i >= 9:
        break
EOF

# Or download just the train CSV directly (~800 MB without adversarial)
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="liamdugan/raid",
    filename="train.csv",
    repo_type="dataset",
    local_dir="datasets/raid"
)
EOF
```

### Loading Code

```python
import csv

def load_raid_labeled(path="datasets/raid/train.csv"):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # row["model"] == "human" => human-written
            # anything else           => AI-generated
            yield row
```

---

## MAGE — Machine-Generated Text Detection in the Wild

**Paper:** [MAGE: Machine-generated Text Detection in the Wild](https://arxiv.org/abs/2305.13242)
**Source:** [yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE)
**License:** Apache 2.0
**Local path:** `datasets/mage/`

### Description

MAGE is a comprehensive testbed for machine-generated text detection covering texts from various
human writings and LLMs across diverse domains. Labels: `0` = human-written, `1` = AI-generated.

### Downloaded Files

| File                    | Rows  | Size   | Description                        |
|-------------------------|------:|-------:|------------------------------------|
| `test_ood_set_gpt.csv`  | 1,562 | 2.1 MB | OOD test set (GPT-generated texts) |

**Label distribution in `test_ood_set_gpt.csv`:** 800 human (label=0), 762 AI (label=1)

### Full Dataset Size

| File        | Rows (approx) | Size  | Split      |
|-------------|:-------------:|------:|------------|
| `train.csv` | ~             | 385 MB| Train      |
| `valid.csv` | ~             | 69 MB | Validation |
| `test.csv`  | ~             | 68 MB | Test       |
| `test_ood_set_gpt.csv` | 1,562 | 2.1 MB | OOD Test (GPT) |

### Download

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download

# Download the small OOD test set (2.1 MB)
hf_hub_download(
    repo_id="yaful/MAGE",
    filename="test_ood_set_gpt.csv",
    repo_type="dataset",
    local_dir="datasets/mage"
)

# For the full test split (68 MB):
# hf_hub_download(repo_id="yaful/MAGE", filename="test.csv",
#                 repo_type="dataset", local_dir="datasets/mage")
EOF
```

### Loading Code

```python
import csv

def load_mage(path="datasets/mage/test_ood_set_gpt.csv"):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "text": row["text"],
                "label": int(row["label"]),  # 0=human, 1=AI
                "src": row["src"]
            })
    return rows

data = load_mage()
human = [r for r in data if r["label"] == 0]
ai    = [r for r in data if r["label"] == 1]
```

---

## Samples

Small sample files (first 10 entries) are saved in `datasets/samples/` and committed to git:

| File                  | Dataset | Entries |
|-----------------------|---------|--------:|
| `hc3_sample.json`     | HC3     | 10      |
| `raid_sample.json`    | RAID    | 10 (5 human + 5 AI) |
| `mage_sample.json`    | MAGE    | 10      |

---

## Notes for Research

- **HC3** is ideal for controlled human-vs-ChatGPT comparisons across domains.
- **RAID** is best for robustness evaluation — it covers adversarial attacks and many LLMs, making it suitable for testing whether iterative rewriting evades a broad range of detectors.
- **MAGE** provides an OOD evaluation scenario with GPT-generated text across diverse topics.

For this project's iterative-rejection experiment, the key field in each dataset is:

| Dataset | Human indicator              | AI indicator                  |
|---------|------------------------------|-------------------------------|
| HC3     | `human_answers` field        | `chatgpt_answers` field       |
| RAID    | `model == "human"`           | `model != "human"`            |
| MAGE    | `label == 0`                 | `label == 1`                  |
