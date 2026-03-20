# LLM Bullying: Iterative Rejection for AI Detection Evasion

Does repeatedly telling an LLM "this sounds too AI-generated" actually make its output pass AI detectors? We tested this empirically using GPT-4.1, two detectors (Binoculars, RoBERTa), and 50 questions from the HC3 dataset.

## Key Findings

- **Iterative rejection reduces detection by 43%** on the Binoculars detector (P(AI): 0.855 → 0.483 over 5 rounds)
- **A well-crafted single-step prompt achieves 60% reduction** (0.798 → 0.342), outperforming iterative rejection
- **Iterative rejection outperforms random selection** (best-of-5): 72% win rate, p < 0.001 — the rejection context itself matters
- **The mechanism is regime shift, not feature suppression**: the model switches to a "trying to be casual" persona rather than suppressing specific AI markers
- **Generic "sound human" instructions backfire**: single-step humanization actually *increases* detection scores

## Project Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature review (15 papers)
├── resources.md                 # Catalog of datasets, papers, code
├── src/
│   ├── experiment.py            # Text generation pipeline (5 conditions)
│   ├── detect.py                # AI detection scoring (RoBERTa + Binoculars)
│   └── analyze.py               # Statistical analysis and visualization
├── results/
│   ├── generation_results.json  # All generated text
│   ├── scored_results.json      # Detection scores for all text
│   ├── analysis.json            # Statistical analysis summary
│   └── plots/                   # Visualizations (9 plots)
├── datasets/                    # HC3, MAGE, RAID datasets
├── papers/                      # 15 downloaded research papers
└── code/                        # 7 cloned baseline repositories
```

## Reproducing

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai transformers torch scipy matplotlib seaborn accelerate

# Run (requires OPENAI_API_KEY and GPU)
python src/experiment.py    # Generate text (~35 min)
python src/detect.py        # Score with detectors (~2 min)
python src/analyze.py       # Analyze and plot
```

**Hardware**: 4× NVIDIA RTX A6000 (49GB each) for Binoculars (Falcon-7B models)
**API cost**: ~$5-10 for 750 GPT-4.1 calls
**Total runtime**: ~40 minutes

See [REPORT.md](REPORT.md) for full methodology and results.
