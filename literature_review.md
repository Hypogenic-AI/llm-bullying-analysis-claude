# Literature Review: LLM Bullying — Iterative Rejection for AI Detection Evasion

## Research Area Overview

This review covers the intersection of AI-generated text (AIGT) detection and evasion techniques, with focus on iterative modification strategies. The core research question is whether iteratively rejecting an LLM's output as "too AI-sounding" leads to text that passes AI detectors more effectively than single-step prompting. The literature reveals a rapidly evolving arms race between detection and evasion methods, with paraphrase-based attacks emerging as the dominant evasion strategy.

## Key Papers

### Paper 1: SICO — Large Language Models can be Guided to Evade AI-Generated Text Detection
- **Authors:** Lu et al.
- **Year:** 2024
- **Source:** TMLR (arXiv: 2305.10847)
- **Key Contribution:** Substitution-based In-Context example Optimization (SICO) — constructs prompts with optimized in-context examples that guide LLMs to generate undetectable text.
- **Methodology:** Iterative greedy substitution loop (word-level via WordNet synonyms + sentence-level via LLM paraphrasing) optimized against a proxy detector over N=6 rounds. Two variants: SICO-Gen (direct generation) and SICO-Para (post-hoc paraphrasing). Only requires 40 human-written examples and ~$1 in API costs.
- **Datasets:** SQuAD (academic writing), ELI5 (open QA), Yelp (reviews) — 200 AI + 200 human per task.
- **Results:** Reduces average AUC by ~0.5 across 6 detectors. At FPR=0.05, drops detection from ~70% to 0-1% on most detectors. Human annotators identify SICO text as AI only 14-37% of the time (comparable to actual human text at 13-22%).
- **Code:** https://github.com/ColinLu50/Evade-GPT-Detector
- **Relevance:** The proxy-detector feedback loop is structurally identical to rejection sampling — candidates scoring high AI-probability are rejected. This is the closest existing work to our hypothesis, but the iteration happens at prompt construction time, not at inference time per query.

### Paper 2: CoPA — Contrastive Paraphrase Attacks on LLM-Generated Text Detectors
- **Authors:** Fang et al. (Tsinghua/HIT)
- **Year:** 2025
- **Source:** EMNLP (arXiv: 2505.15337)
- **Key Contribution:** Training-free contrastive decoding that subtracts machine-like token distributions from human-like distributions during autoregressive generation.
- **Methodology:** Dual prompts (human-like + machine-like) produce two logit distributions per token. Contrastive formula amplifies human-preferred tokens: p_c ∝ exp[(1+λ)·f_human − λ·f_machine]. Adaptive truncation prevents incoherent tokens.
- **Datasets:** XSum, SQuAD, LongQA (150 samples each). Source LLMs: GPT-3.5, GPT-4, Claude-3.5, GPT-4o, Gemini-1.5-Pro.
- **Results:** Average TPR reduced to 7-13% across 8 detectors (vs. ~57-66% baseline). 30.55% average improvement over all detectors. Semantic similarity >90% preserved.
- **Code:** https://github.com/ffhibnese/CoPA_Contrastive_Paraphrase_Attacks
- **Relevance:** Self-corrective at token level — each decoding step checks and corrects machine-style biases. Multiple paraphrase iterations further reduce detection while maintaining quality (unlike DIPPER). Directly demonstrates iterative refinement benefits.

### Paper 3: Adversarial Paraphrasing — A Universal Attack for Humanizing AI-Generated Text
- **Authors:** Cheng, Sadasivan et al. (UMD)
- **Year:** 2025
- **Source:** NeurIPS (arXiv: 2506.07001)
- **Key Contribution:** Detector-guided token selection during LLM paraphrasing. At each decoding step, candidate tokens are scored by a guidance detector and the lowest AI-score token is selected.
- **Methodology:** LLaMA-3-8B paraphraser with top-p/top-k filtering. Each candidate token appended to partial output, scored by guidance detector (e.g., OpenAI-RoBERTa-Large). Lowest AI-score token selected. Training-free, gradient-free.
- **Datasets:** MAGE dataset (2K AI + 2K human, 100-500 tokens each). Also KGW/Unigram watermarked variants.
- **Results:** Average 87.88% T@1%F reduction across 8 detectors. Transfers universally — optimizing against one detector evades all others. Perplexity matches human text (14.26 vs 15.02).
- **Code:** https://github.com/chengez/Adversarial-Paraphrasing
- **Relevance:** Core insight: all well-trained detectors converge toward a shared model of human text, so evading one transfers to others. This supports our hypothesis — iterative rejection against any detector should generalize.

### Paper 4: Self-Disguise Attack (SDA)
- **Authors:** Zhou et al. (China Agricultural U.)
- **Year:** 2025
- **Source:** arXiv: 2508.15848
- **Key Contribution:** Adversarial feature extraction loop that automatically discovers linguistic "disguise features" making text appear human-written, combined with RAG-style retrieval of detection-resistant examples.
- **Methodology:** Iterative loop: generate text → evaluate with proxy detector → collect passing examples → extract linguistic features → update generation prompt → repeat until convergence. Features transfer across LLMs.
- **Datasets:** RAID (1K instances, split 6:2:2).
- **Results:** Average detection accuracy reduced to 26-47% across LLMs and detectors. Lower PPL and higher diversity than SICO.
- **Code:** https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection/tree/main/SDA
- **Relevance:** Most directly relevant to our hypothesis — uses an explicit rejection loop where text that fails detection is rejected and features of successful evasions are extracted and reused. The key difference from our hypothesis: SDA iterates on the prompt/features, not on individual outputs.

### Paper 5: Humanizing Machine-Generated Content (HMGC)
- **Authors:** Zhou, He, Sun (UCAS/ISCAS)
- **Year:** 2024
- **Source:** arXiv: 2404.01907
- **Key Contribution:** Word-level adversarial substitution using gradient-based importance ranking and perplexity-guided token replacement.
- **Methodology:** Gradient norm + perplexity importance ranking → BERT MLM masked substitution → POS/MPR/USE constraints → iterate until detector score falls below threshold. Dynamic adversarial learning: 10-round attacker-detector arms race.
- **Datasets:** CheckGPT (720K train, news/essay/research), HC3 (58K, QA).
- **Results:** White-box: 97.29% delta-Acc (detector reduced to random). Black-box: 46.35% delta-Acc. 10-round arms race reaches equilibrium at round 7.
- **Code:** https://github.com/zhouying20/HMGC
- **Relevance:** Directly models iterative modification with explicit termination criterion (detector score < threshold). The dynamic adversarial learning component models the arms race relevant to understanding detector robustness.

### Paper 6: Paraphrasing Evades Detectors of AI-Generated Text
- **Authors:** Krishna, Song, Karpinska, Wieting, Iyyer
- **Year:** 2023
- **Source:** NeurIPS (arXiv: 2303.13408)
- **Key Contribution:** DIPPER — 11B parameter controllable paraphraser that breaks all major detectors. Also proposes retrieval-based defense.
- **Methodology:** T5-XXL fine-tuned on 6.3M paraphrase pairs from PAR3 dataset. Two control knobs: lexical diversity (L) and content reordering (O). Paragraph-level, context-aware paraphrasing.
- **Datasets:** WikiText-103 (open-ended generation, 3K prompts), ELI5 (long-form QA). 15M-sequence corpus for retrieval defense.
- **Results:** At 1% FPR: DetectGPT drops from 70.3% to 4.6%, watermarking from 100% to 57.2%. Retrieval defense achieves 97%+ detection even after paraphrasing.
- **Code:** https://github.com/martiansideofthemoon/ai-detection-paraphrases
- **Relevance:** Foundational work establishing paraphrasing as the primary evasion method. Authors explicitly discuss iterative extension (repeatedly applying DIPPER) but note semantic drift limits. Establishes the evasion-quality tradeoff central to our hypothesis.

### Paper 7: DetectGPT — Zero-Shot Machine-Generated Text Detection
- **Authors:** Mitchell et al.
- **Year:** 2023
- **Source:** ICML (arXiv: 2301.11305)
- **Key Contribution:** Zero-shot detection based on probability curvature — AI text occupies negative curvature regions of the model's log-probability function.
- **Relevance:** Standard baseline detector. Vulnerable to paraphrasing attacks (drops to <5% with DIPPER).

### Paper 8: Fast-DetectGPT
- **Authors:** Bao et al.
- **Year:** 2024
- **Source:** ICLR (arXiv: 2310.05130)
- **Key Contribution:** Replaces DetectGPT's perturbation step with conditional probability curvature. 340x faster, ~75% higher AUROC.
- **Relevance:** Current SOTA zero-shot detector. Key evaluation target for evasion methods.

### Paper 9: Binoculars — Zero-Shot Detection of LLM-Generated Text
- **Authors:** Hans et al.
- **Year:** 2024
- **Source:** ICML (arXiv: 2401.12070)
- **Key Contribution:** Contrasts two closely related LLMs (e.g., Falcon-7B vs Falcon-7B-Instruct). Detects >90% at 0.01% FPR without training.
- **Code:** https://github.com/ahans30/Binoculars
- **Relevance:** Strong zero-shot detector. Important evaluation target alongside Fast-DetectGPT.

### Paper 10: Self-Refine — Iterative Refinement with Self-Feedback
- **Authors:** Madaan et al.
- **Year:** 2023
- **Source:** NeurIPS (arXiv: 2303.17651)
- **Key Contribution:** LLM acts as generator, feedback provider, and refiner in iterative loop. ~20% preference gains over static generation.
- **Relevance:** Establishes the general framework for iterative LLM self-improvement, which our hypothesis adapts to the specific case of detector evasion feedback.

### Paper 11: PADBen — Benchmark for AI Text Detectors Against Paraphrase Attacks
- **Authors:** Various
- **Year:** 2025
- **Source:** arXiv: 2511.00416
- **Key Contribution:** Comprehensive benchmark evaluating 11 detectors against paraphrase attacks. Shows critical asymmetry in attack effectiveness.
- **Relevance:** Provides standardized evaluation framework and establishes current SOTA performance baselines.

### Paper 12: RAID — Shared Benchmark for Robust Evaluation of Text Detectors
- **Authors:** Dugan et al.
- **Year:** 2024
- **Source:** ACL (arXiv: 2405.07940)
- **Key Contribution:** Largest benchmark: 10M+ generations, 11 models, 8 domains, 11 attacks, 4 decoding strategies.
- **Code:** https://github.com/liamdugan/raid
- **Relevance:** Standard benchmark infrastructure. Includes implementations of multiple detectors and attacks.

## Common Methodologies

### Evasion Approaches
1. **Prompt optimization** (SICO, SDA): Optimize in-context examples or generation prompts against a proxy detector. Iteration at prompt level.
2. **Contrastive decoding** (CoPA): Subtract machine-like distributions from human-like distributions at each token. Iteration at token level during generation.
3. **Detector-guided decoding** (AdvPara): Use detector as oracle during autoregressive generation to select low-AI-score tokens. Iteration at token level.
4. **Post-hoc paraphrasing** (DIPPER): Apply a trained paraphraser to rewrite AI text. Single-pass or iterable.
5. **Word-level substitution** (HMGC): Gradient-guided token replacement with quality constraints. Iteration until threshold met.

### Detection Approaches
1. **Zero-shot statistical** (DetectGPT, Fast-DetectGPT, Binoculars, Log-Rank): Use probability curvature or cross-model perplexity. No training data needed.
2. **Trained classifiers** (RoBERTa-based, RADAR, MAGE): Fine-tuned on labeled human/AI text. Can be adversarially hardened.
3. **Watermarking** (KGW, Unigram): Embed statistical signal during generation. Most resilient to paraphrasing but requires control of the generation process.
4. **Retrieval-based** (Krishna et al.): Search a corpus of prior generations. Most robust to paraphrasing (97%+) but requires API-level access.

## Standard Baselines
- **DIPPER** (Krishna et al., 2023): Standard paraphrase-based evasion baseline used by all subsequent work
- **DetectGPT/Fast-DetectGPT**: Standard zero-shot detection baselines
- **OpenAI RoBERTa detector**: Standard trained classifier baseline
- **GPTZero**: Standard commercial detector baseline

## Evaluation Metrics
- **AUROC**: Area under ROC curve (standard but less operationally relevant)
- **TPR@FPR=1%** (T@1%F): True positive rate at 1% false positive rate (preferred for real-world deployment)
- **TPR@FPR=5%**: More lenient threshold
- **P-SP / SBERT similarity**: Semantic preservation of paraphrased text (threshold: 0.76 from human paraphrase studies)
- **Perplexity (PPL)**: Text quality/naturalness measure (human text ~15 on Pythia/LLaMA)
- **BLEU/Self-BLEU**: Diversity and similarity measures
- **Human evaluation**: AI detectability rating + readability + task completion

## Datasets in the Literature
| Dataset | Used By | Task | Size |
|---------|---------|------|------|
| HC3 | SICO, HMGC, SDA | QA (human vs ChatGPT) | 24K pairs |
| MAGE | AdvPara, CoPA | Mixed domain classification | 4K texts |
| RAID | SDA, PADBen | Multi-model, multi-domain detection | 10M+ generations |
| SQuAD/WikiText | SICO, CoPA, DIPPER | Academic writing / open-ended gen | varies |
| ELI5 | SICO, CoPA, DIPPER | Long-form QA | varies |
| CheckGPT | HMGC | News/essay/research detection | 720K |
| XSum | CoPA | News summarization | 150 samples |

## Gaps and Opportunities

### Gap 1: No Study of Simple Iterative Rejection at Inference Time
All existing work either (a) iterates during a training/optimization phase (SICO, SDA), (b) integrates iteration into the decoding process (CoPA, AdvPara), or (c) applies single-pass paraphrasing (DIPPER). **No paper directly tests the simple scenario: generate → check with detector → reject and re-prompt → repeat.** This is precisely our research hypothesis.

### Gap 2: Single-Step "Sound Less Like AI" Prompt Baseline
While several papers test "human-style" prompts as baselines (e.g., SICO tests a "Human Prompt" baseline), none systematically compare single-step humanization prompting against multi-step iterative rejection with feedback.

### Gap 3: Mechanism Understanding
Papers demonstrate that evasion works but provide limited mechanistic insight into *why* iterative approaches access different generation regimes. Our hypothesis about "accessing relevant features only in the context of rejection" is untested.

### Gap 4: Practical Attack Simplicity
Existing methods require sophisticated implementations (contrastive decoding, gradient computation, trained paraphrasers). The simplest possible attack — just telling the model "this sounds too AI-like, try again" — has not been benchmarked.

## Recommendations for Our Experiment

### Recommended Datasets
1. **HC3** (primary): Large, diverse, well-established. Human/ChatGPT pairs across multiple domains. Already downloaded.
2. **MAGE** (secondary): Standard benchmark used by AdvPara. Good for comparison with existing results. Small OOD subset downloaded.
3. **RAID** (for comprehensive evaluation): Largest benchmark with diverse attacks/domains. Available via streaming.

### Recommended Baselines
1. **Single-step prompt**: "Write this in a way that doesn't sound AI-generated" (our control condition)
2. **DIPPER paraphrase**: Standard post-hoc evasion baseline
3. **Iterative rejection** (our method): Generate → detect → reject with feedback → regenerate
4. **Vanilla generation**: No evasion attempt (upper bound for detection)

### Recommended Detectors
1. **Fast-DetectGPT**: SOTA zero-shot, widely used benchmark
2. **Binoculars**: SOTA zero-shot, different mechanism
3. **OpenAI RoBERTa classifier**: Standard trained classifier
4. **GPTZero API**: Commercial detector (if accessible)

### Recommended Metrics
1. **TPR@1%FPR**: Primary metric (strictest, most operationally relevant)
2. **AUROC**: Secondary metric (for comparison with existing literature)
3. **Semantic similarity** (P-SP or SBERT): Quality preservation
4. **Perplexity**: Text naturalness
5. **Number of iterations to evasion**: Unique to our study — measures efficiency of iterative rejection

### Methodological Considerations
- Use the same LLM for both generation and iterative refinement (to test the "shifting into different generation regime" hypothesis)
- Control for total compute: compare N iterations of rejection against N independent generations with selection
- Test with and without explicit feedback about why text sounds AI-like
- Measure token-level statistics (entropy, rank) across iterations to understand mechanism
- Use multiple detectors to test transferability of iterative evasion
