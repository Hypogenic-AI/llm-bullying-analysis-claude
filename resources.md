# Resources Catalog

## Summary
This document catalogs all resources gathered for the "LLM Bullying" research project — investigating whether iteratively rejecting LLM output as "too AI-sounding" evades AI detectors more effectively than single-step prompting.

## Papers
Total papers downloaded: 15

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| SICO: Guided Evasion of AI Text Detection | Lu et al. | 2024 | papers/2305.10847_SICO_guided_evasion.pdf | Prompt optimization via iterative substitution against proxy detector |
| CoPA: Contrastive Paraphrase Attacks | Fang et al. | 2025 | papers/2505.15337_CoPA_contrastive_paraphrase.pdf | Training-free contrastive decoding subtracting machine distributions |
| Adversarial Paraphrasing | Cheng et al. | 2025 | papers/2506.07001_adversarial_paraphrasing.pdf | Detector-guided token selection during paraphrasing; universal transfer |
| Self-Disguise Attack (SDA) | Zhou et al. | 2025 | papers/2508.15848_self_disguise_attack.pdf | Iterative adversarial feature extraction loop + RAG retrieval |
| Humanizing Machine-Generated Content (HMGC) | Zhou et al. | 2024 | papers/2404.01907_humanizing_machine_content.pdf | Word-level adversarial substitution with gradient + perplexity guidance |
| Paraphrasing Evades Detectors (DIPPER) | Krishna et al. | 2023 | papers/2303.13408_paraphrasing_evades_detectors.pdf | 11B controllable paraphraser; foundational evasion work + retrieval defense |
| DetectGPT | Mitchell et al. | 2023 | papers/2301.11305_DetectGPT.pdf | Zero-shot detection via probability curvature |
| Fast-DetectGPT | Bao et al. | 2024 | papers/2310.05130_Fast_DetectGPT.pdf | 340x faster DetectGPT via conditional probability curvature |
| Binoculars | Hans et al. | 2024 | papers/2401.12070_Binoculars.pdf | Zero-shot detection contrasting two related LLMs |
| HC3 Dataset Paper | Guo et al. | 2023 | papers/2301.07597_HC3_dataset.pdf | Human ChatGPT Comparison Corpus |
| HC3 Plus | Various | 2023 | papers/2309.02731_HC3_Plus.pdf | Extended HC3 with translation/summarization/paraphrasing |
| Self-Refine | Madaan et al. | 2023 | papers/2303.17651_Self_Refine.pdf | Iterative self-feedback refinement framework |
| PADBen | Various | 2025 | papers/2511.00416_PADBen_benchmark.pdf | Benchmark for paraphrase attacks against detectors |
| RAID Benchmark | Dugan et al. | 2024 | papers/2405.07940_RAID_benchmark.pdf | Largest detection benchmark: 10M+ generations |
| Can AI Text Be Reliably Detected? | Sadasivan et al. | 2023 | papers/2303.11156_can_AI_text_reliably_detected.pdf | Theoretical impossibility result for detection |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| HC3 | HuggingFace (Hello-SimpleAI/HC3) | 24K pairs, ~140MB | Human vs ChatGPT QA | datasets/hc3/ | Full download. 6 domain splits. |
| MAGE (OOD subset) | HuggingFace (yaful/MAGE) | 1,562 texts, 2MB | AI text classification | datasets/mage/ | OOD test set only. 800 human + 762 AI. |
| RAID | HuggingFace (liamdugan/raid) | 10M+ generations, ~2GB | Multi-model detection | datasets/ (streaming) | Too large for full download. Samples saved. |

See datasets/README.md for download instructions and loading code.

## Code Repositories
Total repositories cloned: 7

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| SICO | github.com/ColinLu50/Evade-GPT-Detector | Prompt optimization evasion | code/SICO/ | Requires OpenAI API |
| CoPA | github.com/ffhibnese/CoPA_Contrastive_Paraphrase_Attacks | Contrastive decoding attack | code/CoPA/ | Training-free, needs Qwen2.5-72B |
| AdvPara | github.com/chengez/Adversarial-Paraphrasing | Detector-guided paraphrasing | code/AdvPara/ | Training-free, LLaMA-3-8B |
| HMGC | github.com/zhouying20/HMGC | Word-level adversarial attack | code/HMGC/ | Gradient-based, needs surrogate training |
| Binoculars | github.com/ahans30/Binoculars | Zero-shot AI text detector | code/Binoculars/ | Installable package, Falcon-7B |
| Fast-DetectGPT | github.com/baoguangsheng/fast-detect-gpt | Zero-shot detector (ICLR 2024) | code/Fast-DetectGPT/ | Needs LLaMA-3-8B scoring model |
| RAID | github.com/liamdugan/raid | Detection benchmark framework | code/RAID/ | pip install raid-bench; wraps 10+ detectors |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with multiple queries covering: iterative prompting, AI detection evasion, paraphrasing attacks, AI text detection benchmarks
2. Web searches on arXiv, Semantic Scholar, Papers with Code for recent (2023-2025) work
3. Followed citation chains from key papers (DIPPER → SICO → CoPA → AdvPara)
4. Searched HuggingFace for standard datasets (HC3, MAGE, RAID)
5. Identified GitHub repos from paper code availability sections

### Selection Criteria
- **Papers**: Prioritized work on iterative/feedback-based evasion methods, standard detection baselines, and benchmark papers
- **Datasets**: Selected datasets used across multiple papers for comparability; chose datasets with human/AI text pairs
- **Code**: Cloned repos with working implementations that could serve as baselines or evaluation tools

### Challenges Encountered
- HC3 dataset's HuggingFace loading script is deprecated; required direct JSONL download
- RAID dataset is very large (~2GB); only samples downloaded, full streaming recommended
- Some commercial detectors (GPTZero) require API access not available in this environment

### Gaps and Workarounds
- No existing implementation of simple iterative rejection (our novel contribution)
- GPTZero and other commercial detectors cannot be evaluated without API access; use open-source alternatives (Binoculars, Fast-DetectGPT)
- DIPPER model (11B parameters) may be too large to run locally; consider lighter alternatives or API-based approaches

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **HC3** for initial experiments: well-established, diverse domains, manageable size
- **MAGE OOD subset** for validation: standard benchmark used by AdvPara/CoPA papers
- Use RAID infrastructure for comprehensive evaluation if compute allows

### 2. Baseline Methods
- **Vanilla generation**: Standard LLM output (no evasion)
- **Single-step humanization**: "Write this so it doesn't sound AI-generated"
- **Iterative rejection** (our method): Generate → detect → reject with specific feedback → regenerate
- **DIPPER paraphrase**: Standard post-hoc baseline (if compute allows 11B model)

### 3. Evaluation Metrics
- **TPR@1%FPR**: Primary (strictest threshold, used by AdvPara/DIPPER papers)
- **AUROC**: Secondary (used by SICO, standard comparison)
- **Semantic similarity** (SBERT): Quality preservation
- **Perplexity**: Text naturalness
- **Iterations-to-evasion**: Novel metric specific to our study

### 4. Code to Adapt/Reuse
- **RAID benchmark** (`code/RAID/`): Standardized evaluation harness wrapping multiple detectors
- **Binoculars** (`code/Binoculars/`): Easy-to-use detector with simple API
- **Fast-DetectGPT** (`code/Fast-DetectGPT/`): SOTA zero-shot detector
- **SICO** (`code/SICO/`): Reference implementation for comparison with iterative approach
