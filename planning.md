# Research Plan: LLM Bullying — Iterative Rejection for AI Detection Evasion

## Motivation & Novelty Assessment

### Why This Research Matters
AI text detectors are widely deployed in education, publishing, and hiring. If a trivially simple technique—repeatedly telling an LLM "this sounds too AI-like"—can reliably evade these detectors, it has major implications for detector trustworthiness. Understanding *why* iterative rejection works (feature activation vs. generation regime shift) could inform both better detectors and better alignment techniques.

### Gap in Existing Work
Per literature_review.md, all existing iterative evasion methods operate at the *training/optimization* level (SICO iterates on prompt examples, SDA iterates on extracted features, CoPA/AdvPara modify the decoding process). **No paper tests the simplest possible attack: generate → human says "too AI" → regenerate, repeated N times.** Additionally, no study compares single-step humanization prompts against multi-step rejection with feedback.

### Our Novel Contribution
We are the first to:
1. Benchmark simple iterative rejection (no special decoding, no trained paraphrasers) against single-step humanization prompting
2. Distinguish between *selection effect* (best-of-N) and *contextual steering* (rejection feedback) as mechanisms
3. Track per-iteration detector scores and linguistic features to understand the mechanism

### Experiment Justification
- **Experiment 1 (Iterative Rejection vs Single-Step):** Tests the core hypothesis directly. Is iterative rejection more effective than a single strong humanization prompt?
- **Experiment 2 (Best-of-N Control):** Controls for selection effect. If generating N independent samples and picking the lowest-scoring one works equally well, then iterative rejection is just selection, not a regime shift.
- **Experiment 3 (Linguistic Analysis):** Tests the *mechanism*. If the model shifts generation regimes, we should see systematic changes in linguistic features (sentence structure, vocabulary, hedging patterns) across iterations.

## Research Question
Does iteratively rejecting an LLM's output as "too AI-sounding" lead to text that evades AI detectors more effectively than single-step humanization prompting, and if so, is this due to contextual steering (the rejection activating different generation features) or simple selection pressure (best-of-N sampling)?

## Hypothesis Decomposition
- **H1**: Iterative rejection (5 rounds) produces text with significantly lower AI detection scores than single-step humanization prompting.
- **H2**: Iterative rejection outperforms best-of-N independent sampling (controlling for number of generations), suggesting the rejection context itself contributes beyond selection.
- **H3**: Linguistic features (sentence length variance, type-token ratio, hedging markers, first-person usage) shift systematically across rejection iterations, indicating a generation regime change rather than random variation.

## Proposed Methodology

### Approach
Use GPT-4.1 via OpenAI API to generate answers to 50 HC3 questions under 5 conditions. Score all outputs with two detectors: RoBERTa-based classifier (fast, trained) and Binoculars (SOTA zero-shot). Analyze detection scores and linguistic features.

### Conditions
1. **Vanilla**: Standard generation, no evasion instructions
2. **Single-step humanization**: System prompt asking model to write naturally, avoid AI patterns
3. **Strong single-step**: Detailed prompt listing specific AI markers to avoid
4. **Iterative rejection (5 rounds)**: Generate → "this sounds too AI-like, try again" → regenerate, tracking each intermediate output
5. **Best-of-5 selection**: Generate 5 independent vanilla outputs, select lowest AI-detection score

### Baselines
- Vanilla generation (upper bound for detection)
- Human text from HC3 (lower bound / calibration)

### Evaluation Metrics
- **Primary**: Detector AI-probability score (continuous, averaged across samples)
- **Secondary**: TPR@5%FPR (binary classification performance)
- **Quality**: Semantic similarity to vanilla output (SBERT cosine similarity)
- **Mechanism**: Per-iteration score trajectories, linguistic feature trends

### Statistical Analysis Plan
- Paired Wilcoxon signed-rank tests for condition comparisons (non-parametric, handles non-normal detector scores)
- Bonferroni correction for multiple comparisons (4 pairwise tests → α = 0.0125)
- Effect sizes: matched-pairs rank-biserial correlation
- Linear mixed-effects analysis for iteration trajectories

## Expected Outcomes
- H1 supported: Iterative rejection scores drop significantly below single-step by round 3-5
- H2 partially supported: Iterative rejection may outperform best-of-N slightly, but best-of-N may capture much of the effect (selection accounts for a substantial portion)
- H3: Expect increased sentence length variance, more first-person pronouns, fewer "certainly/importantly/furthermore" markers across iterations

## Timeline
1. Environment setup & data prep: 15 min
2. Implement generation pipeline: 30 min
3. Implement detection pipeline: 30 min
4. Run experiments: 60 min (API calls + local detection)
5. Analysis & visualization: 30 min
6. Documentation: 20 min

## Potential Challenges
- API rate limits for GPT-4.1 → use backoff; 50 questions × 5 conditions × up to 5 iterations = ~750 API calls, manageable
- Detector model loading → A6000 has ample VRAM
- Binoculars requires Falcon-7B models → download during setup

## Success Criteria
- Clear quantitative comparison across all conditions with statistical tests
- Per-iteration score trajectories showing convergence pattern
- Linguistic feature analysis providing mechanistic insight
- Results documented in REPORT.md with visualizations
