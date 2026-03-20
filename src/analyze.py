"""
Analysis and Visualization Pipeline

Performs statistical analysis and generates plots for the LLM Bullying experiment.
"""

import json
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Linguistic Feature Extraction ──────────────────────────────────────────

AI_MARKERS = [
    "delve", "crucial", "comprehensive", "landscape", "multifaceted",
    "furthermore", "moreover", "it's important to note", "it is important to note",
    "in conclusion", "in summary", "overall", "certainly", "absolutely",
    "encompasses", "facilitate", "utilize", "leverage", "streamline",
    "paramount", "pivotal", "myriad", "plethora", "nuanced",
    "tapestry", "beacon", "bustling", "whimsical", "testament",
]


def extract_features(text):
    """Extract linguistic features from text."""
    if not text or not text.strip():
        return {}

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()

    if not words or not sentences:
        return {}

    # Sentence length stats
    sent_lens = [len(s.split()) for s in sentences]

    # Type-token ratio (vocabulary richness)
    unique_words = set(w.lower().strip(".,!?;:\"'()[]") for w in words)
    ttr = len(unique_words) / len(words) if words else 0

    # Contractions count
    contractions = len(re.findall(r"\b\w+'\w+\b", text))
    contraction_rate = contractions / len(words)

    # First person pronouns
    first_person = len(re.findall(r'\b(I|me|my|mine|myself|we|us|our|ours)\b', text, re.I))
    first_person_rate = first_person / len(words)

    # AI marker count
    text_lower = text.lower()
    marker_count = sum(1 for m in AI_MARKERS if m in text_lower)
    marker_rate = marker_count / len(words) * 100

    # Average word length
    avg_word_len = np.mean([len(w) for w in words])

    # Paragraph count
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": np.mean(sent_lens),
        "sentence_length_std": np.std(sent_lens) if len(sent_lens) > 1 else 0,
        "type_token_ratio": ttr,
        "contraction_rate": contraction_rate,
        "first_person_rate": first_person_rate,
        "ai_marker_count": marker_count,
        "ai_marker_rate": marker_rate,
        "avg_word_length": avg_word_len,
        "paragraph_count": len(paragraphs),
    }


# ── Statistical Tests ──────────────────────────────────────────────────────

def compare_conditions(scores_a, scores_b, name_a, name_b):
    """Paired Wilcoxon signed-rank test between two conditions."""
    assert len(scores_a) == len(scores_b), f"Length mismatch: {len(scores_a)} vs {len(scores_b)}"

    diffs = np.array(scores_a) - np.array(scores_b)
    mean_diff = np.mean(diffs)

    if np.all(diffs == 0):
        return {"test": "wilcoxon", "statistic": 0, "p_value": 1.0,
                "mean_diff": 0, "effect_size": 0,
                "comparison": f"{name_a} vs {name_b}"}

    stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')

    # Effect size: matched-pairs rank-biserial correlation
    n = len(diffs)
    r = 1 - (2 * stat) / (n * (n + 1) / 2)

    return {
        "test": "wilcoxon",
        "comparison": f"{name_a} vs {name_b}",
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "effect_size_r": float(r),
        "n": n,
        "significant_bonferroni": p_value < 0.0125,  # α/4
    }


# ── Visualization ──────────────────────────────────────────────────────────

def plot_condition_comparison(scored_data, detector_name):
    """Bar chart comparing AI detection scores across conditions."""
    conditions = {
        "Human": [],
        "Vanilla": [],
        "Single-step": [],
        "Strong\nsingle-step": [],
        "Iterative\n(final)": [],
        "Best-of-5\n(selected)": [],
    }

    for q in scored_data["questions"]:
        s = q["scores"][detector_name]
        if s.get("human") is not None:
            conditions["Human"].append(s["human"])
        conditions["Vanilla"].append(s["vanilla"])
        conditions["Single-step"].append(s["single_step"])
        conditions["Strong\nsingle-step"].append(s["strong_single"])
        conditions["Iterative\n(final)"].append(s["iterative"][-1])
        conditions["Best-of-5\n(selected)"].append(min(s["best_of_5"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(conditions.keys())
    means = [np.mean(v) if v else 0 for v in conditions.values()]
    stds = [np.std(v) if v else 0 for v in conditions.values()]

    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22', '#3498db', '#9b59b6']
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('P(AI-generated)', fontsize=13)
    ax.set_title(f'AI Detection Scores by Condition ({detector_name.upper()})', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/condition_comparison_{detector_name}.png', dpi=150)
    plt.close()
    print(f"  Saved condition_comparison_{detector_name}.png")


def plot_iteration_trajectory(scored_data, detector_name):
    """Line plot showing how detection scores change across rejection rounds."""
    n_rounds = max(len(q["scores"][detector_name]["iterative"]) for q in scored_data["questions"])

    # Collect per-round scores
    round_scores = [[] for _ in range(n_rounds)]
    for q in scored_data["questions"]:
        for r, score in enumerate(q["scores"][detector_name]["iterative"]):
            round_scores[r].append(score)

    means = [np.mean(rs) for rs in round_scores]
    stds = [np.std(rs) for rs in round_scores]
    rounds = list(range(n_rounds))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot individual trajectories (light)
    for q in scored_data["questions"]:
        scores = q["scores"][detector_name]["iterative"]
        ax.plot(range(len(scores)), scores, alpha=0.1, color='blue', linewidth=0.5)

    # Plot mean trajectory (bold)
    ax.plot(rounds, means, 'b-o', linewidth=2.5, markersize=8, label='Mean', zorder=5)
    ax.fill_between(rounds, np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds), alpha=0.2, color='blue')

    # Reference lines
    vanilla_scores = [q["scores"][detector_name]["vanilla"] for q in scored_data["questions"]]
    single_scores = [q["scores"][detector_name]["single_step"] for q in scored_data["questions"]]
    strong_scores = [q["scores"][detector_name]["strong_single"] for q in scored_data["questions"]]
    human_scores = [q["scores"][detector_name].get("human") for q in scored_data["questions"]
                    if q["scores"][detector_name].get("human") is not None]

    ax.axhline(np.mean(vanilla_scores), color='red', linestyle='--', alpha=0.7, label=f'Vanilla ({np.mean(vanilla_scores):.3f})')
    ax.axhline(np.mean(single_scores), color='orange', linestyle='--', alpha=0.7, label=f'Single-step ({np.mean(single_scores):.3f})')
    ax.axhline(np.mean(strong_scores), color='#e67e22', linestyle=':', alpha=0.7, label=f'Strong single ({np.mean(strong_scores):.3f})')
    if human_scores:
        ax.axhline(np.mean(human_scores), color='green', linestyle='--', alpha=0.7, label=f'Human ({np.mean(human_scores):.3f})')

    ax.set_xlabel('Rejection Round (0 = initial)', fontsize=12)
    ax.set_ylabel('P(AI-generated)', fontsize=12)
    ax.set_title(f'Detection Score Trajectory Across Rejection Rounds ({detector_name.upper()})', fontsize=13)
    ax.set_xticks(rounds)
    ax.set_xticklabels([f'R{r}' for r in rounds])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/iteration_trajectory_{detector_name}.png', dpi=150)
    plt.close()
    print(f"  Saved iteration_trajectory_{detector_name}.png")


def plot_linguistic_features(scored_data):
    """Show how linguistic features change across iterative rejection rounds."""
    feature_names = ["avg_sentence_length", "sentence_length_std", "type_token_ratio",
                     "contraction_rate", "first_person_rate", "ai_marker_count"]
    display_names = ["Avg Sentence Length", "Sentence Length Std", "Type-Token Ratio",
                     "Contraction Rate", "First-Person Rate", "AI Marker Count"]

    n_rounds = max(len(q["texts"]["iterative"]) for q in scored_data["questions"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (feat, display) in enumerate(zip(feature_names, display_names)):
        ax = axes[idx]

        # Per-round feature values
        round_vals = [[] for _ in range(n_rounds)]
        for q in scored_data["questions"]:
            for r, text in enumerate(q["texts"]["iterative"]):
                features = extract_features(text)
                if features and feat in features:
                    round_vals[r].append(features[feat])

        means = [np.mean(rv) if rv else 0 for rv in round_vals]
        stds = [np.std(rv) if rv else 0 for rv in round_vals]

        # Also compute for vanilla and human
        vanilla_vals = [extract_features(q["texts"]["vanilla"]).get(feat, 0)
                       for q in scored_data["questions"]]
        human_vals = [extract_features(q.get("human_answer", "")).get(feat, 0)
                     for q in scored_data["questions"]
                     if q.get("human_answer")]

        ax.plot(range(n_rounds), means, 'b-o', linewidth=2, markersize=6)
        ax.fill_between(range(n_rounds), np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), alpha=0.2, color='blue')

        if vanilla_vals:
            ax.axhline(np.mean(vanilla_vals), color='red', linestyle='--', alpha=0.5, label='Vanilla')
        if human_vals:
            ax.axhline(np.mean(human_vals), color='green', linestyle='--', alpha=0.5, label='Human')

        ax.set_title(display, fontsize=11)
        ax.set_xlabel('Rejection Round')
        ax.set_xticks(range(n_rounds))
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Linguistic Feature Trajectories Across Rejection Rounds', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/linguistic_features.png', dpi=150)
    plt.close()
    print("  Saved linguistic_features.png")


def plot_best_of_n_vs_iterative(scored_data, detector_name):
    """Scatter plot comparing best-of-5 selection vs iterative rejection."""
    iterative_final = []
    bon_selected = []

    for q in scored_data["questions"]:
        s = q["scores"][detector_name]
        iterative_final.append(s["iterative"][-1])
        bon_selected.append(min(s["best_of_5"]))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(bon_selected, iterative_final, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal performance')

    ax.set_xlabel('Best-of-5 Score (P(AI))', fontsize=12)
    ax.set_ylabel('Iterative Rejection Final Score (P(AI))', fontsize=12)
    ax.set_title(f'Best-of-5 vs Iterative Rejection ({detector_name.upper()})', fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Count which is better
    iter_wins = sum(1 for i, b in zip(iterative_final, bon_selected) if i < b)
    bon_wins = sum(1 for i, b in zip(iterative_final, bon_selected) if b < i)
    ax.text(0.05, 0.95, f'Iterative wins: {iter_wins}\nBest-of-5 wins: {bon_wins}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/bon_vs_iterative_{detector_name}.png', dpi=150)
    plt.close()
    print(f"  Saved bon_vs_iterative_{detector_name}.png")


def plot_score_distributions(scored_data, detector_name):
    """Violin/box plots of score distributions."""
    condition_data = {
        "Human": [],
        "Vanilla": [],
        "Single-step": [],
        "Strong single": [],
        "Iterative (R0)": [],
        "Iterative (R2)": [],
        "Iterative (final)": [],
        "Best-of-5 (sel)": [],
    }

    for q in scored_data["questions"]:
        s = q["scores"][detector_name]
        if s.get("human") is not None:
            condition_data["Human"].append(s["human"])
        condition_data["Vanilla"].append(s["vanilla"])
        condition_data["Single-step"].append(s["single_step"])
        condition_data["Strong single"].append(s["strong_single"])
        condition_data["Iterative (R0)"].append(s["iterative"][0])
        if len(s["iterative"]) > 2:
            condition_data["Iterative (R2)"].append(s["iterative"][2])
        condition_data["Iterative (final)"].append(s["iterative"][-1])
        condition_data["Best-of-5 (sel)"].append(min(s["best_of_5"]))

    fig, ax = plt.subplots(figsize=(12, 6))

    data_list = []
    labels = []
    for name, vals in condition_data.items():
        if vals:
            data_list.append(vals)
            labels.append(name)

    parts = ax.violinplot(data_list, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22', '#95a5a6', '#7f8c8d', '#3498db', '#9b59b6']
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('P(AI-generated)', fontsize=12)
    ax.set_title(f'Score Distributions by Condition ({detector_name.upper()})', fontsize=13)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/score_distributions_{detector_name}.png', dpi=150)
    plt.close()
    print(f"  Saved score_distributions_{detector_name}.png")


# ── Main Analysis ──────────────────────────────────────────────────────────

def main():
    # Load scored results
    with open(f"{RESULTS_DIR}/scored_results.json") as f:
        scored = json.load(f)

    analysis = {"detectors": {}, "linguistic": {}}

    for det_name in scored["detectors"]:
        print(f"\n{'='*60}")
        print(f"Analyzing: {det_name.upper()}")
        print('='*60)

        # Collect condition scores
        vanilla = [q["scores"][det_name]["vanilla"] for q in scored["questions"]]
        single = [q["scores"][det_name]["single_step"] for q in scored["questions"]]
        strong = [q["scores"][det_name]["strong_single"] for q in scored["questions"]]
        iter_final = [q["scores"][det_name]["iterative"][-1] for q in scored["questions"]]
        bon_selected = [min(q["scores"][det_name]["best_of_5"]) for q in scored["questions"]]
        human = [q["scores"][det_name].get("human") for q in scored["questions"]
                 if q["scores"][det_name].get("human") is not None]

        print(f"\nCondition means (P(AI)):")
        print(f"  Human:          {np.mean(human):.4f} ± {np.std(human):.4f}" if human else "  Human: N/A")
        print(f"  Vanilla:        {np.mean(vanilla):.4f} ± {np.std(vanilla):.4f}")
        print(f"  Single-step:    {np.mean(single):.4f} ± {np.std(single):.4f}")
        print(f"  Strong single:  {np.mean(strong):.4f} ± {np.std(strong):.4f}")
        print(f"  Iterative (fin):{np.mean(iter_final):.4f} ± {np.std(iter_final):.4f}")
        print(f"  Best-of-5 (sel):{np.mean(bon_selected):.4f} ± {np.std(bon_selected):.4f}")

        # Statistical comparisons
        print(f"\nStatistical Tests (Wilcoxon signed-rank, α=0.0125 after Bonferroni):")
        tests = []

        # H1: Iterative vs single-step
        t1 = compare_conditions(single, iter_final, "single_step", "iterative_final")
        tests.append(t1)
        print(f"  Single-step vs Iterative: p={t1['p_value']:.4e}, r={t1['effect_size_r']:.3f}, "
              f"{'SIGNIFICANT' if t1['significant_bonferroni'] else 'not significant'}")

        # H1b: Strong single vs iterative
        t2 = compare_conditions(strong, iter_final, "strong_single", "iterative_final")
        tests.append(t2)
        print(f"  Strong-single vs Iterative: p={t2['p_value']:.4e}, r={t2['effect_size_r']:.3f}, "
              f"{'SIGNIFICANT' if t2['significant_bonferroni'] else 'not significant'}")

        # H2: Iterative vs best-of-5
        t3 = compare_conditions(bon_selected, iter_final, "best_of_5", "iterative_final")
        tests.append(t3)
        print(f"  Best-of-5 vs Iterative: p={t3['p_value']:.4e}, r={t3['effect_size_r']:.3f}, "
              f"{'SIGNIFICANT' if t3['significant_bonferroni'] else 'not significant'}")

        # Vanilla vs iterative
        t4 = compare_conditions(vanilla, iter_final, "vanilla", "iterative_final")
        tests.append(t4)
        print(f"  Vanilla vs Iterative: p={t4['p_value']:.4e}, r={t4['effect_size_r']:.3f}, "
              f"{'SIGNIFICANT' if t4['significant_bonferroni'] else 'not significant'}")

        analysis["detectors"][det_name] = {
            "means": {
                "human": float(np.mean(human)) if human else None,
                "vanilla": float(np.mean(vanilla)),
                "single_step": float(np.mean(single)),
                "strong_single": float(np.mean(strong)),
                "iterative_final": float(np.mean(iter_final)),
                "best_of_5_selected": float(np.mean(bon_selected)),
            },
            "stds": {
                "human": float(np.std(human)) if human else None,
                "vanilla": float(np.std(vanilla)),
                "single_step": float(np.std(single)),
                "strong_single": float(np.std(strong)),
                "iterative_final": float(np.std(iter_final)),
                "best_of_5_selected": float(np.std(bon_selected)),
            },
            "tests": tests,
        }

        # Per-round iteration analysis
        n_rounds = max(len(q["scores"][det_name]["iterative"]) for q in scored["questions"])
        round_means = []
        for r in range(n_rounds):
            vals = [q["scores"][det_name]["iterative"][r] for q in scored["questions"]
                    if r < len(q["scores"][det_name]["iterative"])]
            round_means.append(float(np.mean(vals)))
        analysis["detectors"][det_name]["iteration_means"] = round_means
        print(f"\n  Per-round means: {[f'{m:.3f}' for m in round_means]}")

        # Generate plots
        print("\n  Generating plots...")
        plot_condition_comparison(scored, det_name)
        plot_iteration_trajectory(scored, det_name)
        plot_best_of_n_vs_iterative(scored, det_name)
        plot_score_distributions(scored, det_name)

    # Linguistic analysis
    print(f"\n{'='*60}")
    print("Linguistic Feature Analysis")
    print('='*60)
    plot_linguistic_features(scored)

    # Compute per-condition linguistic features
    for cond_name in ["vanilla", "single_step", "strong_single"]:
        feats = [extract_features(q["texts"][cond_name]) for q in scored["questions"]]
        feats = [f for f in feats if f]
        if feats:
            analysis["linguistic"][cond_name] = {
                k: float(np.mean([f[k] for f in feats]))
                for k in feats[0].keys()
            }

    # Iterative per-round
    n_rounds = max(len(q["texts"]["iterative"]) for q in scored["questions"])
    for r in range(n_rounds):
        feats = []
        for q in scored["questions"]:
            if r < len(q["texts"]["iterative"]):
                f = extract_features(q["texts"]["iterative"][r])
                if f:
                    feats.append(f)
        if feats:
            analysis["linguistic"][f"iterative_round_{r}"] = {
                k: float(np.mean([f[k] for f in feats]))
                for k in feats[0].keys()
            }

    # Human baseline
    human_feats = [extract_features(q.get("human_answer", "")) for q in scored["questions"]
                   if q.get("human_answer")]
    human_feats = [f for f in human_feats if f]
    if human_feats:
        analysis["linguistic"]["human"] = {
            k: float(np.mean([f[k] for f in human_feats]))
            for k in human_feats[0].keys()
        }

    # Print linguistic summary
    print("\nLinguistic features by condition:")
    key_feats = ["avg_sentence_length", "sentence_length_std", "type_token_ratio",
                 "contraction_rate", "first_person_rate", "ai_marker_count"]
    header = f"{'Condition':25s}" + "".join(f"{f:>20s}" for f in key_feats)
    print(header)
    for cond in ["human", "vanilla", "single_step", "strong_single"] + [f"iterative_round_{r}" for r in range(n_rounds)]:
        if cond in analysis["linguistic"]:
            vals = analysis["linguistic"][cond]
            row = f"{cond:25s}" + "".join(f"{vals.get(f, 0):>20.4f}" for f in key_feats)
            print(row)

    # Save analysis (convert numpy types)
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)

    with open(f"{RESULTS_DIR}/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, cls=NpEncoder)
    print(f"\nAnalysis saved to {RESULTS_DIR}/analysis.json")


if __name__ == "__main__":
    main()
