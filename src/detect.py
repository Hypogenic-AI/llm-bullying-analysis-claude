"""
AI Text Detection Pipeline

Scores generated text using two detectors:
1. RoBERTa-based OpenAI detector (trained classifier)
2. Binoculars (SOTA zero-shot detector)

Outputs detection scores for all experimental conditions.
"""

import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"


# ── Detector 1: RoBERTa-based OpenAI Detector ─────────────────────────────

class RobertaDetector:
    """OpenAI's RoBERTa-based AI text detector."""

    def __init__(self, model_name="openai-community/roberta-base-openai-detector"):
        print(f"Loading RoBERTa detector on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def score(self, text):
        """Return P(AI-generated) for the text. Higher = more likely AI."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # Label 0 = "Fake" (AI-generated), Label 1 = "Real" (human)
        ai_prob = probs[0, 0].item()  # P(AI)
        return ai_prob

    def score_batch(self, texts, batch_size=16):
        """Score multiple texts."""
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(DEVICE)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, 0].cpu().tolist())  # Label 0 = Fake/AI
        return scores


# ── Detector 2: Binoculars ─────────────────────────────────────────────────

class BinocularsDetector:
    """Binoculars zero-shot detector using two related LLMs.

    Implements the exact formula from Hans et al. (2024):
    score = performer_perplexity / cross_entropy(observer→performer)

    Lower score = more likely AI-generated.
    We convert to P(AI) for consistency with other detectors.
    """

    def __init__(self):
        from transformers import AutoModelForCausalLM
        print("Loading Binoculars models (Falcon-7B and Falcon-7B-Instruct)...")
        observer_name = "tiiuae/falcon-7b"
        performer_name = "tiiuae/falcon-7b-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_name, torch_dtype=torch.bfloat16, device_map="cuda:1"
        )
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_name, torch_dtype=torch.bfloat16, device_map="cuda:2"
        )
        self.observer.eval()
        self.performer.eval()

        # Thresholds from paper (bfloat16, Falcon-7B)
        self.threshold = 0.9015310749276843  # accuracy-optimized

        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    @torch.inference_mode()
    def score(self, text):
        """Return P(AI-generated). Higher = more likely AI.

        Computes actual Binoculars score (ppl/x_ppl), then converts:
        score < threshold → AI → high P(AI).
        """
        encoding = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
            return_token_type_ids=False,
        )

        # Get logits from both models
        observer_logits = self.observer(**encoding.to("cuda:1")).logits
        performer_logits = self.performer(**encoding.to("cuda:2")).logits

        # Performer perplexity (standard CE loss)
        shifted_logits = performer_logits[..., :-1, :].contiguous()
        shifted_labels = encoding.input_ids.to(performer_logits.device)[..., 1:].contiguous()
        shifted_mask = encoding.attention_mask.to(performer_logits.device)[..., 1:].contiguous()

        ppl = (self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_mask).sum(1) / shifted_mask.sum(1)
        ppl = ppl.cpu().float().numpy()

        # Cross-entropy: observer probs, performer logits
        vocab_size = observer_logits.shape[-1]
        total_tokens = observer_logits.shape[-2]

        p_proba = self.softmax_fn(observer_logits.to("cuda:1")).view(-1, vocab_size)
        q_scores = performer_logits.to("cuda:1").view(-1, vocab_size)

        ce = self.ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens)
        padding_mask = (encoding.input_ids.to("cuda:1") != self.pad_token_id).type(torch.uint8)
        x_ppl = ((ce * padding_mask).sum(1) / padding_mask.sum(1)).cpu().float().numpy()

        # Binoculars score
        bino_score = float(ppl[0] / x_ppl[0]) if x_ppl[0] != 0 else 0.5

        # Convert to P(AI): score < threshold → AI
        # Use sigmoid centered on threshold, steepness=30
        p_ai = 1.0 / (1.0 + np.exp(30 * (bino_score - self.threshold)))
        return float(p_ai)


# ── Lightweight alternative: Log-likelihood ratio ──────────────────────────

class LogRankDetector:
    """Simple log-rank based detector using a small GPT-2 model.

    Faster alternative to Binoculars. AI text tends to have lower
    average token rank (more predictable).
    """

    def __init__(self):
        from transformers import AutoModelForCausalLM
        print("Loading GPT-2 for log-rank detection...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def score(self, text):
        """Return average log-rank score. Higher = more likely AI.

        AI text has lower average token rank (more predictable).
        We normalize to [0,1] range.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] < 2:
            return 0.5

        outputs = self.model(input_ids)
        logits = outputs.logits[:, :-1, :]  # predict next token
        target_ids = input_ids[:, 1:]

        # Get rank of each actual token in the predicted distribution
        sorted_indices = logits.argsort(dim=-1, descending=True)
        ranks = (sorted_indices == target_ids.unsqueeze(-1)).nonzero(as_tuple=True)[-1]
        avg_log_rank = torch.log(ranks.float() + 1).mean().item()

        # Normalize: lower log-rank = more predictable = more likely AI
        # Typical human text: avg_log_rank ~ 4-6
        # Typical AI text: avg_log_rank ~ 2-4
        ai_prob = 1.0 / (1.0 + np.exp(1.5 * (avg_log_rank - 3.5)))
        return float(ai_prob)


# ── Score all results ──────────────────────────────────────────────────────

def score_all_results(results, detectors):
    """Score all generated text with all detectors."""
    scored = {
        "config": results["config"],
        "detectors": list(detectors.keys()),
        "questions": [],
    }

    for q_data in tqdm(results["questions"], desc="Scoring questions"):
        q_scored = {
            "question": q_data["question"],
            "human_answer": q_data["human_answer"],
            "index": q_data["index"],
            "scores": {},
        }

        for det_name, detector in detectors.items():
            q_scored["scores"][det_name] = {}

            # Score human answer for calibration
            if q_data["human_answer"]:
                q_scored["scores"][det_name]["human"] = detector.score(q_data["human_answer"])

            # Score each condition
            conditions = q_data["conditions"]

            # Vanilla, single_step, strong_single are single texts
            for cond in ["vanilla", "single_step", "strong_single"]:
                text = conditions[cond]
                q_scored["scores"][det_name][cond] = detector.score(text)

            # Iterative: score each round
            iterative_scores = []
            for text in conditions["iterative"]:
                iterative_scores.append(detector.score(text))
            q_scored["scores"][det_name]["iterative"] = iterative_scores

            # Best-of-5: score each candidate
            bon_scores = []
            for text in conditions["best_of_5"]:
                bon_scores.append(detector.score(text))
            q_scored["scores"][det_name]["best_of_5"] = bon_scores

        # Also store the texts for linguistic analysis
        q_scored["texts"] = q_data["conditions"]

        scored["questions"].append(q_scored)

    return scored


def main():
    # Load generation results
    gen_path = f"{RESULTS_DIR}/generation_results.json"
    print(f"Loading generation results from {gen_path}...")
    with open(gen_path) as f:
        results = json.load(f)

    # Initialize detectors
    detectors = {}

    print("\n── Loading detectors ──")
    detectors["roberta"] = RobertaDetector()

    # Try Binoculars, fall back to LogRank
    try:
        detectors["binoculars"] = BinocularsDetector()
    except Exception as e:
        print(f"Binoculars failed to load: {e}")
        print("Falling back to LogRank detector...")
        detectors["logrank"] = LogRankDetector()

    # Score everything
    print("\n── Scoring all outputs ──")
    scored = score_all_results(results, detectors)

    # Save
    out_path = f"{RESULTS_DIR}/scored_results.json"
    with open(out_path, "w") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"\nScored results saved to {out_path}")

    # Print summary
    print("\n── Quick Summary ──")
    for det_name in scored["detectors"]:
        print(f"\n{det_name.upper()} detector:")
        for cond in ["human", "vanilla", "single_step", "strong_single"]:
            vals = [q["scores"][det_name].get(cond) for q in scored["questions"]
                    if q["scores"][det_name].get(cond) is not None]
            if vals:
                print(f"  {cond:20s}: mean={np.mean(vals):.3f} ± {np.std(vals):.3f}")

        # Iterative: show per-round means
        for r in range(MAX_REJECTION_ROUNDS + 1):
            vals = [q["scores"][det_name]["iterative"][r]
                    for q in scored["questions"]
                    if r < len(q["scores"][det_name]["iterative"])]
            if vals:
                print(f"  iterative_round_{r:d}   : mean={np.mean(vals):.3f} ± {np.std(vals):.3f}")

        # Best-of-5: show min (selected) score
        vals = [min(q["scores"][det_name]["best_of_5"]) for q in scored["questions"]]
        print(f"  best_of_5 (selected) : mean={np.mean(vals):.3f} ± {np.std(vals):.3f}")


MAX_REJECTION_ROUNDS = 5

if __name__ == "__main__":
    main()
