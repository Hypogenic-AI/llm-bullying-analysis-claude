"""
LLM Bullying Experiment: Iterative Rejection vs Single-Step Humanization

Tests whether iteratively rejecting LLM output as "too AI-sounding" evades
AI detectors more effectively than single-step humanization prompting.

Conditions:
1. Vanilla - standard generation
2. Single-step humanization - "write naturally"
3. Strong single-step - detailed anti-AI instructions
4. Iterative rejection (5 rounds) - reject & regenerate with feedback
5. Best-of-5 selection - 5 independent generations, pick lowest detector score
"""

import json
import os
import time
import random
import numpy as np
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────

SEED = 42
NUM_QUESTIONS = 50
MAX_REJECTION_ROUNDS = 5
BEST_OF_N = 5
MODEL = "gpt-4.1"
RESULTS_DIR = "results"
DATA_PATH = "datasets/hc3/open_qa.jsonl"

random.seed(SEED)
np.random.seed(SEED)

client = OpenAI()

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_VANILLA = "You are a helpful assistant. Answer the question concisely in 2-4 paragraphs."

SYSTEM_SINGLE_STEP = """You are a helpful assistant. Answer the question concisely in 2-4 paragraphs.
Write in a natural, human-sounding way. Avoid patterns that sound AI-generated."""

SYSTEM_STRONG_SINGLE = """You are a helpful assistant. Answer the question concisely in 2-4 paragraphs.
Write as a real person would — not as an AI assistant. Specifically:
- Vary your sentence length naturally (mix short and long)
- Use contractions (don't, can't, it's)
- Include occasional informal phrasing
- Avoid starting paragraphs with "In conclusion" or "It's important to note"
- Don't use words like "delve", "crucial", "comprehensive", "landscape", "multifaceted"
- Don't use bullet points or numbered lists
- Don't hedge everything or add unnecessary qualifiers
- Have a clear opinion rather than presenting all sides equally
- Use first person occasionally"""

SYSTEM_ITERATIVE = "You are a helpful assistant. Answer the question concisely in 2-4 paragraphs."

REJECTION_MESSAGES = [
    "This sounds too AI-generated. Rewrite it to sound more like a real person wrote it.",
    "Still sounds like AI. Try again — write like an actual human would.",
    "Nope, still reads like ChatGPT output. Make it sound genuinely human.",
    "This still has that AI quality to it. One more try — really make it sound natural.",
    "Getting closer but still detectably AI. Give it one final shot, as human-sounding as possible.",
]


# ── Data Loading ───────────────────────────────────────────────────────────

def load_questions(path, n):
    """Load n questions from HC3 dataset."""
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    # Sample diverse questions
    random.shuffle(entries)
    selected = entries[:n]
    return [
        {
            "question": e["question"],
            "human_answer": e["human_answers"][0] if e["human_answers"] else "",
            "index": e.get("index", i),
        }
        for i, e in enumerate(selected)
    ]


# ── Generation Functions ──────────────────────────────────────────────────

def generate(system, user_msg, conversation_history=None, temperature=1.0):
    """Generate a response from GPT-4.1."""
    if conversation_history:
        messages = [{"role": "system", "content": system}] + conversation_history
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
                seed=SEED,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise e


def run_vanilla(question):
    """Condition 1: Vanilla generation."""
    return generate(SYSTEM_VANILLA, question)


def run_single_step(question):
    """Condition 2: Single-step humanization."""
    return generate(SYSTEM_SINGLE_STEP, question)


def run_strong_single(question):
    """Condition 3: Strong single-step humanization."""
    return generate(SYSTEM_STRONG_SINGLE, question)


def run_iterative_rejection(question, n_rounds=MAX_REJECTION_ROUNDS):
    """Condition 4: Iterative rejection with feedback.

    Returns list of all intermediate outputs (round 0 = initial, round N = final).
    """
    outputs = []

    # Round 0: initial generation
    initial = generate(SYSTEM_ITERATIVE, question)
    outputs.append(initial)

    # Build conversation history for subsequent rounds
    history = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": initial},
    ]

    for i in range(n_rounds):
        rejection_msg = REJECTION_MESSAGES[min(i, len(REJECTION_MESSAGES) - 1)]
        history.append({"role": "user", "content": rejection_msg})

        response = generate(SYSTEM_ITERATIVE, question, conversation_history=history)
        outputs.append(response)

        history.append({"role": "assistant", "content": response})

    return outputs


def run_best_of_n(question, n=BEST_OF_N):
    """Condition 5: Generate N independent samples.

    Returns all N outputs (selection happens after detection scoring).
    """
    outputs = []
    for i in range(n):
        # Use different temperatures for diversity
        temp = 0.9 + 0.2 * random.random()
        output = generate(SYSTEM_VANILLA, question, temperature=temp)
        outputs.append(output)
    return outputs


# ── Main Experiment ────────────────────────────────────────────────────────

def run_experiment():
    """Run all conditions on all questions."""
    print(f"Loading {NUM_QUESTIONS} questions from HC3...")
    questions = load_questions(DATA_PATH, NUM_QUESTIONS)

    results = {
        "config": {
            "model": MODEL,
            "seed": SEED,
            "num_questions": NUM_QUESTIONS,
            "max_rejection_rounds": MAX_REJECTION_ROUNDS,
            "best_of_n": BEST_OF_N,
            "timestamp": datetime.now().isoformat(),
        },
        "questions": [],
    }

    for i, q in enumerate(tqdm(questions, desc="Questions")):
        print(f"\n── Question {i+1}/{NUM_QUESTIONS}: {q['question'][:80]}...")

        entry = {
            "question": q["question"],
            "human_answer": q["human_answer"],
            "index": q["index"],
            "conditions": {},
        }

        # Condition 1: Vanilla
        print("  [1/5] Vanilla...")
        entry["conditions"]["vanilla"] = run_vanilla(q["question"])

        # Condition 2: Single-step humanization
        print("  [2/5] Single-step...")
        entry["conditions"]["single_step"] = run_single_step(q["question"])

        # Condition 3: Strong single-step
        print("  [3/5] Strong single-step...")
        entry["conditions"]["strong_single"] = run_strong_single(q["question"])

        # Condition 4: Iterative rejection
        print("  [4/5] Iterative rejection (5 rounds)...")
        entry["conditions"]["iterative"] = run_iterative_rejection(q["question"])

        # Condition 5: Best-of-5
        print("  [5/5] Best-of-5...")
        entry["conditions"]["best_of_5"] = run_best_of_n(q["question"])

        results["questions"].append(entry)

        # Save incrementally
        with open(f"{RESULTS_DIR}/generation_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGeneration complete. {len(results['questions'])} questions processed.")
    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_experiment()
    print(f"Results saved to {RESULTS_DIR}/generation_results.json")
