import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from accelerate import dispatch_model

dataset = load_dataset("allenai/sciq")

# model_id = "google/gemma-3-1b-it"
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# SBERT evaluation
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_distractors(context, question, correct_answer):
    prompt = f"""
    Given the following multiple-choice question:
    Context: "{context}"
    Question: "{question}"
    Correct Answer: "{correct_answer}"

    Generate three plausible but incorrect answer choices:
    """
    return prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def do_inference(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def evaluate_bleu(reference_distractors, generated_distractors):
    return sentence_bleu([reference_distractors], generated_distractors)


def evaluate_rouge(reference_distractors, generated_distractors):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, gen)["rougeL"].fmeasure for ref, gen in
              zip(reference_distractors, generated_distractors)]
    return np.mean(scores) if scores else 0


def evaluate_sbert_similarity(correct_answer, generated_distractors):
    embeddings = sbert_model.encode([correct_answer] + generated_distractors, convert_to_tensor=True)
    correct_embedding = embeddings[0]
    similarities = [util.cos_sim(correct_embedding, emb).item() for emb in embeddings[1:]]
    return np.mean(similarities)

num_samples = 100  # example, need to be adjusted
bleu_scores, rouge_scores, sbert_similarities = [], [], []

for idx, sample in enumerate(dataset["train"].select(range(num_samples))):
    if not isinstance(sample, dict):
        print(f"Unexpected sample format at index {idx}: {sample}")
        continue

    context, question, correct_answer = sample["support"], sample["question"], sample["correct_answer"]
    reference_distractors = [sample["distractor1"], sample["distractor2"], sample["distractor3"]]

    # Generate Distractors
    prompt = generate_distractors(context, question, correct_answer)
    generated_text = do_inference(model, tokenizer, prompt, device)

    # Extract generated distractors
    generated_distractors = generated_text.strip().split("\n")[:3]

    # Evaluate the generated distractors
    bleu = evaluate_bleu(reference_distractors, generated_distractors)
    rouge = evaluate_rouge(reference_distractors, generated_distractors)
    sbert_sim = evaluate_sbert_similarity(correct_answer, generated_distractors)

    bleu_scores.append(bleu)
    rouge_scores.append(rouge)
    sbert_similarities.append(sbert_sim)

    print(f"Sample {idx + 1}/{num_samples} processed.")

bleu_scores = np.array(bleu_scores)
rouge_scores = np.array(rouge_scores)
sbert_similarities = np.array(sbert_similarities)

def plot_metrics(bleu_scores, rouge_scores, sbert_similarities):
    """Plot BLEU, ROUGE, and SBERT similarity scores."""
    plt.figure(figsize=(10, 5))

    # BLEU Score Distribution
    plt.subplot(1, 3, 1)
    plt.hist(bleu_scores, bins=20, color="blue", alpha=0.7)
    plt.title("BLEU Score Distribution")
    plt.xlabel("BLEU Score")
    plt.ylabel("Frequency")

    # ROUGE Score Distribution
    plt.subplot(1, 3, 2)
    plt.hist(rouge_scores, bins=20, color="red", alpha=0.7)
    plt.title("ROUGE Score Distribution")
    plt.xlabel("ROUGE-L Score")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.hist(sbert_similarities, bins=20, color="green", alpha=0.7)
    plt.title("SBERT Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("plots/bleu-gemma-2b.png")

plot_metrics(bleu_scores, rouge_scores, sbert_similarities)
