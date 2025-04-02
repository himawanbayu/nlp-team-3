import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoConfig
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# model_name = "bert-base-uncased"
# checkpoint_path = "../fine-tuning/best_bert_sciq_model.pt"

# model_name = "albert-base-v2"
# checkpoint_path = "../fine-tuning/best_albert-base-v2_sciq_model.pt"

# model_name = "roberta-base"
# checkpoint_path = "../fine-tuning/best_roberta-base_sciq_model.pt"

# model_name = "distilbert-base-uncased"
# checkpoint_path = "../fine-tuning/best_distilbert-base-uncased_sciq_model.pt"

model_name = "bert-large-uncased"
checkpoint_path = "../fine-tuning/best_bert-large-uncased_sciq_model.pt"

distractors_file = "../generated_distractors/qwen_distractors_filtered.csv"

hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
use_support = False
num_samples = 5  # For MC Dropout

output_dir = f"qwen/{use_support}/{model_name}/{hidden_dropout_prob}"
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(
    model_name,
    hidden_dropout_prob=hidden_dropout_prob,
    attention_probs_dropout_prob=attention_probs_dropout_prob
)
model = AutoModelForMultipleChoice.from_pretrained(model_name, config=config)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.train()  # MC Dropout

df = pd.read_csv(distractors_file)

def prepare_input(support, question, choices):
    prompt = f"{support} {question}" if use_support else question
    choices = [str(c) for c in choices]

    encoded = tokenizer([prompt] * len(choices), choices,
                        padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    input_dict = {
        "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
        "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
    }

    # Only add token_type_ids for BERT-style models
    if "token_type_ids" in encoded:
        input_dict["token_type_ids"] = encoded["token_type_ids"].unsqueeze(0).to(device)

    return input_dict


def evaluate_question(support, question, correct, distractors):
    choices = [correct] + distractors
    correct_idx = 0
    inputs = prepare_input(support, question, choices)
    preds = []

    for _ in range(num_samples):
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = logits.argmax(dim=-1).item()
            preds.append(pred)

    majority = Counter(preds).most_common(1)[0][0]
    accuracy = int(majority == correct_idx)
    difficulty = preds.count(correct_idx) / num_samples
    distractor_effect = {correct: preds.count(0)}
    for i, d in enumerate(distractors, 1):
        distractor_effect[d] = preds.count(i)

    return accuracy, difficulty, distractor_effect, preds

# Evaluation
results = []
for _, row in df.iterrows():
    question = row["question"]
    support = row["support"]
    correct = row["correct_answer"]

    for source, prefix in [("ref", "ref_distractor"), ("sys", "sys_distractor")]:
        distractors = [row[f"{prefix}{i}"] for i in range(1, 4)]
        accuracy, difficulty, effect, preds = evaluate_question(support, question, correct, distractors)

        results.append({
            "question": question,
            "support": support,
            "correct_answer": correct,
            "source": source,
            "distractor1": distractors[0],
            "distractor2": distractors[1],
            "distractor3": distractors[2],
            "accuracy": accuracy,
            "difficulty": difficulty,
            "distractor_effectiveness": effect,
            "raw_preds": preds
        })

source_performance = {}
for result in results:
    source = result["source"]
    accuracy = result["accuracy"]

    if source not in source_performance:
        source_performance[source] = {"total": 0, "correct": 0}

    source_performance[source]["total"] += 1
    source_performance[source]["correct"] += accuracy

plt.figure(figsize=(8, 5))
sources = ["ref", "sys"]
accuracies = [
    source_performance[src]["correct"] / source_performance[src]["total"]
    for src in sources
]

plt.bar(sources, accuracies, color=["skyblue", "salmon"])
plt.xlabel("Distractor Source")
plt.ylabel("Accuracy")
plt.title("Overall Model Performance by Distractor Source")
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.3f}", ha='center')
plt.savefig(f"{output_dir}/overall_performance_by_source.png")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "test_taker_results.csv"), index=False)
print(f"Evaluation complete. Saved to {output_dir}/test_taker_results.csv")