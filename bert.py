import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMultipleChoice, AutoTokenizer, AutoModelForMultipleChoice
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# model_name = "bert-large-uncased"
model_name = "bert-base-uncased"
# model_name = "roberta-base"
# model_name = "distilbert-base-uncased"
# model_name = "albert-base-v2"

output_dir = f"cohort_plots_new/{model_name}"
os.makedirs(output_dir, exist_ok=True)
llama_df = pd.read_csv("generated_distractors/llama3.2_1B_generated_distractors.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
model.train()  # Monte Carlo Dropout

sbert = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

bleu_scores, rouge_l_scores, sbert_scores = [], [], []

for _, row in llama_df.iterrows():
    ref_distractors = [row[f"ref_distractor{i}"] for i in range(1, 4)]
    sys_distractors = [row[f"sys_distractor{i}"] for i in range(1, 4)]

    for ref, sys in zip(ref_distractors, sys_distractors):
        bleu = sentence_bleu([ref.split()], sys.split())
        rouge_l = rouge.score(ref, sys)["rougeL"].fmeasure
        emb1 = sbert.encode(sys, convert_to_tensor=True)
        emb2 = sbert.encode(ref, convert_to_tensor=True)
        sbert_sim = util.pytorch_cos_sim(emb1, emb2).item()

        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)
        sbert_scores.append(sbert_sim)

pd.DataFrame({
    "BLEU": bleu_scores,
    "ROUGE-L": rouge_l_scores,
    "SBERT": sbert_scores
}).to_csv(f"{output_dir}/bert_intrinsic_scores.csv", index=False)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(bleu_scores, bins=20, color="skyblue")
plt.title("BLEU Score Distribution")

plt.subplot(1, 3, 2)
plt.hist(rouge_l_scores, bins=20, color="salmon")
plt.title("ROUGE-L Distribution")

plt.subplot(1, 3, 3)
plt.hist(sbert_scores, bins=20, color="lightgreen")
plt.title("SBERT Similarity Distribution")

plt.tight_layout()
plt.savefig(f"{output_dir}/intrinsic_score_distributions.png")

def prepare_mc_input(question, choices):
    encoded = tokenizer(
        [question] * len(choices), choices,
        padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
        "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device)
    }

results = []
num_samples = 5

for _, row in llama_df.iterrows():
    question = row["question"]
    correct = row["correct_answer"]

    for source, prefix in [("ref", "ref_distractor"), ("sys", "sys_distractor")]:
        distractors = [row[f"{prefix}{i}"] for i in range(1, 4)]
        choices = [correct] + distractors
        choices = list(set(choices))[:4]
        choices = sorted(choices)
        correct_idx = choices.index(correct)

        inputs = prepare_mc_input(question, choices)
        preds = []

        for _ in range(num_samples):
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = logits.argmax(dim=-1).item()
                preds.append(pred)

        majority = Counter(preds).most_common(1)[0][0]
        accuracy = int(majority == correct_idx)
        difficulty = preds.count(correct_idx) / num_samples
        distractor_effect = {choices[i]: preds.count(i) for i in range(len(choices))}

        results.append({
            "question": question,
            "source": source,
            "accuracy": accuracy,
            "difficulty": difficulty,
            "distractor_effectiveness": distractor_effect
        })

pd.DataFrame(results).to_csv(f"{output_dir}/bert_extrinsic_scores.csv", index=False)
print("Evaluation complete")
