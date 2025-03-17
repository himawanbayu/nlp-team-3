from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_distractors(
    context, question, answer, instruction, model, tokenizer, max_seq_length=1024
):
    """Generates distractors for a given question-answer pair."""
    model.eval()
    prompt = f"""{instruction}
    Context: '{context}'
    Question: '{question}'
    Answer: '{answer}'.
    Provide only three distractors as a comma-separated list. Do not include explanations, commentary, or additional text."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    ).to("cuda")

    generation_config = GenerationConfig(
        max_new_tokens=128,
        use_cache=True,
        temperature=0.7,
    )

    outputs = model.generate(**inputs, generation_config=generation_config)
    distractors_text = (
        tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt) :]
        .strip()
        .split("\n")[0]
    )
    distractors = [d.strip() for d in distractors_text.split(",") if d.strip()]
    while len(distractors) < 3:
        distractors.append("")
    return distractors[:3]


def generate_all(ds):
    # Load model and tokenizer
    model_path = "BirendraSharma/llama3.2_1B_distractors_generation"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    instruction = "Generate plausible but incorrect answer choices."

    df = pd.DataFrame()
    for example in tqdm(ds):
        distractors = generate_distractors(
            example["support"],
            example["question"],
            example["correct_answer"],
            instruction,
            model,
            tokenizer,
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "question": example["question"],
                            "support": example["support"],
                            "correct_answer": example["correct_answer"],
                            "ref_distractor1": example["distractor1"],
                            "ref_distractor2": example["distractor2"],
                            "ref_distractor3": example["distractor3"],
                            "sys_distractor1": distractors[0],
                            "sys_distractor2": distractors[1],
                            "sys_distractor3": distractors[2],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )


def evaluate_bleu(reference_distractors, generated_distractors):
    return sentence_bleu([reference_distractors], generated_distractors)


def evaluate_rouge(reference_distractors, generated_distractors):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, gen)["rougeL"].fmeasure
        for ref, gen in zip(reference_distractors, generated_distractors)
    ]
    return np.mean(scores) if scores else 0


def evaluate_sbert_similarity(correct_answer, generated_distractors):
    embeddings = sbert_model.encode(
        [correct_answer] + generated_distractors, convert_to_tensor=True
    )
    correct_embedding = embeddings[0]
    similarities = [
        util.cos_sim(correct_embedding, emb).item() for emb in embeddings[1:]
    ]
    return np.mean(similarities)


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
    plt.savefig("plots/llama3.2_1B_distractors_generation.png")


def evaluate(df):
    bleu_scores, rouge_scores, sbert_similarities = [], [], []

    for idx, row in tqdm(df.iterrows()):
        reference_distractors = [
            row["ref_distractor1"],
            row["ref_distractor2"],
            row["ref_distractor3"],
        ]
        generated_distractors = [
            row["sys_distractor1"],
            row["sys_distractor2"],
            row["sys_distractor3"],
        ]

        for i in range(3):
            if not isinstance(generated_distractors[i], str):
                generated_distractors[i] = ""

        bleu = evaluate_bleu(reference_distractors, generated_distractors)
        rouge = evaluate_rouge(reference_distractors, generated_distractors)
        sbert_sim = evaluate_sbert_similarity(
            row["correct_answer"], generated_distractors
        )

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        sbert_similarities.append(sbert_sim)

    bleu_scores = np.array(bleu_scores)
    rouge_scores = np.array(rouge_scores)
    sbert_similarities = np.array(sbert_similarities)

    plot_metrics(bleu_scores, rouge_scores, sbert_similarities)


def main():
    # ds = load_dataset("allenai/sciq")
    # df = generate_all(ds["test"])

    # df.to_csv("llama3.2_1B_distractors_generation.csv", index=False)

    # Evaluate
    with open("llama3.2_1B_distractors_generation.csv") as f:
        df = pd.read_csv(f)

    evaluate(df)


if __name__ == "__main__":
    main()
