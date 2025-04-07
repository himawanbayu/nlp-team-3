from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate_distractors(
    context, question, answer, model, tokenizer
):
    """Generates distractors for a given question-answer pair."""
    prompt = f"""
    Generate plausible but incorrect answer choices (distractors) for the following question-answer pair.
    Context: '{context}'
    Question: '{question}'
    Correct answer: '{answer}'.
    Provide exactly three different distractors as a comma-separated list. Do not include explanations, commentary, or additional text.
    """.strip()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)

    distractors = [d.strip() for d in response.split(",") if d.strip()]
    distractors = [d for d in distractors if d.lower() != answer.lower()]
    while len(distractors) < 3:
        distractors.append("")
    return distractors[:3]


def generate_all(ds):
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    data = []
    for example in tqdm(ds):
        distractors = generate_distractors(
            example["support"],
            example["question"],
            example["correct_answer"],
            model,
            tokenizer,
        )
        data.append({
            "question": example["question"],
            "support": example["support"],
            "correct_answer": example["correct_answer"],
            "ref_distractor1": example["distractor1"],
            "ref_distractor2": example["distractor2"],
            "ref_distractor3": example["distractor3"],
            "sys_distractor1": distractors[0],
            "sys_distractor2": distractors[1],
            "sys_distractor3": distractors[2]
        })
    return pd.DataFrame(data)


def main():
    ds = load_dataset("allenai/sciq")
    df = generate_all(ds["test"])

    df.to_csv("generated_distractors/qwen_distractors.csv", index=False)

if __name__ == "__main__":
    main()
