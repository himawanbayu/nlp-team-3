"""
This script uses the model sileod/deberta-v3-large-tasksource-nli with first 500 entries from llama generated distractors. 
It uses HF pipeline zero-shot classification instead of AutoModelForMultiChoice.
Other testing settings:
* choices are randomly shuffled
* the question stems and context are concatenated in query-context format
"""

import pandas as pd
import numpy as np
import random
import torch
from tqdm import tqdm
from transformers import pipeline

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main function
def main():
    # 1. Set up
    set_seed(42)
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 2. Load the model using the pipeline
    print("Loading model...")
    classifier = pipeline(
        "zero-shot-classification", 
        model="sileod/deberta-v3-large-tasksource-nli",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # 3. Load the dataset
    print("Loading dataset...")
    dataset_path = '/scratch/s5788668/llama3.2_1B_generated_distractors.tsv'
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Take only the first 1500 questions if needed
    df = df.head(1500)
    print(f"Loaded {len(df)} questions from the dataset")
    
    # 4. Create results dataframe
    results = []
    
    # 5. Process each question
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Extract question, context, correct answer, and distractors
        question = row['question']
        context = str(row['support']) # Make sure it is string
        correct_answer = row['correct_answer']
        
        # Get distractors from the dataset -- subject to change to better distractors
        distractors = [
            row['sys_distractor1'],
            row['sys_distractor2'], 
            row['sys_distractor3']
        ] # IF YOU ARE SEEING THIS: sys_distractor means system-generated distractors, AKA auto-generated
        # You are not wrong, take a break, do a san check
        
        # Create a list of all options (correct answer + distractors)
        choices = [correct_answer] + distractors
        
        # Create a mapping of original positions to track correct answer
        choice_mapping = {choice: i for i, choice in enumerate(choices)}
        
        # Shuffle the choices
        shuffled_choices = choices.copy()
        random.shuffle(shuffled_choices)
        
        # Find new position of correct answer
        new_correct_index = shuffled_choices.index(correct_answer)
        
        # Run the zero-shot classification
        hypothesis_template = "The answer is {}."
        
        # Use the question and context together for classification
        if len(context) > 0:
            classification_input = f"{question} Context: {context}"
        else:
            classification_input = question
            
        result = classifier(
            classification_input, 
            shuffled_choices,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )
        
        # Extract scores and normalize to ensure they sum to 1
        scores = result['scores']
        scores_tensor = torch.tensor(scores)
        normalized_scores = torch.nn.functional.softmax(scores_tensor, dim=0).tolist()
        
        # Find the predicted answer
        predicted_index = np.argmax(normalized_scores)
        is_correct = (predicted_index == new_correct_index)
        
        # Record results
        result_item = {
            'original_index': idx,
            'question': question,
            'context': context,
            'correct_answer': correct_answer,
            'shuffled_choices': shuffled_choices,
            'correct_index': new_correct_index, 
            'predicted_index': predicted_index,
            'is_correct': is_correct,
            'confidence_scores': normalized_scores,
            'chosen_confidence': normalized_scores[predicted_index],
            'correct_confidence': normalized_scores[new_correct_index]
        }
        results.append(result_item)
        
        # Print some debug information for the first few questions
        if idx < 15:
            print(f"\nQuestion {idx+1}: {question}")
            print(f"Correct answer: '{correct_answer}'")
            print(f"Options after shuffling:")
            for j, opt in enumerate(shuffled_choices):
                prob = normalized_scores[j]
                correct_mark = " ✓" if j == new_correct_index else ""
                pred_mark = " (model choice)" if j == predicted_index else ""
                print(f"  {j}: '{opt}' - {prob:.4f}{correct_mark}{pred_mark}")
    
    # 6. Save results
    output_data = []
    for result in results:
        # Format confidence scores as a comma-separated string
        confidence_str = ','.join([f"{score:.4f}" for score in result['confidence_scores']])
        
        # Format options as a comma-separated string
        options_str = ','.join(result['shuffled_choices'])
        
        output_data.append({
            'question': result['question'],
            'correct_answer': result['correct_answer'],
            'options': options_str,
            'correct_index': result['correct_index'],
            'predicted_index': result['predicted_index'],
            'is_correct': result['is_correct'],
            'confidence_scores': confidence_str,
            'chosen_confidence': result['chosen_confidence'],
            'correct_confidence': result['correct_confidence']
        })
    
    # Create and save DataFrame
    output_df = pd.DataFrame(output_data)
    output_file = '/scratch/s5788668/deberta_nli_llama_generated_distractors_results.csv'
    output_df.to_csv(output_file, index=False)
    
    # 7. Report overall accuracy
    accuracy = sum(r['is_correct'] for r in results) / len(results)
    print(f"\nEvaluation completed with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
