import pandas as pd
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMultipleChoice

"""
This script uses a *question + context* format for building the prompt. 
It aligns the idea of query-first format suggested by the paper *Lost in the Middle*
However, it does not work well - Hongxu Zhou
"""

# Set seeds for reproducibility
def set_seed(seed=42):
    """
    Set seeds for reproducibility across all libraries used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset class to handle the multiple choice data
class MultipleChoiceDataset(Dataset):
    """
    Dataset class to prepare data for BertForMultipleChoice model
    """
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Extract question, context, correct answer, and distractors
        question = row['question']
        context = str(row['support'])  # Ensure context is a string
        correct_answer = row['correct_answer']
        
        # Get distractors from the dataset
        distractors = [
            row['ref_distractor1'],
            row['ref_distractor2'],
            row['ref_distractor3']
        ]
        
        # Create a list of all options (correct answer + distractors)
        choices = [correct_answer] + distractors
        
        # Shuffle the choices and keep track of correct answer index
        correct_index = 0  # Original position of correct answer
        shuffled_indices = list(range(len(choices)))
        random.shuffle(shuffled_indices)
        
        # Find the new position of the correct answer after shuffling
        new_correct_index = shuffled_indices.index(correct_index)
        shuffled_choices = [choices[i] for i in shuffled_indices]
        
        # Combine question and context (Option 3 format)
        combined_context = f"{question} {context}"
        
        # Encode each context+choice pair for BERT
        # The context+choice format may be uncommon for BERT, but is on purpose
        # It aligns the query-first format suggested by *Lost in the Middle* paper 
        encodings = []
        for choice in shuffled_choices:
            # Format: [CLS] question + context [SEP] option [SEP]
            encoding = self.tokenizer.encode_plus(
                combined_context,
                choice,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'token_type_ids': encoding['token_type_ids'].squeeze() if 'token_type_ids' in encoding else None
            })
        
        # Stack encoded inputs for all options
        input_ids = torch.stack([enc['input_ids'] for enc in encodings])
        attention_mask = torch.stack([enc['attention_mask'] for enc in encodings])
        
        # Handle token_type_ids if present (depends on tokenizer)
        if encodings[0]['token_type_ids'] is not None:
            token_type_ids = torch.stack([enc['token_type_ids'] for enc in encodings])
        else:
            token_type_ids = None 
        # This code should let us verify the input formats
        if idx < 2:  # Print first 2 examples
            print(f"\nExample {idx+1}:")
            print(f"Question: {question}")
            print(f"Correct answer: {correct_answer}")
            for i, choice in enumerate(shuffled_choices):
                print(f"Option {i}: {choice}" + (" (CORRECT)" if i == new_correct_index else ""))
                encoded = self.tokenizer.encode_plus(combined_context, choice, add_special_tokens=True, max_length=30)
                decoded = self.tokenizer.decode(encoded['input_ids'])
                print(f"  Tokenized (truncated): {decoded}")    

        
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'correct_index': torch.tensor(new_correct_index),
            'question': question,
            'context': context,
            'correct_answer': correct_answer,
            'shuffled_choices': shuffled_choices,
            'original_index': idx
        }

def collate_fn(batch):
    """
    Custom collate function to process batches for DataLoader.
    Handles variable-length inputs properly.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Handle token_type_ids if present
    if batch[0]['token_type_ids'] is not None:
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    else:
        token_type_ids = None
        
    correct_indices = torch.stack([item['correct_index'] for item in batch])
    
    # Collect metadata that doesn't need tensor operations
    metadata = [{
        'question': item['question'],
        'context': item['context'],
        'correct_answer': item['correct_answer'],
        'shuffled_choices': item['shuffled_choices'],
        'original_index': item['original_index']
    } for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'correct_indices': correct_indices,
        'metadata': metadata
    }

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the test dataset.
    Returns detailed results for each question.
    """
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if batch['token_type_ids'] is not None else None
            correct_indices = batch['correct_indices'].to(device)
            metadata = batch['metadata']
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Get predictions
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Process results for each item in batch
            for i in range(len(metadata)):
                # Check if prediction is correct
                is_correct = (predictions[i].item() == correct_indices[i].item())
                
                # Get confidence scores
                confidence_scores = probabilities[i].cpu().numpy()
                chosen_confidence = confidence_scores[predictions[i].item()]
                correct_confidence = confidence_scores[correct_indices[i].item()]
                
                # Record result
                result = {
                    'original_index': metadata[i]['original_index'],
                    'question': metadata[i]['question'],
                    'correct_answer': metadata[i]['correct_answer'],
                    'shuffled_choices': metadata[i]['shuffled_choices'],
                    'correct_index': correct_indices[i].item(),
                    'predicted_index': predictions[i].item(),
                    'is_correct': is_correct,
                    'confidence_scores': confidence_scores.tolist(),
                    'chosen_confidence': chosen_confidence,
                    'correct_confidence': correct_confidence
                }
                all_results.append(result)
    
    # Calculate overall accuracy
    accuracy = sum(result['is_correct'] for result in all_results) / len(all_results)
    print(f"Overall accuracy: {accuracy:.4f}")

    # The debugging code here prints detials about the first few questions
    if len(all_results) >= 10:  # Print first 10 results for debugging
        for i, result in enumerate(all_results[:10]):
            print(f"\nQuestion {i+1}: {result['question']}")
            print(f"Correct answer: '{result['correct_answer']}'")
            print(f"Options after shuffling:")
            for j, opt in enumerate(result['shuffled_choices']):
                prob = result['confidence_scores'][j]
                correct_mark = " ✓" if j == result['correct_index'] else ""
                pred_mark = " (model choice)" if j == result['predicted_index'] else ""
                print(f"  {j}: '{opt}' - {prob:.4f}{correct_mark}{pred_mark}")
    
    return all_results, accuracy

def save_results_to_csv(results, output_file):
    """
    Save evaluation results to a CSV file
    """
    # The debugging code will provide exmaples to verify the calculation method
    correct_count = 0
    for i, result in enumerate(results[:min(10, len(results))]):
        print(f"\nQuestion {i+1}")
        print(f"Predicted: {result['predicted_index']}, Correct: {result['correct_index']}")
        print(f"Is correct: {result['is_correct']} (calculated) vs {result['predicted_index'] == result['correct_index']} (manual check)")
        if result['predicted_index'] == result['correct_index']:
            correct_count += 1
    print(f"Manual accuracy check on first 10: {correct_count/min(10, len(results)):.4f}")

    # Prepare data for DataFrame
    data = []
    for result in results:
        # Format confidence scores as a comma-separated string
        confidence_str = ','.join([f"{score:.4f}" for score in result['confidence_scores']])
        
        # Format options as a comma-separated string
        options_str = ','.join(result['shuffled_choices'])
        
        data.append({
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
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def cleanup():
    """
    Clean up GPU memory between runs
    """
    if torch.cuda.is_available():
        # Clear the cache
        torch.cuda.empty_cache()
        
        # Force a garbage collection cycle
        import gc
        gc.collect()
    
    print("Memory cleaned up for next run")

def main():
    # 1. Set up
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load the model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
    model.to(device)
    
    # 3. Load and preprocess the dataset
    # Adjust the path to the dataset as needed
    print("Loading dataset...")
    dataset_path = '/scratch/s5788668/llama3.2_1B_generated_distractors.tsv'  # Adjust path as needed
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Take only the first 500 questions
    df = df.head(500)
    print(f"Loaded {len(df)} questions from the dataset")
    
    # 4. Create dataset and dataloader
    mc_dataset = MultipleChoiceDataset(df, tokenizer)
    dataloader = DataLoader(
        mc_dataset, 
        batch_size=8, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 5. Evaluate the model
    print("Starting evaluation...")
    results, accuracy = evaluate_model(model, dataloader, device)
    
    # 6. Save results
    output_file = '/scratch/s5788668/bert_original_distractor_question_context_results.csv'
    save_results_to_csv(results, output_file)
    
    print(f"Evaluation completed with accuracy: {accuracy:.4f}")

    # 7. Clean up resources
    cleanup()

if __name__ == "__main__":
    main()