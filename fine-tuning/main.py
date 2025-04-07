import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    AdamW
)
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "albert-base-v2"
# MODEL_NAME = "roberta-base"
# MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "bert-large-uncased"
# MAX_LEN = 128 if "large" not in MODEL_NAME else 256
MAX_LEN = 256
# BATCH_SIZE = 8
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
# LEARNING_RATE = 2e-5 
# EPOCHS = 3
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sciq_from_huggingface():
    dataset = load_dataset("sciq")
    train_df = pd.DataFrame(dataset['train'])
    validation_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    return train_df, validation_df, test_df

class SciQMultipleChoiceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, use_token_type_ids=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_token_type_ids = use_token_type_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        question = item['question']
        choices = [
            item['correct_answer'],
            item['distractor1'],
            item['distractor2'],
            item['distractor3']
        ]
        shuffled_indices = np.random.permutation(4)
        shuffled_choices = [choices[i] for i in shuffled_indices]
        correct_idx = np.where(shuffled_indices == 0)[0][0]
        
        encodings = []
        for choice in shuffled_choices:
            encoding = self.tokenizer(
                question,
                choice,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encoding_dict = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
            if self.use_token_type_ids and 'token_type_ids' in encoding:
                encoding_dict['token_type_ids'] = encoding['token_type_ids'].squeeze()
            encodings.append(encoding_dict)
        
        out = {
            'input_ids': torch.stack([e['input_ids'] for e in encodings]),
            'attention_mask': torch.stack([e['attention_mask'] for e in encodings]),
            'label': torch.tensor(correct_idx, dtype=torch.long)
        }
        if self.use_token_type_ids and 'token_type_ids' in encodings[0]:
            out['token_type_ids'] = torch.stack([e['token_type_ids'] for e in encodings])
        return out

def train_model(model, train_loader, val_loader, optimizer, device, epochs, use_token_type_ids=True):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels=labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_accuracy = correct / total
        print(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_path = f"best_{MODEL_NAME.replace('/', '_')}_sciq_model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved new best model to {model_path}")
    return model

# Evaluation
def evaluate_model(model, test_loader, device, use_token_type_ids=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    print(f"📌 Fine-tuning model: {MODEL_NAME}")
    print(f"📏 MAX_LEN: {MAX_LEN}, DEVICE: {DEVICE}")

    train_df, val_df, test_df = load_sciq_from_huggingface()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME).to(DEVICE)

    use_token_type_ids = 'token_type_ids' in tokenizer.model_input_names

    train_dataset = SciQMultipleChoiceDataset(train_df, tokenizer, MAX_LEN, use_token_type_ids)
    val_dataset = SciQMultipleChoiceDataset(val_df, tokenizer, MAX_LEN, use_token_type_ids)
    test_dataset = SciQMultipleChoiceDataset(test_df, tokenizer, MAX_LEN, use_token_type_ids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model = train_model(model, train_loader, val_loader, optimizer, DEVICE, EPOCHS, use_token_type_ids)

    model.load_state_dict(torch.load(f"best_{MODEL_NAME.replace('/', '_')}_sciq_model.pt"))
    model.to(DEVICE)
    evaluate_model(model, test_loader, DEVICE, use_token_type_ids)

    # Save 
    model.save_pretrained(f"{MODEL_NAME.replace('/', '_')}_final_model")
    tokenizer.save_pretrained(f"{MODEL_NAME.replace('/', '_')}_final_tokenizer")
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    main()
