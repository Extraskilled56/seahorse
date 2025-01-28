import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

def load_dataset():
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
        return df
    print("Dataset not found! Add examples using dataset.py.")
    return None

def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

def main():
    # Force CPU usage
    device = torch.device("cpu")

    # Initialize components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    
    # Load and prepare data
    df = load_dataset()
    if df is None:
        return

    # Tokenization
    encoded = tokenize_texts(df['text'].tolist(), tokenizer)
    labels = torch.tensor(df['label'].values).to(device)
    
    # Create dataset
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], labels)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    print("Training Seahorse model...")
    for epoch in range(15):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {correct/total:.4f}")

    # Save PyTorch model
    model.save_pretrained("seahorse_pt")
    tokenizer.save_pretrained("seahorse_pt")
    print("Saved PyTorch model to 'seahorse_pt' directory")

    # Test the model on the entire dataset
    print("\nTesting the model on the entire dataset...")
    test_loader = DataLoader(dataset, batch_size=16)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()