import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import psutil
from torch.cuda.amp import GradScaler, autocast

def get_system_resources():
    """Get available system resources (CPU cores, RAM, GPU memory)."""
    cpu_cores = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
    return cpu_cores, ram_gb, gpu_memory

def adjust_parameters(cpu_cores, ram_gb, gpu_memory):
    """Adjust batch size, number of workers, and other parameters based on system resources."""
    if gpu_memory >= 24:  # High-end GPU
        batch_size = 32
    elif gpu_memory >= 16:  # Mid-range GPU
        batch_size = 16
    else:  # Low-end GPU or CPU
        batch_size = 8
    num_workers = min(cpu_cores, 4)  # Use up to 4 workers
    use_amp = torch.cuda.is_available()
    return batch_size, num_workers, use_amp

def load_dataset():
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
        return df
    print("Dataset not found! Add examples using dataset.py.")
    return None

def tokenize_texts(texts, tokenizer, max_length=256):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    cpu_cores, ram_gb, gpu_memory = get_system_resources()
    print(f"System Resources: CPU Cores={cpu_cores}, RAM={ram_gb:.2f} GB, GPU Memory={gpu_memory:.2f} GB")
    batch_size, num_workers, use_amp = adjust_parameters(cpu_cores, ram_gb, gpu_memory)
    print(f"Adjusted Parameters: Batch Size={batch_size}, Workers={num_workers}, Mixed Precision={use_amp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
    model.to(device)
    
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    df = load_dataset()
    if df is None:
        return

    encoded = tokenize_texts(df['text'].tolist(), tokenizer)
    labels = torch.tensor(df['label'].values).to(device)
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], labels)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(enabled=use_amp)

    print("Training Seahorse model...")
    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            with autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

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

    model.save_pretrained("seahorse_pt")
    tokenizer.save_pretrained("seahorse_pt")
    print("Saved PyTorch model to 'seahorse_pt' directory")

    print("\nTesting the model on the entire dataset...")
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
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
