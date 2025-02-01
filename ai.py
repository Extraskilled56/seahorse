import os
from pathlib import Path

# Disable oneDNN custom operations warnings (if TensorFlow is used elsewhere)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import psutil
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # Use PyTorch's AdamW to avoid deprecation warnings.
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# Define paths in a cross-platform way.
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
LOG_FILE = BASE_DIR / "ai.log"
MODEL_SAVE_DIR = BASE_DIR / "seahorse_pt"

# Configure logging to output to both console and a log file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler()
    ]
)

def get_system_resources():
    """Get available system resources (CPU cores, RAM in GB, GPU memory in GB)."""
    cpu_cores = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_memory = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                  if torch.cuda.is_available() else 0)
    return cpu_cores, ram_gb, gpu_memory

def adjust_parameters(cpu_cores, ram_gb, gpu_memory, device):
    """
    Adjust batch size, number of workers, and whether to use mixed precision
    based on device type and available resources.
    """
    if device.type == "cuda":
        if gpu_memory >= 24:
            batch_size = 32
        elif gpu_memory >= 16:
            batch_size = 16
        else:
            batch_size = 8
        num_workers = min(cpu_cores, 4)
    else:
        # For CPU, use a smaller batch size and more workers for data loading.
        batch_size = 4
        num_workers = min(cpu_cores, 8)
        torch.set_num_threads(cpu_cores)  # Utilize all available CPU cores.

    # Only use AMP when on GPU.
    use_amp = (device.type == "cuda")
    return batch_size, num_workers, use_amp

def load_dataset():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    logging.error("Dataset not found! Please add examples using dataset.py.")
    return None

def tokenize_texts(texts, tokenizer, max_length=256):
    """Tokenize a list of texts using the provided tokenizer."""
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
    # Detect system resources and device.
    cpu_cores, ram_gb, gpu_memory = get_system_resources()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"System Resources: CPU Cores={cpu_cores}, RAM={ram_gb:.2f} GB, GPU Memory={gpu_memory:.2f} GB")
    logging.info(f"Using device: {device}")

    # Adjust parameters based on system performance.
    batch_size, num_workers, use_amp = adjust_parameters(cpu_cores, ram_gb, gpu_memory, device)
    logging.info(f"Adjusted Parameters: Batch Size={batch_size}, Workers={num_workers}, Mixed Precision={use_amp}")

    # Load tokenizer and model.
    # Use "bert-base-uncased" for a smaller model.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    logging.info(f"Total Trainable Parameters: {count_parameters(model):,}")

    # Load and tokenize the dataset.
    df = load_dataset()
    if df is None:
        return

    encoded = tokenize_texts(df['text'].tolist(), tokenizer)
    labels = torch.tensor(df['label'].values)
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'], labels)

    # Split dataset into training and validation sets.
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # DataLoader settings.
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True if device.type == "cuda" else False,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_data, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_data, **dataloader_kwargs)

    # Compute class weights for imbalanced datasets.
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Use PyTorch's AdamW optimizer.
    optimizer = AdamW(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    # Use AMP only if enabled (this will be disabled on CPU).
    scaler = torch.amp.GradScaler(enabled=use_amp)

    logging.info("Training Seahorse model...")
    num_epochs = 10  # Adjust as needed.
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Training loop.
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # Use autocast with device_type parameter.
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # Validation loop.
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        val_acc = correct / total
        logging.info(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save the trained model and tokenizer.
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    model.save_pretrained(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    logging.info(f"Saved PyTorch model to '{MODEL_SAVE_DIR}' directory")

    # Test the model on the entire dataset.
    logging.info("Testing the model on the entire dataset...")
    test_loader = DataLoader(dataset, **dataloader_kwargs)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.info(f"Overall Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
