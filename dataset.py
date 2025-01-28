import pandas as pd
import os
import sys

# Define the path to the dataset file
DATASET_FILE = "dataset.csv"

def add_example(text, label):
    """
    Add a new example to the dataset.
    - text: The text content (string)
    - label: 0 for Human, 1 for AI
    """
    # Load the existing dataset (if any)
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
    else:
        # If the file doesn't exist, create a new one with the headers
        df = pd.DataFrame(columns=["text", "label"])
    
    # Create a new example row
    new_example = pd.DataFrame({"text": [text], "label": [label]})
    
    # Use pd.concat() to append the new row
    df = pd.concat([df, new_example], ignore_index=True)
    
    # Save the updated dataset to the CSV file
    df.to_csv(DATASET_FILE, index=False)
    print(f"Added new example: {text[:50]}... with label {'Human' if label == 0 else 'AI'}")

def insert_from_txt(file_path, label):
    """
    Insert text examples from a file into the dataset.
    - file_path: The path to the text file
    - label: 0 for Human, 1 for AI
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip the line to avoid leading/trailing whitespace
                line = line.strip()
                if line:  # Only add non-empty lines
                    add_example(line, label)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def show_examples():
    """
    Show all current examples in the dataset.
    """
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        print(df.head())
    else:
        print("No dataset found!")

if __name__ == "__main__":
    # Check command-line arguments for action
    if len(sys.argv) > 2 and sys.argv[1] == "insert-0":
        file_path = sys.argv[2]
        insert_from_txt(file_path, label=0)  # Add with label 0 (Human)
    elif len(sys.argv) > 2 and sys.argv[1] == "insert-1":
        file_path = sys.argv[2]
        insert_from_txt(file_path, label=1)  # Add with label 1 (AI)
    else:
        print("Usage: dataset.py insert-0 path-to.txt or dataset.py insert-1 path-to.txt")
        print("Use 'insert-0' for Human (0) and 'insert-1' for AI (1).")
