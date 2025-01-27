import pandas as pd
import os

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
    while True:
        action = input("Enter 'add' to add a new example or 'show' to view examples (or 'exit' to quit): ").strip().lower()
        
        if action == "add":
            text = input("Enter the text: ")
            label = input("Enter label (0 for Human, 1 for AI): ").strip()
            try:
                label = int(label)
                if label not in [0, 1]:
                    print("Invalid label! Please enter 0 for Human or 1 for AI.")
                    continue
                add_example(text, label)
            except ValueError:
                print("Invalid input! Please enter a numeric value (0 or 1) for the label.")
        
        elif action == "show":
            show_examples()
        
        elif action == "exit":
            break
        else:
            print("Invalid action! Please enter 'add', 'show', or 'exit'.")
