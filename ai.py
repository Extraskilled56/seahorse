import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataset import show_examples
import os

# Load the dataset from dataset.csv
def load_dataset():
    # Load the existing dataset
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
        return df
    else:
        print("Dataset not found! Please add some examples using dataset.py.")
        return None

# Step 2: Preprocess and tokenize text
def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )

# Main logic for model training and testing
def main():
    # Load dataset
    df = load_dataset()
    if df is None:
        return

    # Step 1: Prepare the dataset
    X = tokenize_texts(df["text"].tolist(), tokenizer)
    y = np.array(df["label"])

    X_input_ids = X["input_ids"].numpy()
    X_attention_mask = X["attention_mask"].numpy()

    # Step 3: Train-test split
    X_train_ids, X_test_ids, y_train, y_test = train_test_split(
        X_input_ids, y, test_size=0.2, random_state=42
    )

    X_train_mask, X_test_mask, _, _ = train_test_split(
        X_attention_mask, y, test_size=0.2, random_state=42
    )

    # Step 4: Define and compile the model
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Step 5: Train the model
    print("Training the model...")
    model.fit(
        x={
            "input_ids": X_train_ids,
            "attention_mask": X_train_mask,
        },
        y=y_train,
        validation_data=(
            {
                "input_ids": X_test_ids,
                "attention_mask": X_test_mask,
            },
            y_test,
        ),
        epochs=5,
        batch_size=4,
    )

    # Step 6: Evaluate the model
    loss, accuracy = model.evaluate(
        {
            "input_ids": X_test_ids,
            "attention_mask": X_test_mask,
        },
        y_test,
    )
    print(f"Test accuracy: {accuracy:.2f}")

    # Step 7: Dynamic input for predictions
    print("\nEnter text to classify (type 'exit' to quit):")
    while True:
        user_input = input("Text: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        # Tokenize user input
        tokens = tokenize_texts([user_input], tokenizer)
        predictions = model.predict(
            {
                "input_ids": tokens["input_ids"].numpy(),
                "attention_mask": tokens["attention_mask"].numpy(),
            }
        ).logits
        
        # Apply softmax to get probabilities
        prob = tf.nn.softmax(predictions, axis=-1).numpy()
        
        # Get prediction (class with highest probability)
        predicted_class = np.argmax(prob, axis=1)[0]
        confidence = prob[0, predicted_class] * 100  # Confidence in percentage
        
        label = "Human" if predicted_class == 0 else "AI"
        print(f"Predicted Label: {label}")
        print(f"Confidence: {confidence:.2f}%\n")

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    main()
