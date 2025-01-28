import torch
from transformers import BertTokenizer, BertForSequenceClassification

def classify_text(text, model, tokenizer, device):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    # Determine the predicted class and confidence
    predicted_class = "AI" if torch.argmax(logits).item() == 1 else "Human"
    confidence = probabilities[1] if predicted_class == "AI" else probabilities[0]

    return predicted_class, confidence * 100  # Return confidence as a percentage

def main():
    # Force CPU usage
    device = torch.device("cpu")

    # Load the trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("seahorse_pt")
    model = BertForSequenceClassification.from_pretrained("seahorse_pt")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    print("Text Classification CLI")
    print("Type 'exit' or 'quit' to stop the program.\n")

    while True:
        # Get user input
        text = input("Enter text to classify: ").strip()

        # Exit condition
        if text.lower() in ["exit", "quit"]:
            print("Exiting the program. Goodbye!")
            break

        # Classify the text
        predicted_class, confidence = classify_text(text, model, tokenizer, device)

        # Print the result
        print(f"Classification: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%\n")

if __name__ == '__main__':
    main()