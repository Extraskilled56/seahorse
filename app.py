from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
device = torch.device("cpu")  # Force CPU usage
tokenizer = BertTokenizer.from_pretrained("seahorse_pt")
model = BertForSequenceClassification.from_pretrained("seahorse_pt")
model.to(device)
model.eval()  # Set the model to evaluation mode

def classify_text(text):
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

@app.route('/classify', methods=['POST'])
def classify():
    # Get the input text from the request
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data['text']
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Invalid or empty text"}), 400

    # Classify the text
    try:
        predicted_class, confidence = classify_text(text)
        return jsonify({
            "classification": predicted_class,
            "confidence": round(confidence, 2)  # Round to 2 decimal places
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)