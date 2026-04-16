from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline(
    "text-classification",
    model="news_ai_model",
    tokenizer="news_ai_model"
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    result = classifier(text)

    return jsonify({
        "category": result[0]['label'],
        "confidence": float(result[0]['score'])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)