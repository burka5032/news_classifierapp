from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="news_ai_model",
    tokenizer="news_ai_model"
)

def predict(text):
    result = classifier(text)
    label = result[0]["label"]
    score = float(result[0]["score"])
    return f"{label} ({score:.2f})"

# Gradio interface
app = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="News AI Classifier",
    description="Enter news text to classify"
)

app.launch()