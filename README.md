# Amazon Polarity Sentiment Classifier

This project fine-tunes the [`microsoft/deberta-v3-small`](https://huggingface.co/microsoft/deberta-v3-small) transformer model on the [Amazon Polarity dataset](https://huggingface.co/datasets/amazon_polarity) to classify user reviews as either **Positive** or **Negative**.

## 🚀 Model Highlights

- Model: microsoft/deberta-v3-small
- Task: Binary Sentiment Classification
- Dataset: Amazon Polarity (subset for training/testing)
- Framework: Hugging Face Transformers
- Achieved high accuracy on validation set using custom fine-tuning.

## 🔎 Example Prediction

```python
from inference import classify_review

title = "Best headphones ever!"
content = "The audio quality is incredible and battery lasts long."
label, confidence = classify_review(title, content)

print(f"Sentiment: {label} (Confidence: {confidence:.2f})")
