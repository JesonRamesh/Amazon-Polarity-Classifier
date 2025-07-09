
---

### ⚙️ `inference.py`

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model & tokenizer (adjust path if needed)
model_path = "./amazon-polarity-model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def classify_review(title, content):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text = title.strip() + " " + content.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        label = "Positive" if predicted.item() == 1 else "Negative"

    return label, confidence.item()
