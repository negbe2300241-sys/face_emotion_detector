from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

MODEL_NAME = "trpakov/vit-face-expression"

print("Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully.")

def predict_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = probs.argmax().item()
        predicted_label = model.config.id2label[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()
    return predicted_label, confidence