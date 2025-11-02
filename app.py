import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# --- Page setup ---
st.set_page_config(page_title="Facial Emotion Detector", layout="centered")

st.title("😃 Facial Emotion Detector")
st.write("Upload an image to detect emotion using ViT transformer model.")

# --- Step 1: Load the model once ---
@st.cache_resource
def load_model():
    model_name = "dima806/facial_emotions_image_detection"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    model.eval()  # important for inference
    return model, processor

# Load model once when app starts
model, processor = load_model()

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Emotion"):
        with st.spinner("Detecting emotion..."):
            # Process image safely
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():  # prevents memory leaks
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
                label = model.config.id2label[predicted_class_idx]

        st.success(f"**Detected emotion:** {label}")

st.caption("Model: [dima806/facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection)")
