import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

MODEL_NAME = "trpakov/vit-face-expression"

@st.cache_resource
def load_model():
    st.write("Loading model...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return extractor, model

extractor, model = load_model()

def predict_emotion(image: Image.Image):
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = probs.argmax().item()
        predicted_label = model.config.id2label[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()
    return predicted_label, confidence


st.title("🧠 Emotion Detector (Pretrained ViT Model)")
st.write("This app uses a pretrained Vision Transformer model from Hugging Face to detect facial emotions.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Emotion"):
        with st.spinner("Detecting emotion..."):
            label, conf = predict_emotion(image)
        st.success(f"**Emotion:** {label}")
        st.info(f"**Confidence:** {conf*100:.2f}%")
else:
    st.write("👆 Upload a face image to start.")

st.caption("Model: [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)")
