import streamlit as st
from PIL import Image
import torch

# Try both old and new names depending on Transformers version
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    def load_processor(model_name):
        return AutoImageProcessor.from_pretrained(model_name)
except ImportError:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    def load_processor(model_name):
        return AutoFeatureExtractor.from_pretrained(model_name)

MODEL_NAME = "trpakov/vit-face-expression"

@st.cache_resource
def load_model():
    st.write("⏳ Loading model and processor... please wait.")
    processor = load_processor(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model

processor, model = load_model()

def predict_emotion(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_idx = probs.argmax().item()
        predicted_label = model.config.id2label[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()
    return predicted_label, confidence


st.title("🧠 Emotion Detector (ViT Model)")
st.write("This app uses a pretrained Vision Transformer model to detect facial emotions.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Emotion"):
        with st.spinner("Detecting emotion..."):
            label, conf = predict_emotion(image)
        st.success(f"**Emotion:** {label}")
        st.info(f"**Confidence:** {conf * 100:.2f}%")
else:
    st.write("👆 Please upload a face image to get started.")

st.caption("Model source: [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression)")
