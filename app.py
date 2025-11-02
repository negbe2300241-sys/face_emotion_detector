import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

MODEL_NAME = "trpakov/vit-face-expression"


# Cache the model and processor
@st.cache_resource(show_spinner="Loading model and processor...")
def load_model():
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


# Load once
processor, model = load_model()


def predict_emotion(image: Image.Image):
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = probs.argmax(-1).item()
        confidence = probs[0, predicted_idx].item()
        label = model.config.id2label[predicted_idx]

    return label, confidence


# === Streamlit UI ===
st.title("Emotion Detector (ViT)")
st.write("Detect facial emotions using a fine-tuned Vision Transformer.")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Emotion"):
        with st.spinner("Analyzing..."):
            label, conf = predict_emotion(image)

        st.success(f"**Detected Emotion:** {label.title()}")
        st.metric("Confidence", f"{conf:.1%}")

        # Optional: Show all probabilities
        with st.expander("View all emotion scores"):
            probs = torch.nn.functional.softmax(model(**processor(images=image, return_tensors="pt")).logits, dim=-1)[0]
            for idx, prob in enumerate(probs):
                emo = model.config.id2label[idx]
                st.write(f"**{emo.title()}**: {prob.item():.1%}")
else:
    st.info("👆 Upload an image with a clear face to get started.")

st.caption(f"Model: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})")