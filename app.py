import streamlit as st
from PIL import Image
import torch

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification


    def load_processor(model_name):
        return AutoImageProcessor.from_pretrained(model_name)
except ImportError:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification


    def load_processor(model_name):
        return AutoFeatureExtractor.from_pretrained(model_name)

# ✅ Use Auto classes instead of specific ViT classes
MODEL_NAME = "dima806/facial_emotions_image_detection"


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        st.write("⏳ Loading model and processor...")

        # Use AutoModelForImageClassification instead of ViTForImageClassification
        processor = load_processor(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        model.eval()

        st.write("✅ Model loaded successfully")
        return processor, model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None


processor, model = load_model()


def predict_emotion(image: Image.Image):
    if model is None or processor is None:
        return "Model not loaded", 0.0

    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_idx = probs.argmax().item()
            predicted_label = model.config.id2label[predicted_class_idx]
            confidence = probs[0][predicted_class_idx].item()
        return predicted_label, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0


st.title("🧠 Lightweight Emotion Detector")
st.write("This app uses a smaller pretrained model for emotion detection.")

if model is None:
    st.error("⚠️ Model failed to load. The app may not work properly on free tier.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Emotion"):
            with st.spinner("Detecting emotion..."):
                label, conf = predict_emotion(image)

            if "Error" in label:
                st.error(label)
            else:
                st.success(f"**Emotion:** {label}")
                st.info(f"**Confidence:** {conf * 100:.2f}%")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
else:
    st.write("👆 Please upload a face image to get started.")

st.caption(
    "Model: [dima806/facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection)")