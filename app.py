import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load saved model, scaler, and label encoder
model = joblib.load("style_classifier.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Simple color-to-style paint recommendation mapping
color_recommendations = {
    "bohemain": ["Terracotta Warmth", "Desert Sand", "Deep Mustard"],
    "industrial": ["Charcoal Grey", "Concrete Grey", "Steel Blue"],
    "minimalist": ["Pure White", "Soft Greige", "Pale Stone"],
    "traditional": ["Deep Burgundy", "Forest Green", "Warm Ivory"],
}

def extract_features(img, size=(64, 64)):
    img = img.convert("RGB").resize(size)
    img_array = np.array(img)
    hist_r, _ = np.histogram(img_array[:, :, 0], bins=16, range=(0, 256))
    hist_g, _ = np.histogram(img_array[:, :, 1], bins=16, range=(0, 256))
    hist_b, _ = np.histogram(img_array[:, :, 2], bins=16, range=(0, 256))
    features = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    features = features / features.sum()
    return features

st.set_page_config(page_title="RoomStyle AI", page_icon="🎨")
st.title("🎨 RoomStyle AI")
st.write("Upload a photo of a room to detect its interior design style and get matching paint color suggestions.")

uploaded_file = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Room", use_container_width=True)

    feat = extract_features(img).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    pred_encoded = model.predict(feat_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(feat_scaled)[0]
    confidence = probs[pred_encoded] * 100

    st.subheader(f"Predicted Style: **{pred_label.capitalize()}**")
    st.write(f"Confidence: {confidence:.1f}%")

    st.subheader("Suggested Paint Colors")
    for color in color_recommendations.get(pred_label, []):
        st.write(f"- {color}")

    with st.expander("See prediction breakdown across all styles"):
        for cls, p in zip(le.classes_, probs):
            st.write(f"{cls}: {p*100:.1f}%")
