import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd

# ─────────────────────────────────────────
# Load model ONCE when app starts
# demand_model.pkl = our trained GradientBoosting model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/demand_model.pkl")
    return model

# ─────────────────────────────────────────
# Load MobileNetV2 ONCE
# We use it as feature extractor — same as during training
# include_top=False → we don't need classification layer
# pooling='avg' → converts feature map to 1280 numbers
# ─────────────────────────────────────────
@st.cache_resource
def load_mobilenet():
    mobilenet = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    return mobilenet

# ─────────────────────────────────────────
# Extract same 1280 features from new image
# MUST be identical to training — same size, same preprocessing
# ─────────────────────────────────────────
def extract_features(img, mobilenet):
    # Resize to 224x224 — MobileNetV2 requirement
    img = img.resize((224, 224))
    
    # Convert to RGB — in case image is RGBA or grayscale
    img = img.convert("RGB")
    
    # Convert to array
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess — normalizes pixel values for MobileNetV2
    x = preprocess_input(x)
    
    # Extract 1280 features
    features = mobilenet.predict(x, verbose=0)
    return features.flatten()


# ─────────────────────────────────────────
# Load sales stats to show context
# Shows min/max/avg sales so prediction makes sense
# ─────────────────────────────────────────
@st.cache_data
def load_sales_stats():
    df = pd.read_csv("data/aggregated_sales.csv")
    return df

# ─────────────────────────────────────────
# UI STARTS HERE
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Dress Demand Predictor",
    page_icon="👗",
    layout="centered"
)

st.title("👗 Dress Demand Predictor")
st.markdown("Upload a dress photo to predict how many pieces it will sell.")
st.divider()

# Load everything
model = load_model()
mobilenet = load_mobilenet()
df_stats = load_sales_stats()

# ─────────────────────────────────────────
# File uploader — accepts jpg/png/jpeg
# User uploads NEW dress photo here
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Dress Photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Dress", width=300)
    st.divider()

    # Predict button
    if st.button("🔍 Predict Demand", use_container_width=True):

        with st.spinner("Analyzing dress..."):

            # Extract features from uploaded image
            features = extract_features(img, mobilenet)

            # Predict using trained model
            predicted_qty = model.predict([features])[0]
            predicted_qty = max(0, round(predicted_qty))

        # ─────────────────────────────────────────
        # Show results in clean metrics
        # ─────────────────────────────────────────
        st.success("✅ Prediction Complete!")
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="📦 Predicted Demand",
                value=f"{predicted_qty} pieces"
            )

        with col2:
            avg_qty = int(df_stats["total_qty"].mean())
            st.metric(
                label="📊 Avg Sales (all designs)",
                value=f"{avg_qty} pieces"
            )

        with col3:
            avg_rate = int(df_stats["avg_rate"].mean())
            st.metric(
                label="💰 Avg Selling Rate",
                value=f"₹{avg_rate}"
            )

        st.divider()

        # ─────────────────────────────────────────
        # Demand category helps business understand
        # low/medium/high demand in simple words
        # ─────────────────────────────────────────
        if predicted_qty < 20:
            st.warning("📉 Low Demand — Manufacture limited stock (10–20 pieces)")
        elif predicted_qty < 60:
            st.info("📈 Medium Demand — Manufacture moderate stock (40–60 pieces)")
        else:
            st.success("🔥 High Demand — Manufacture higher stock (80–100 pieces)")

        # ─────────────────────────────────────────
        # Show sales data context
        # Helps owner understand prediction vs reality
        # ─────────────────────────────────────────
        st.divider()
        st.subheader("📊 Sales Data Overview")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Total Designs in Data", len(df_stats))
        with col5:
            st.metric("Highest Selling Design", f"{int(df_stats['total_qty'].max())} pcs")
        with col6:
            st.metric("Lowest Selling Design", f"{int(df_stats['total_qty'].min())} pcs")