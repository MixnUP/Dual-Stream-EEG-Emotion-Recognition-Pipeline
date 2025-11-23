import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "eeg_state_model.pkl"
DSP_FILE = "entropy_features.npy"
CNN_FILE = "cnn_feature_vectors.npy"

LABEL_MAP = {
    1: "Rest (T0)",
    2: "Hand Imagery (T1)",
    3: "Foot Imagery (T2)",
}

# -----------------------------
# Load model + scaler
# -----------------------------
@st.cache_resource
def load_model(model_path):
    try:
        artifacts = joblib.load(model_path)
        return artifacts["model"], artifacts["scaler"]
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# -----------------------------
# Load feature arrays
# -----------------------------
def load_feature_file(path):
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return None

    try:
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("EEG State Classification Predictor (DSP + CWT-CNN Fusion)")

model, scaler = load_model(MODEL_PATH)

if model is None or scaler is None:
    st.stop()

st.success("Model + scaler loaded successfully!")

# -----------------------------
# Load DSP + CNN Features
# -----------------------------
dsp_features = load_feature_file(DSP_FILE)
cnn_features = load_feature_file(CNN_FILE)

if dsp_features is None or cnn_features is None:
    st.stop()

if dsp_features.shape[0] != cnn_features.shape[0]:
    st.error("Mismatch: DSP and CNN features have different sample counts.")
    st.write("DSP shape:", dsp_features.shape)
    st.write("CNN shape:", cnn_features.shape)
    st.stop()

st.info(f"Loaded DSP features: {dsp_features.shape}")
st.info(f"Loaded CNN features: {cnn_features.shape}")

# -----------------------------
# Select sample index
# -----------------------------
sample_idx = st.number_input(
    "Select sample to classify:",
    min_value=0,
    max_value=dsp_features.shape[0] - 1,
    value=0
)

st.write(f"Predicting sample #{sample_idx}")

dsp_sample = dsp_features[sample_idx].reshape(1, -1)
cnn_sample = cnn_features[sample_idx].reshape(1, -1)

# Exactly the same fusion as training
fused = np.concatenate([dsp_sample, cnn_sample], axis=1)
st.write("Fused feature vector shape:", fused.shape)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict EEG State"):
    try:
        scaled = scaler.transform(fused)
        pred = model.predict(scaled)
        predicted_label = LABEL_MAP.get(pred[0], "Unknown")

        st.success(f"🧠 Predicted EEG State: **{predicted_label}**")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
