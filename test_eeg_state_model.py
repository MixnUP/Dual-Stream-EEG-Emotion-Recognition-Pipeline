import joblib
import numpy as np
import os

MODEL = "eeg_state_model.pkl"
DSP = "entropy_features.npy"
CNN = "cnn_feature_vectors.npy"

LABEL_MAP = {
    1: "Rest (T0)",
    2: "Hand Imagery (T1)",
    3: "Foot Imagery (T2)"
}

# -----------------------------------
# Load model
# -----------------------------------
if not os.path.exists(MODEL):
    print("Error: Model file not found.")
    exit()

artifacts = joblib.load(MODEL)
model = artifacts["model"]
scaler = artifacts["scaler"]
print("Model & scaler loaded!")

# -----------------------------------
# Load features
# -----------------------------------
if not os.path.exists(DSP) or not os.path.exists(CNN):
    print("Error: DSP or CNN feature file missing.")
    exit()

dsp = np.load(DSP)
cnn = np.load(CNN)

print("DSP shape:", dsp.shape)
print("CNN shape:", cnn.shape)

if dsp.shape[0] != cnn.shape[0]:
    print("Error: DSP and CNN feature counts do not match.")
    exit()

# First sample for testing
dsp_sample = dsp[0:1]
cnn_sample = cnn[0:1]

# Fuse
fused = np.concatenate([dsp_sample, cnn_sample], axis=1)
print("Fused feature shape:", fused.shape)

# Scale
scaled = scaler.transform(fused)

# Predict
pred = model.predict(scaled)[0]
print("Predicted class:", pred)
print("Label interpretation:", LABEL_MAP.get(pred, "Unknown"))
