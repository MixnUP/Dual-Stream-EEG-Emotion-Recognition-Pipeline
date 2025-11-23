import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Import the pipeline and utilities
from eeg_pipeline import run_eeg_pipeline
from utils import (
    load_npy_file,
    load_model,
    load_eeg_data,
    preprocess_eeg,
    combine_features,
    save_features,
    get_data_paths
)

st.set_page_config(layout="wide")

st.title("Dual-Stream EEG Emotion Recognition Pipeline")
st.write("Run the EEG processing and classification pipeline using DSP + CWT + CNN Fusion.")

# ---------------------------
#   CONFIGURATION
# ---------------------------

st.header("Pipeline Configuration")

# TODO: Make these configurable later if needed
edf_data_directories = [
    "data/files/S001",
    "data/files/S002",
    "data/files/S003",
    "data/files/S004",
    "data/files/S005",
    "data/files/S006",
    "data/files/S007",
    "data/files/S008",
    "data/files/S009",
    "data/files/S010",
]

# Verify data directories exist
valid_dirs = []
for dir_path in edf_data_directories:
    if os.path.exists(dir_path):
        valid_dirs.append(dir_path)
    else:
        st.warning(f"Directory not found: {dir_path}")

if not valid_dirs:
    st.error("No valid data directories found. Please check your configuration.")
    st.stop()

st.write(f"**Found {len(valid_dirs)} valid data directories**")
st.write(f"**Scanning Directories:** `{', '.join(valid_dirs)}`")

window_size = st.slider(
    "Window Size (seconds)", 
    min_value=0.5, 
    max_value=5.0, 
    value=2.0, 
    step=0.1
)

overlap = st.slider(
    "Window Overlap (0.0 - 1.0)", 
    min_value=0.0, 
    max_value=0.95, 
    value=0.5, 
    step=0.05
)

# ---------------------------
#   RUN PIPELINE BUTTON
# ---------------------------

if st.button("Run EEG Pipeline"):
    st.info("Pipeline started... This may take several minutes.")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Initializing...")

    import traceback

    try:
        results = run_eeg_pipeline(
            edf_data_directories,
            window_size,
            overlap,
            st_progress_bar=progress_bar,
            st_status_text=status_text
        )

        # ---------------------------
        #   PIPELINE FINISHED
        # ---------------------------

        if not results or results.get("fused_feature_vectors") is None:
            st.error("Pipeline finished, but no usable feature vectors were generated.")
            st.stop()

        st.success("Pipeline completed successfully!")
        status_text.text("Pipeline complete.")
        progress_bar.progress(1.0)

        # ---------------------------
        #   METRICS
        # ---------------------------

        st.header("Classification Performance")

        metrics = results.get("metrics", {})
        if metrics:
            st.subheader("Performance Metrics")
            st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
            st.write(f"**Precision:** {metrics['precision']:.4f}")
            st.write(f"**Recall:** {metrics['recall']:.4f}")
            st.write(f"**F1-score:** {metrics['f1_score']:.4f}")
        else:
            st.warning("Metrics unavailable (LOSO may have been skipped due to only one subject).")

        # ---------------------------
        #   CONFUSION MATRIX
        # ---------------------------

        cm_fig = results.get("confusion_matrix_fig")
        if cm_fig:
            st.subheader("Confusion Matrix")
            st.pyplot(cm_fig)
            plt.close(cm_fig)
        else:
            st.info("Confusion matrix not available.")

        # ---------------------------
        #   DOWNLOAD ARTIFACTS
        # ---------------------------

        st.header("Download Artifacts")

        # Create a temporary directory for downloads if it doesn't exist
        os.makedirs('downloads', exist_ok=True)

        def add_download_button(filename, label):
            if not os.path.exists(filename):
                st.warning(f"{label} file not found.")
                return
                
            try:
                # For .npy files, provide a preview
                if filename.endswith('.npy'):
                    data = load_npy_file(filename)
                    with st.expander(f"Preview {label}"):
                        st.write(f"Shape: {data.shape}")
                        st.write(f"First 5 rows:")
                        st.dataframe(data[:5] if len(data.shape) > 1 else data[:5].reshape(1, -1))
                # For model files, show basic info
                elif filename.endswith('.pkl'):
                    model_data = load_model(filename)
                    with st.expander(f"Model Info"):
                        st.write(f"Model type: {type(model_data.get('model')).__name__}")
                        st.write(f"Features shape: {model_data.get('n_features_in_', 'N/A')}")
                        
                # Create download button
                with open(filename, 'rb') as f:
                    st.download_button(
                        label=f"Download {label}",
                        data=f,
                        file_name=os.path.basename(filename),
                        mime='application/octet-stream'
                    )
                    
            except Exception as e:
                st.error(f"Error loading {filename}: {str(e)}")

        # List available artifacts
        artifacts = {
            'entropy_features.npy': 'Entropy Features',
            'cnn_feature_vectors.npy': 'CNN Feature Vectors',
            'cwt_scalograms.npy': 'CWT Scalograms',
            'emotion_model.pkl': 'Trained Model'
        }

        # Show download buttons with previews
        for filename, label in artifacts.items():
                with open(filename, "rb") as f:
                    st.download_button(
                        label=label,
                        data=f.read(),
                        file_name=filename
                    )

        add_download_button("entropy_features.npy", "Entropy Features (NPY)")
        add_download_button("cnn_feature_vectors.npy", "CNN Feature Vectors (NPY)")
        add_download_button("cwt_scalograms.npy", "CWT Scalograms (NPY)")
        add_download_button("emotion_model.pkl", "Trained SVM Model (PKL)")

        st.balloons()

    except Exception as e:
        status_text.error(f"An error occurred during pipeline execution: {e}")
        st.error("Pipeline crashed. See details below:")
        st.code(traceback.format_exc())
        progress_bar.progress(1.0)
