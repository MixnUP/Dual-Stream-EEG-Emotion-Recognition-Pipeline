import os
import glob
import pickle

import mne
import numpy as np
import antropy as ent
from scipy.signal import welch
import pywt
import matplotlib.pyplot as plt
from PIL import Image
 
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------
# DATA LOADING & PREPROCESSING
# -------------------

def load_eeg_data(file_path, tmin=0, tmax=2.0):
    """
    Loads EEG data from a PhysioNet EEG Motor Movement/Imagery EDF file.

    IMPORTANT:
    - eegmmidb EDF files contain an annotation channel with labels like 'T0', 'T1', 'T2'.
    - The separate .edf.event files contain identical info but are NOT needed for MNE.
    
    This loader:
      • Loads only the EDF file
      • Extracts events from annotations
      • Converts annotation labels to integers
      • Creates MNE Epochs
    """
    try:
        print(f"[INFO] Loading EDF: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Filtering recommended for this dataset
        raw.filter(1., 45., verbose=False)
        raw.notch_filter([50, 60], verbose=False)

        # ---------------------------------------------
        # Extract events from EDF annotations
        # ---------------------------------------------
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # event_id example:
        # {'T0': 1, 'T1': 2, 'T2': 3}
        print(f"[INFO] Annotation event_id map: {event_id}")

        if len(events) == 0:
            print(f"[WARN] No events found in annotations for {file_path}.")
            return None, None

        # ---------------------------------------------
        # Epoching (each event creates 1 epoch)
        # ---------------------------------------------
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )

        print(f"[INFO] Loaded {len(epochs)} epochs from {os.path.basename(file_path)}")
        return epochs, event_id

    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None, None


def preprocess_epochs(epochs):
    """
    Applies ICA to epoched EEG data when feasible.

    Args:
        epochs (mne.Epochs): Epoched EEG data.

    Returns:
        mne.Epochs: Preprocessed epoched EEG data.
    """
    if epochs is None:
        return None

    print(f"[INFO] Applying ICA to {len(epochs)} epochs...")
    try:
        picks_eeg = mne.pick_types(epochs.info, eeg=True, exclude="bads")

        # Heuristic: if too few EEG channels, ICA is pointless / unstable
        if len(picks_eeg) < max(4, epochs.info["nchan"] // 3):
            print("[WARN] Not enough EEG channels for reliable ICA. Skipping ICA.")
            return epochs

        ica = mne.preprocessing.ICA(
            n_components=min(15, len(picks_eeg)),
            max_iter=500,
            random_state=99,
            verbose=False,
        )

        # Use decimation to speed up and stabilize ICA fitting
        ica.fit(
            epochs.copy().pick_types(eeg=True, meg=False, stim=False, exclude="bads"),
            decim=3,
        )

        epochs_cleaned = ica.apply(epochs.copy(), verbose=False)
        print("[INFO] ICA applied successfully.")
        return epochs_cleaned

    except Exception as e:
        print(f"[WARN] Error applying ICA: {e}. Returning original epochs.")
        return epochs


def segment_epochs_into_windows(epochs, window_size=2.0, overlap=0.5):
    """
    Segments MNE Epochs into smaller overlapping windows.

    Args:
        epochs (mne.Epochs): The epoched EEG data.
        window_size (float): The size of each window in seconds.
        overlap (float): The overlap between consecutive windows (0.0 to 1.0).

    Returns:
        list: List of (window_data, event_id_int) tuples.
              window_data: ndarray (n_channels, n_samples)
    """
    if epochs is None:
        return []

    sfreq = epochs.info["sfreq"]
    window_samples = int(window_size * sfreq)
    step_samples = max(1, int(window_samples * (1 - overlap)))

    all_windows_data = []

    for i, epoch in enumerate(epochs.get_data(copy=True)):  # (n_channels, n_samples)
        n_samples_epoch = epoch.shape[1]

        start_sample = 0
        while (start_sample + window_samples) <= n_samples_epoch:
            end_sample = start_sample + window_samples
            window_data = epoch[:, start_sample:end_sample]
            event_id_int = epochs.events[i, 2]
            all_windows_data.append((window_data, event_id_int))
            start_sample += step_samples

    print(
        f"[INFO] Segmented epochs into {len(all_windows_data)} windows "
        f"(window={window_size}s, overlap={overlap * 100:.1f}%)"
    )
    return all_windows_data


# -------------------
# DSP FEATURES (Entropy + Bandpower)
# -------------------

def compute_dsp_features_for_window(window_data, sfreq):
    """
    Computes DSP (entropy + bandpower) features for a single window.

    Args:
        window_data (ndarray): EEG window (n_channels, n_samples).
        sfreq (float): Sampling frequency.

    Returns:
        ndarray: 1D array of DSP features.
    """
    features = []

    for channel_data in window_data:
        channel_data_1d = np.squeeze(channel_data)

        if channel_data_1d.ndim > 1:
            channel_data_1d = channel_data_1d.flatten()

        if len(channel_data_1d) < 3:
            # Too short for entropy/Welch – pad with zeros
            features.extend([0.0] * 8)  # 3 entropies + 5 bandpowers
            continue

        try:
            # Entropy-based features
            features.append(ent.sample_entropy(channel_data_1d))
            features.append(ent.app_entropy(channel_data_1d))
            # Spectral entropy as a "differential-like" descriptor
            features.append(
                ent.spectral_entropy(
                    channel_data_1d, sfreq, method="welch", normalize=True
                )
            )

            # Bandpower using Welch PSD
            nperseg_val = min(len(channel_data_1d), int(sfreq * 2))
            if nperseg_val < 4:
                features.extend([0.0] * 5)
                continue

            freqs, psd = welch(channel_data_1d, sfreq, nperseg=nperseg_val)

            def bandpower(psd, freqs, band):
                idx = np.logical_and(freqs >= band[0], freqs <= band[1])
                if not np.any(idx):
                    return 0.0
                power = np.trapz(psd[idx], freqs[idx])
                total_power = np.trapz(psd, freqs)
                return power / total_power if total_power > 0 else 0.0

            features.append(bandpower(psd, freqs, [1, 4]))   # Delta
            features.append(bandpower(psd, freqs, [4, 8]))   # Theta
            features.append(bandpower(psd, freqs, [8, 12]))  # Alpha
            features.append(bandpower(psd, freqs, [12, 30])) # Beta
            features.append(bandpower(psd, freqs, [30, 45])) # Gamma

        except Exception as e:
            print(f"[WARN] Error computing DSP features for a channel: {e}. Using zeros.")
            features.extend([0.0] * 8)

    return np.array(features)


# -------------------
# CWT SCALOGRAM GENERATION
# -------------------

def generate_cwt_scalogram(channel_data, sfreq, img_size=(128, 128), wavelet="morl"):
    fig = None
    try:
        channel_data = np.asarray(channel_data).flatten()
        if len(channel_data) < 2:
            print("[WARN] Channel data too short for CWT.")
            return None

        scales = np.arange(1, 128)
        coefficients, freqs = pywt.cwt(
            channel_data, scales, wavelet, sampling_period=1/sfreq
        )
        scalogram_data = np.abs(coefficients)

        fig = plt.figure(
            frameon=False,
            figsize=(img_size[0]/100, img_size[1]/100),
            dpi=100
        )
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(
            scalogram_data,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[0, len(channel_data)/sfreq, freqs.min(), freqs.max()],
        )

        fig.canvas.draw()

        # Modern Matplotlib API (3.10 compatible)
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_array = buf[:, :, :3]  # keep RGB, drop alpha

        img_pil = Image.fromarray(img_array)
        img_resized = img_pil.resize(img_size, Image.LANCZOS)

        return np.array(img_resized)

    except Exception as e:
        print(f"[WARN] Error generating CWT scalogram: {e}")
        return None

    finally:
        if fig is not None:
            plt.close(fig)


# -------------------
# SHALLOW CNN (PyTorch)
# -------------------

class ShallowCNN(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # (B, 64)
        return x


def build_shallow_cnn(input_shape=(128, 128, 3), device=None):
    """
    Builds a shallow CNN model for feature extraction using PyTorch.

    Args:
        input_shape (tuple): (H, W, C) of input images.
        device (str or torch.device, optional): Device to use.

    Returns:
        (model, device): PyTorch model and device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShallowCNN(input_channels=input_shape[2]).to(device)
    model.eval()
    print(f"[INFO] ShallowCNN initialized on device: {device}")
    return model, device


def extract_cnn_features(scalogram_image, cnn_model, device):
    """
    Extracts CNN features from a scalogram image using a PyTorch CNN.

    Args:
        scalogram_image (ndarray): RGB image (H, W, 3).
        cnn_model (nn.Module): PyTorch CNN model.
        device (torch.device): Device for inference.

    Returns:
        ndarray or None: 1D CNN feature vector.
    """
    if scalogram_image is None:
        return None
    try:
        img = np.asarray(scalogram_image).astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad():
            features = cnn_model(img_tensor)  # (1, F)
        return features.cpu().numpy().flatten()

    except Exception as e:
        print(f"[WARN] Error extracting CNN features: {e}. Returning None.")
        return None


# -------------------
# EDF DATA COLLECTION
# -------------------

def collect_all_edf_data(data_dirs, tmin=0, tmax=2.0):
    """
    Collects and processes all EDF files from specified directories.

    Args:
        data_dirs (list): Directories to search for EDF files.
        tmin (float): Epoch start time.
        tmax (float): Epoch end time.

    Returns:
        list of (epochs, subject_id)
    """
    all_data = []

    for data_dir in data_dirs:
        edf_files = glob.glob(os.path.join(data_dir, "**", "*.edf"), recursive=True)
        print(f"[INFO] Found {len(edf_files)} EDF files in {data_dir}")

        for edf_file in edf_files:
            epochs, event_id = load_eeg_data(edf_file, tmin, tmax)
            if epochs is not None:
                preprocessed_epochs = preprocess_epochs(epochs)
                if preprocessed_epochs is not None:
                    # Extract subject ID (e.g., 'S001' from 'S001R04.edf')
                    base = os.path.basename(edf_file)
                    subject_id = base.split("R")[0]
                    all_data.append((preprocessed_epochs, subject_id))

    return all_data


# -------------------
# CLASSIFICATION (LOSO + METRICS)
# -------------------

def train_and_evaluate_loso(X, y, groups):
    """
    Leave-One-Group-Out CV with SVM classifier.

    Args:
        X (ndarray): Feature vectors.
        y (ndarray): Labels.
        groups (ndarray): Group labels (subject IDs).

    Returns:
        (predictions, y_test): Concatenated arrays from all folds.
    """
    if X is None or y is None or groups is None or len(X) == 0:
        print("[ERROR] No features/labels/groups. Skipping classification.")
        return None, None

    unique_groups = np.unique(groups)
    print(f"[INFO] Starting LOSO with {len(unique_groups)} groups (subjects).")

    logo = LeaveOneGroupOut()
    all_predictions = []
    all_y_test = []
    processed_subjects = set()

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test_fold = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)
        predictions_fold = model.predict(X_test_scaled)

        all_predictions.extend(predictions_fold)
        all_y_test.extend(y_test_fold)

        subject_out = groups[test_idx][0]
        if subject_out not in processed_subjects:
            acc = accuracy_score(y_test_fold, predictions_fold)
            print(f"  - Held-out subject {subject_out} | Accuracy: {acc:.4f}")
            processed_subjects.add(subject_out)

    print("[INFO] LOSO cross-validation complete.")
    return np.array(all_predictions), np.array(all_y_test)


def evaluate_classifier_performance(y_true, y_pred, labels=None):
    """
    Computes metrics and confusion matrix figure.

    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        labels (list or ndarray): Class labels for confusion matrix.

    Returns:
        (metrics_dict, fig or None)
    """
    if y_true is None or y_pred is None or len(y_true) == 0:
        print("[WARN] No labels for evaluation. Returning empty metrics.")
        return {}, None

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    return metrics, fig


# -------------------
# MAIN PIPELINE
# -------------------

def run_eeg_pipeline(
    edf_data_directories,
    window_size=2.0,
    overlap=0.5,
    st_progress_bar=None,
    st_status_text=None,
):
    """
    Executes the EEG emotion recognition pipeline.

    Args:
        edf_data_directories (list): List of directories to search for EDF files.
        window_size (float): Window size for segmentation in seconds.
        overlap (float): Overlap between windows (0-1).
        st_progress_bar: Optional Streamlit progress element.
        st_status_text: Optional Streamlit text element.

    Returns:
        dict: Results dictionary with features, metrics, model, etc.
    """
    results = {}

    def log(msg):
        if st_status_text is not None:
            st_status_text.text(msg)
        else:
            print(msg)

    log("[PIPELINE] Starting data collection & preprocessing...")
    all_processed_data = collect_all_edf_data(edf_data_directories)
    total_files = len(all_processed_data)
    log(f"[PIPELINE] Data collection done. Processed file sets: {total_files}")

    all_dsp_features_list = []
    all_cnn_features_list = []
    all_labels_list = []
    all_subjects_list = []
    all_scalograms_list = []

    log("[PIPELINE] Initializing CNN feature extractor...")
    cnn_model, device = build_shallow_cnn()

    for i, (epochs_set, subject_id) in enumerate(all_processed_data):
        if total_files > 0 and st_progress_bar is not None:
            st_progress_bar.progress(i / max(1, total_files))

        log(f"[PIPELINE] Processing subject {subject_id} ({i+1}/{total_files})...")
        windows_with_labels = segment_epochs_into_windows(
            epochs_set,
            window_size=window_size,
            overlap=overlap,
        )
        sfreq = epochs_set.info["sfreq"]

        for window_data, event_id_int in windows_with_labels:
            dsp_vec = compute_dsp_features_for_window(window_data, sfreq)

            # CNN features from CWT of first channel (can be extended later)
            cnn_vec = None
            if window_data.shape[0] > 0:
                channel_data_1d = window_data[0, :]
                scalogram_image = generate_cwt_scalogram(channel_data_1d, sfreq)
                if scalogram_image is not None:
                    cnn_vec = extract_cnn_features(
                        scalogram_image, cnn_model, device
                    )
                    if cnn_vec is not None:
                        all_scalograms_list.append(scalogram_image)

            if dsp_vec is not None and cnn_vec is not None:
                all_dsp_features_list.append(dsp_vec)
                all_cnn_features_list.append(cnn_vec)
                all_labels_list.append(event_id_int)
                all_subjects_list.append(subject_id)

    if st_progress_bar is not None:
        st_progress_bar.progress(0.8)

    log("[PIPELINE] Aggregating features...")

    all_dsp_features = np.array(all_dsp_features_list)
    all_cnn_features = np.array(all_cnn_features_list)
    all_labels = np.array(all_labels_list)
    all_subjects = np.array(all_subjects_list)
    all_cwt_scalograms = np.array(all_scalograms_list, dtype=object)

    results.update(
        {
            "entropy_features": all_dsp_features,
            "cnn_features": all_cnn_features,
            "all_labels": all_labels,
            "cwt_scalograms": all_cwt_scalograms,
        }
    )

    log(
        f"[PIPELINE] Feature extraction complete. "
        f"Valid windows: {len(all_labels)}"
    )

    if (
        all_dsp_features.shape[0] > 0
        and all_dsp_features.shape[0] == all_cnn_features.shape[0]
    ):
        fused_feature_vectors = np.concatenate(
            (all_dsp_features, all_cnn_features), axis=1
        )
        results["fused_feature_vectors"] = fused_feature_vectors
        log(f"[PIPELINE] Fused feature vectors shape: {fused_feature_vectors.shape}")

        unique_subjects = np.unique(all_subjects)
        if len(unique_subjects) > 1:
            log(
                f"[PIPELINE] {len(unique_subjects)} subjects found. "
                "Running LOSO cross-validation..."
            )
            predictions, y_test = train_and_evaluate_loso(
                fused_feature_vectors, all_labels, all_subjects
            )
            if predictions is not None:
                unique_labels = np.unique(all_labels)
                metrics, cm_fig = evaluate_classifier_performance(
                    y_test,
                    predictions,
                    labels=unique_labels,
                )
                results.update(
                    {
                        "predictions": predictions,
                        "y_test": y_test,
                        "metrics": metrics,
                        "confusion_matrix_fig": cm_fig,
                    }
                )
                log("[PIPELINE] Cross-validation completed.")
        else:
            log("[WARN] Only one subject. Skipping LOSO cross-validation.")
            results.update(
                {
                    "metrics": {},
                    "confusion_matrix_fig": None,
                    "predictions": None,
                    "y_test": None,
                }
            )

        if st_progress_bar is not None:
            st_progress_bar.progress(0.9)

        # Train final model on all data
        log("[PIPELINE] Training final SVM model on all fused features...")
        final_scaler = StandardScaler()
        X_scaled_final = final_scaler.fit_transform(fused_feature_vectors)
        final_model = SVC(kernel="rbf", probability=True, random_state=42)
        final_model.fit(X_scaled_final, all_labels)
        results["trained_model"] = final_model
        log("[PIPELINE] Final model trained.")

        # Save artifacts
        log("[PIPELINE] Saving artifacts to disk...")
        np.save("entropy_features.npy", all_dsp_features)
        np.save("cnn_feature_vectors.npy", all_cnn_features)
        np.save("cwt_scalograms.npy", all_cwt_scalograms)

        model_and_scaler = {"model": final_model, "scaler": final_scaler}
        with open("emotion_model.pkl", "wb") as f:
            pickle.dump(model_and_scaler, f)
        log("[PIPELINE] Artifacts saved successfully.")

    else:
        log("[ERROR] Feature extraction produced no usable results. Aborting.")
        results["fused_feature_vectors"] = None

    if st_progress_bar is not None:
        st_progress_bar.progress(1.0)

    return results


# Optional helper (unchanged logic, just cleaned import)
def concatenate_all_epochs(all_processed_epochs):
    from mne import concatenate_epochs

    if all_processed_epochs:
        try:
            all_epochs_concatenated = concatenate_epochs(all_processed_epochs)
            print(
                f"[INFO] Total concatenated epochs after preprocessing: "
                f"{len(all_epochs_concatenated)}"
            )
            print(f"[DEBUG] Concatenated epochs info: {all_epochs_concatenated.info}")
            return all_epochs_concatenated
        except ValueError as e:
            print(
                f"[WARN] Could not concatenate epochs: {e}. "
                "Channels may differ between files."
            )
            print("[INFO] Proceeding with individual epoch sets.")
    else:
        print("[WARN] No epochs processed successfully.")
    return None
