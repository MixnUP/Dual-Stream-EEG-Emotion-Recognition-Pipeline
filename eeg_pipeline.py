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
    try:
        print(f"[INFO] Loading EDF: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        raw.filter(1., 45., verbose=False)
        raw.notch_filter([50, 60], verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        print(f"[INFO] Annotation event_id map: {event_id}")

        if len(events) == 0:
            print(f"[WARN] No events found in annotations for {file_path}.")
            return None, None

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
    if epochs is None:
        return None

    print(f"[INFO] Applying ICA to {len(epochs)} epochs...")
    try:
        picks_eeg = mne.pick_types(epochs.info, eeg=True, exclude="bads")

        if len(picks_eeg) < max(4, epochs.info["nchan"] // 3):
            print("[WARN] Not enough EEG channels for reliable ICA. Skipping ICA.")
            return epochs

        ica = mne.preprocessing.ICA(
            n_components=min(15, len(picks_eeg)),
            max_iter=500,
            random_state=99,
            verbose=False,
        )

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
    if epochs is None:
        return []

    sfreq = epochs.info["sfreq"]
    window_samples = int(window_size * sfreq)
    step_samples = max(1, int(window_samples * (1 - overlap)))

    all_windows_data = []

    for i, epoch in enumerate(epochs.get_data(copy=True)):
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
    features = []

    for channel_data in window_data:
        channel_data_1d = np.squeeze(channel_data)

        if channel_data_1d.ndim > 1:
            channel_data_1d = channel_data_1d.flatten()

        if len(channel_data_1d) < 3:
            features.extend([0.0] * 8)
            continue

        try:
            features.append(ent.sample_entropy(channel_data_1d))
            features.append(ent.app_entropy(channel_data_1d))
            features.append(
                ent.spectral_entropy(
                    channel_data_1d, sfreq, method="welch", normalize=True
                )
            )

            nperseg_val = min(len(channel_data_1d), int(sfreq * 2))
            if nperseg_val < 4:
                features.extend([0.0] * 5)
                continue

            freqs, psd = welch(channel_data_1d, sfreq, nperseg=nperseg_val)

            def bp(psd, freqs, band):
                idx = np.logical_and(freqs >= band[0], freqs <= band[1])
                if not np.any(idx):
                    return 0.0
                p = np.trapz(psd[idx], freqs[idx])
                t = np.trapz(psd, freqs)
                return p / t if t > 0 else 0.0

            features.append(bp(psd, freqs, [1, 4]))
            features.append(bp(psd, freqs, [4, 8]))
            features.append(bp(psd, freqs, [8, 12]))
            features.append(bp(psd, freqs, [12, 30]))
            features.append(bp(psd, freqs, [30, 45]))

        except:
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
            return None

        scales = np.arange(1, 128)
        coeffs, freqs = pywt.cwt(
            channel_data, scales, wavelet, sampling_period=1/sfreq
        )
        scalogram = np.abs(coeffs)

        fig = plt.figure(
            frameon=False,
            figsize=(img_size[0]/100, img_size[1]/100),
            dpi=100
        )
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(
            scalogram,
            cmap="viridis",
            aspect="auto",
            origin="lower"
        )

        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba())
        img_array = buf[:, :, :3]

        img_pil = Image.fromarray(img_array)
        return np.array(img_pil.resize(img_size, Image.LANCZOS))

    except Exception as e:
        print(f"[WARN] Error generating CWT scalogram: {e}")
        return None

    finally:
        if fig is not None:
            plt.close(fig)


# -------------------
# SHALLOW CNN
# -------------------

class ShallowCNN(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        return x.view(x.size(0), -1)


def build_shallow_cnn(input_shape=(128, 128, 3), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShallowCNN(input_channels=input_shape[2]).to(device)
    model.eval()
    return model, device


def extract_cnn_features(img, cnn_model, device):
    if img is None:
        return None
    try:
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = cnn_model(tensor)
        return feat.cpu().numpy().flatten()
    except:
        return None


# -------------------
# EDF LOADING
# -------------------

def collect_all_edf_data(data_dirs, tmin=0, tmax=2.0):
    data = []
    for d in data_dirs:
        files = glob.glob(os.path.join(d, "**", "*.edf"), recursive=True)
        print(f"[INFO] Found {len(files)} EDF files in {d}")
        for f in files:
            epochs, _ = load_eeg_data(f, tmin, tmax)
            if epochs is not None:
                cleaned = preprocess_epochs(epochs)
                if cleaned is not None:
                    subj = os.path.basename(f).split("R")[0]
                    data.append((cleaned, subj))
    return data


# -------------------
# LOSO CLASSIFICATION
# -------------------

def train_and_evaluate_loso(X, y, groups):
    if len(X) == 0:
        return None, None

    logo = LeaveOneGroupOut()
    all_preds = []
    all_true = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test_fold = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = SVC(kernel="rbf", probability=True)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        all_preds.extend(preds)
        all_true.extend(y_test_fold)

    return np.array(all_preds), np.array(all_true)


def evaluate_classifier_performance(y_true, y_pred, labels=None):
    if len(y_true) == 0:
        return {}, None

    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    return m, fig


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
    results = {}

    def log(msg):
        if st_status_text:
            st_status_text.text(msg)
        print(msg)

    log("[PIPELINE] Starting...")

    data = collect_all_edf_data(edf_data_directories)
    total = len(data)

    all_dsp = []
    all_cnn = []
    all_labels = []
    all_subjects = []
    all_scalograms = []

    cnn_model, device = build_shallow_cnn()

    for i, (epochs_set, subj) in enumerate(data):
        if st_progress_bar:
            st_progress_bar.progress(i / max(1, total))

        log(f"[PIPELINE] Subject {subj} ({i+1}/{total})")
        windows = segment_epochs_into_windows(epochs_set, window_size, overlap)
        sfreq = epochs_set.info["sfreq"]

        for window_data, label in windows:
            dsp = compute_dsp_features_for_window(window_data, sfreq)

            cnn_feat = None
            if window_data.shape[0] > 0:
                ch = window_data[0, :]
                img = generate_cwt_scalogram(ch, sfreq)
                if img is not None:
                    cnn_feat = extract_cnn_features(img, cnn_model, device)
                    if cnn_feat is not None:
                        all_scalograms.append(img)

            if dsp is not None and cnn_feat is not None:
                all_dsp.append(dsp)
                all_cnn.append(cnn_feat)
                all_labels.append(label)
                all_subjects.append(subj)

    if st_progress_bar:
        st_progress_bar.progress(0.8)

    all_dsp = np.array(all_dsp)
    all_cnn = np.array(all_cnn)
    all_labels = np.array(all_labels)
    all_subjects = np.array(all_subjects)
    # Store scalograms as a regular numeric array (N, H, W, C) so they
    # can be loaded with np.load(..., allow_pickle=False)
    all_scalograms = np.array(all_scalograms) if all_scalograms else np.empty((0,))

    results["entropy_features"] = all_dsp
    results["cnn_features"] = all_cnn
    results["all_labels"] = all_labels
    results["cwt_scalograms"] = all_scalograms

    if len(all_dsp) > 0 and len(all_dsp) == len(all_cnn):
        fused = np.concatenate([all_dsp, all_cnn], axis=1)
        results["fused_feature_vectors"] = fused

        subjects = np.unique(all_subjects)
        if len(subjects) > 1:
            preds, y_test = train_and_evaluate_loso(fused, all_labels, all_subjects)
            if preds is not None:
                labels = np.unique(all_labels)
                metrics, cm_fig = evaluate_classifier_performance(
                    y_test, preds, labels
                )
                results["metrics"] = metrics
                results["confusion_matrix_fig"] = cm_fig

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(fused)
        model = SVC(kernel="rbf", probability=True)
        model.fit(X_scaled, all_labels)

        results["trained_model"] = model

        np.save("entropy_features.npy", all_dsp)
        np.save("cnn_feature_vectors.npy", all_cnn)
        np.save("cwt_scalograms.npy", all_scalograms)

        with open("eeg_state_model.pkl", "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)

    if st_progress_bar:
        st_progress_bar.progress(1.0)

    return results
