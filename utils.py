import numpy as np
import os
import pickle
from typing import Dict, Tuple, Optional, Union
import mne
from pathlib import Path

# File I/O Utilities
def load_npy_file(filepath: str) -> np.ndarray:
    """Load a single .npy file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return np.load(filepath)

def load_model(filepath: str) -> Dict:
    """Load the saved model and scaler from a .pkl file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Data Processing
def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to zero mean and unit variance."""
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

def get_data_paths(directory: str, extension: str = '.edf') -> list:
    """Get all files with given extension from directory."""
    return list(Path(directory).rglob(f'*{extension}'))

# EEG Specific
def load_eeg_data(filepath: str) -> mne.io.Raw:
    """Load EEG data from .edf file using MNE."""
    if not filepath.endswith('.edf'):
        raise ValueError("File must be in .edf format")
    return mne.io.read_raw_edf(filepath, preload=True)

def preprocess_eeg(raw: mne.io.Raw, l_freq: float = 1.0, h_freq: float = 45.0) -> mne.io.Raw:
    """Apply basic preprocessing to EEG data."""
    # Bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    # Notch filter for powerline noise
    raw.notch_filter(freqs=[50, 60], verbose=False)
    return raw

# Feature Handling
def combine_features(feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine multiple feature arrays into a single feature matrix."""
    return np.concatenate([f for f in feature_dict.values()], axis=1)

def save_features(features: np.ndarray, filename: str, output_dir: str = 'features') -> str:
    """Save features to disk and return the filepath."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.npy")
    np.save(filepath, features)
    return filepath

if __name__ == "__main__":
    # Example usage
    print("Utils module loaded successfully.")
