import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pywt import wavedec
from numba import jit

# Paths
ehg_folder_path = os.path.abspath("EHGs")  # Ensure correct absolute path
preterm_path = os.path.join(ehg_folder_path, "preterm")
term_path = os.path.join(ehg_folder_path, "term")

# Check if paths exist
if not os.path.exists(preterm_path):
    raise FileNotFoundError(f"Preterm folder not found: {preterm_path}")
if not os.path.exists(term_path):
    raise FileNotFoundError(f"Term folder not found: {term_path}")

# Get file lists
preterm_files = os.listdir(preterm_path)
term_files = os.listdir(term_path)

if not preterm_files:
    raise FileNotFoundError("No preterm files found.")
if not term_files:
    raise FileNotFoundError("No term files found.")

# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.3, highcut=3.0, fs=20.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# FIR filter
def fir_filter(data, lowcut=0.3, highcut=4.0, fs=20.0, numtaps=101):
    from scipy.signal import firwin, lfilter
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return lfilter(fir_coeff, 1.0, data)

# Load and process EHG signal
def load_ehg_signal(file_path):
    data = np.loadtxt(file_path)
    signal = data[:, 3]  # Extract 4th column (S1 channel)
    return signal

# Compute Wavelet Entropy
def compute_wavelet_entropy(signal, wavelet='db5', level=5):
    coeffs = wavedec(signal, wavelet, level=level)
    detail_coeffs = coeffs[1:]  # Exclude approximation coefficients
    energy = np.array([np.sum(np.square(c)) for c in detail_coeffs])
    probability = energy / np.sum(energy)
    return -np.sum(probability * np.log(probability))

# Function to compute entropy on epochs
def compute_entropy_on_epochs(signal, window_size=1000, overlap=False):
    step = window_size if not overlap else window_size // 2
    entropies = [compute_wavelet_entropy(signal[i:i + window_size]) for i in range(0, len(signal) - window_size, step)]
    return np.mean(entropies)

# Process first sample from each category
sample_preterm_signal = load_ehg_signal(os.path.join(preterm_path, preterm_files[0]))
sample_term_signal = load_ehg_signal(os.path.join(term_path, term_files[0]))

# Apply Butterworth Filter
sample_preterm_butter = bandpass_filter(sample_preterm_signal)
sample_term_butter = bandpass_filter(sample_term_signal)

# Apply FIR Filter
sample_preterm_fir = fir_filter(sample_preterm_signal)
sample_term_fir = fir_filter(sample_term_signal)

# Compute Entropy for different methods
we_preterm_butter = compute_wavelet_entropy(sample_preterm_butter)
we_term_butter = compute_wavelet_entropy(sample_term_butter)
we_preterm_fir = compute_wavelet_entropy(sample_preterm_fir)
we_term_fir = compute_wavelet_entropy(sample_term_fir)

# Compute Entropy for epochs
we_preterm_butter_epochs = compute_entropy_on_epochs(sample_preterm_butter, overlap=False)
we_term_butter_epochs = compute_entropy_on_epochs(sample_term_butter, overlap=False)
we_preterm_fir_epochs = compute_entropy_on_epochs(sample_preterm_fir, overlap=False)
we_term_fir_epochs = compute_entropy_on_epochs(sample_term_fir, overlap=False)

# Compute Entropy for overlapping epochs
we_preterm_butter_overlap = compute_entropy_on_epochs(sample_preterm_butter, overlap=True)
we_term_butter_overlap = compute_entropy_on_epochs(sample_term_butter, overlap=True)
we_preterm_fir_overlap = compute_entropy_on_epochs(sample_preterm_fir, overlap=True)
we_term_fir_overlap = compute_entropy_on_epochs(sample_term_fir, overlap=True)

# Create DataFrame for results
data = {
    "Filter": ["Butterworth", "Butterworth", "Butterworth", "FIR", "FIR", "FIR"],
    "Metric": ["Wavelet", "Non-Overlapping Epochs", "Overlapping Epochs", "Wavelet", "Non-Overlapping Epochs", "Overlapping Epochs"],
    "Term": [we_term_butter, we_term_butter_epochs, we_term_butter_overlap, we_term_fir, we_term_fir_epochs, we_term_fir_overlap],
    "Pre-Term": [we_preterm_butter, we_preterm_butter_epochs, we_preterm_butter_overlap, we_preterm_fir, we_preterm_fir_epochs, we_preterm_fir_overlap],
}
df = pd.DataFrame(data)

# Create table figure
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")

# Display table
plt.show()

# Print results
print(df)
