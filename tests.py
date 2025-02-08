import numpy as np
import os
from scipy.spatial.distance import pdist


ehg_folder_path = "EHGs"
preterm_folder = os.path.join(ehg_folder_path, "preterm")
term_folder = os.path.join(ehg_folder_path, "term")

# List available files
preterm_files = sorted(os.listdir(preterm_folder))[:10]
term_files = sorted(os.listdir(term_folder))[:10]

# Function to read the 4th column (channel S1) from a file
def read_s1_column(file_path):
    try:
        data = np.loadtxt(file_path, usecols=3)  # 4th column (S1 channel)
        print(data)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.array([])  # Return empty array to avoid crashes

# Function to compute Sample Entropy
def sample_entropy(signal, m=3, r=0.15):
    N = len(signal)
    if N == 0:
        return np.nan  # Avoid errors with empty signals
    
    r *= np.std(signal)  # Set r in relation to standard deviation

    def _phi(m):
        X = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum(pdist(X, metric='chebyshev') <= r) / (N - m + 1)
        return C

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return np.nan  # Avoid log(0)

    return -np.log(phi_m1 / phi_m)

# Compute Sample Entropy for preterm and term groups
print("\nSample Entropy for Preterm Files:")
for file in preterm_files:
    file_path = os.path.join(preterm_folder, file)
    signal = read_s1_column(file_path)
    if len(signal) == 0:
        print(f"Skipping {file} due to empty signal.")
        continue
    entropy = sample_entropy(signal)
    print(f"Preterm - {file}: sE = {entropy:.4f}")

print("\nSample Entropy for Term Files:")
for file in term_files:
    file_path = os.path.join(term_folder, file)
    signal = read_s1_column(file_path)
    if len(signal) == 0:
        print(f"Skipping {file} due to empty signal.")
        continue
    entropy = sample_entropy(signal)
    print(f"Term - {file}: sE = {entropy:.4f}")
