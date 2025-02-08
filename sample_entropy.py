import numpy as np
import os
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist

ehg_folder_path = "EHGs"
preterm_folder = os.path.join(ehg_folder_path, "preterm")
term_folder = os.path.join(ehg_folder_path, "term")

# Λίστα αρχείων
preterm_files = sorted(os.listdir(preterm_folder))[:10]  # Επιλογή 10 αρχείων
term_files = sorted(os.listdir(term_folder))[:10]  # Επιλογή 10 αρχείων
# Define paths for preterm and term files

# Function to read the 4th column (channel S1) from a file
def read_s1_column(file_path):
    data = np.loadtxt(file_path, usecols=4)  # 4th column (S1 channel)
    return data

# Function to compute Sample Entropy
def sample_entropy(signal, m=3, r=0.15):
    N = len(signal)
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
sampen_preterm = []
sampen_term = []

for file in preterm_files:
    signal = read_s1_column(os.path.join(preterm_folder, file))
    # signal = signal[::10]  # Downsampling (κάθε 10ο δείγμα)
    sampen_preterm.append(sample_entropy(signal))

for file in term_files:
    signal = read_s1_column(os.path.join(term_folder, file))
    # signal = signal[::10]  # Downsampling (κάθε 10ο δείγμα)
    sampen_term.append(sample_entropy(signal))

# Perform t-test
t_stat, p_value = ttest_ind(sampen_preterm, sampen_term, nan_policy='omit')

# Display results
sampen_preterm, sampen_term, p_value
print("Sample Entropy για Preterm:", sampen_preterm)
print("Sample Entropy για Term:", sampen_term)
print(f"t-test p-value: {p_value:.5f}")
