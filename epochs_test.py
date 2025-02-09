import numpy as np
import os
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# Define folder paths
ehg_folder_path = "EHGs"
preterm_folder = os.path.join(ehg_folder_path, "preterm")
term_folder = os.path.join(ehg_folder_path, "term")

# List files and select the first 10
preterm_files = sorted(os.listdir(preterm_folder))[:10]  # Select 10 preterm files
term_files = sorted(os.listdir(term_folder))[:10]  # Select 10 term files

# Function to read the 4th column (S1 channel) from a file
def read_s1_column(file_path):
    try:
        data = np.loadtxt(file_path, usecols=3)  # 4th column (S1 channel)
        if len(data) > 360:  # Ensure enough data points before slicing
            data = data[181:-181]  # Ignore the first and last 180 values
        else:
            print(f"Skipping {file_path} due to insufficient data after trimming.")
            return np.array([])
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.array([])

# Function to compute Sample Entropy
def sample_entropy(signal, m=3, r=0.15):
    N = len(signal)
    if N == 0:
        return np.nan  # Avoid errors with empty signals
    
    r *= np.std(signal)  # Set r in relation to standard deviation

    def _phi(m):
        # Create embedding vectors
        X = np.array([signal[i:i + m] for i in range(N - m + 1)])
        # Compute pairwise Chebyshev distances
        distances = pdist(X, metric='chebyshev')
        # Count the number of distances <= r
        count = np.sum(distances <= r)
        # Normalize by the number of possible pairs
        return count / (N - m + 1)

    phi_m = _phi(m)  # phi(m)
    phi_m1 = _phi(m + 1)  # phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return np.nan  # Avoid log(0)

    return -np.log(phi_m1 / phi_m)

# Function for epoching EEG signals
def segment_epochs(signal, fs, epoch_duration=2, overlap=0):
    epoch_length = int(epoch_duration * fs)
    overlap_length = int(overlap * fs)
    num_epochs = (len(signal) - epoch_length) // overlap_length + 1
    
    if num_epochs <= 0:
        return np.array([])  # Return empty if signal is too short
    
    epochs = np.array([signal[i * overlap_length:i * overlap_length + epoch_length] for i in range(num_epochs)])
    return epochs

fs = 256  # Example sampling frequency

preterm_entropy = []
preterm_filename = []
term_entropy = []
term_filename = []

print("\nSample Entropy for Preterm Files:")
for file in preterm_files:
    file_path = os.path.join(preterm_folder, file)
    signal = read_s1_column(file_path)
    if len(signal) == 0:
        print(f"Skipping {file} due to empty signal.")
        continue
    
    # Epoching with 2s duration and 1s overlap
    epochs = segment_epochs(signal, fs, epoch_duration=2, overlap=1)
    if epochs.size == 0:
        print(f"Skipping {file} due to insufficient data for epoching.")
        continue
    
    entropy_values = [sample_entropy(epoch) for epoch in epochs]
    avg_entropy = np.nanmean(entropy_values)  # Compute mean entropy per file
    
    print(f"Preterm - {file}: sE = {avg_entropy:.4f}")
    preterm_filename.append(file)
    preterm_entropy.append(avg_entropy)

print("\nSample Entropy for Term Files:")
for file in term_files:
    file_path = os.path.join(term_folder, file)
    signal = read_s1_column(file_path)
    if len(signal) == 0:
        print(f"Skipping {file} due to empty signal.")
        continue
    
    # Epoching with 2s duration and 1s overlap
    epochs = segment_epochs(signal, fs, epoch_duration=2, overlap=1)
    if epochs.size == 0:
        print(f"Skipping {file} due to insufficient data for epoching.")
        continue
    
    entropy_values = [sample_entropy(epoch) for epoch in epochs]
    avg_entropy = np.nanmean(entropy_values)  # Compute mean entropy per file
    
    print(f"Term - {file}: sE = {avg_entropy:.4f}")
    term_filename.append(file)
    term_entropy.append(avg_entropy)

# Plotting Sample Entropy for Preterm
plt.figure(figsize=(10, 5))
plt.bar(preterm_filename, preterm_entropy, color='blue')
plt.xlabel('Preterm Files')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy for Preterm Files')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plotting Sample Entropy for Term
plt.figure(figsize=(10, 5))
plt.bar(term_filename, term_entropy, color='green')
plt.xlabel('Term Files')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy for Term Files')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
