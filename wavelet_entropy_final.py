import numpy as np
import os
import pywt
import pandas as pd
import matplotlib.pyplot as plt

def read_s1_column(file_path):
    """Reads the 4th column (index 4) from a text file."""
    data = np.loadtxt(file_path, usecols=4)
    return data

def wavelet_entropy(signal, wavelet='db4', level=4):
    """Computes Wavelet Entropy (WE) for a given signal."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energy = np.array([np.sum(np.square(c)) for c in coeffs])
    prob = energy / np.sum(energy)  # Normalize to get probability distribution
    entropy = -np.sum(prob * np.log2(prob + 1e-10))  # Compute entropy
    return entropy

ehg_folder_path = "EHGs"  
preterm_folder = os.path.join(ehg_folder_path, "preterm") 
term_folder = os.path.join(ehg_folder_path, "term")  

preterm_files = sorted(os.listdir(preterm_folder))[:10]  
term_files = sorted(os.listdir(term_folder))[:10]  

results = []
preterm_data = []
term_data = []

# Process Preterm Files
for file_name in preterm_files:
    file_path = os.path.join(preterm_folder, file_name)
    signal = read_s1_column(file_path)
    entropy = wavelet_entropy(signal)
    results.append([file_name, "Preterm", entropy])
    preterm_data.append((file_name, entropy))
    print(f"Preterm - {file_name}: WE = {entropy:.4f}")

# Process Term Files
for file_name in term_files:
    file_path = os.path.join(term_folder, file_name)
    signal = read_s1_column(file_path)
    entropy = wavelet_entropy(signal)
    results.append([file_name, "Term", entropy])
    term_data.append((file_name, entropy))
    print(f"Term - {file_name}: WE = {entropy:.4f}")

# Save results to CSV
df = pd.DataFrame(results, columns=["Filename", "Category", "Wavelet Entropy"])
csv_filename = "wavelet_entropy_results.csv"
df.to_csv(csv_filename, index=False)

print(f"\nWavelet Entropy values saved to {csv_filename}")

# Create a figure with two side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot Preterm Signals
preterm_filenames, preterm_entropies = zip(*preterm_data) if preterm_data else ([], [])
axes[0].bar(preterm_filenames, preterm_entropies, color='red')
axes[0].set_title("Wavelet Entropy for Preterm Signals")
axes[0].set_xlabel("File Name")
axes[0].set_ylabel("Wavelet Entropy")
axes[0].tick_params(axis='x', rotation=45)

# Plot Term Signals
term_filenames, term_entropies = zip(*term_data) if term_data else ([], [])
axes[1].bar(term_filenames, term_entropies, color='blue')
axes[1].set_title("Wavelet Entropy for Term Signals")
axes[1].set_xlabel("File Name")
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout for better visibility
plt.tight_layout()
plt.show()
