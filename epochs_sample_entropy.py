import numpy as np
import os
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

ehg_folder_path = "EHGs"
preterm_folder = os.path.join(ehg_folder_path, "preterm")
term_folder = os.path.join(ehg_folder_path, "term")

preterm_files = sorted(os.listdir(preterm_folder))[:10]  #10 preterm files
term_files = sorted(os.listdir(term_folder))[:10]  #10 term files

def read_s1_column(file_path):
    data = np.loadtxt(file_path, usecols=3)  # 4th column (S1 channel)
    data = data[181:-181]  
    return data

def sample_entropy(signal, m=3, r=0.15):
    N = len(signal)
    if N == 0:
        return np.nan  
    
    r *= np.std(signal) 
    def _phi(m):
        X = np.array([signal[i:i + m] for i in range(N - m + 1)])
        distances = pdist(X, metric='chebyshev')
        count = np.sum(distances <= r)
        return count / (N - m + 1)
        
    phi_m = _phi(m)  #(m)
    phi_m1 = _phi(m + 1)  #(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return np.nan 

    return -np.log(phi_m1 / phi_m)

def segment_epochs(signal, fs, epoch_duration=2, overlap=0):
    epoch_length = int(epoch_duration * fs)
    overlap_length = int(overlap * fs) if overlap > 0 else epoch_length
    num_epochs = (len(signal) - epoch_length) // overlap_length + 1
    
    epochs = np.array([signal[i * overlap_length:i * overlap_length + epoch_length] for i in range(num_epochs)])
    return epochs

fs = 256  
preterm_entropy = []
term_entropy = []
preterm_non_overlap_entropy = []
term_non_overlap_entropy = []
preterm_overlap_entropy = []
term_overlap_entropy = []

print("\nSample Entropy, Non-Overlapping and Overlapping Epochs for Preterm and Term Files:")
for category, folder, files, entropy_list, non_overlap_list, overlap_list in [
    ("Preterm", preterm_folder, preterm_files, preterm_entropy, preterm_non_overlap_entropy, preterm_overlap_entropy),
    ("Term", term_folder, term_files, term_entropy, term_non_overlap_entropy, term_overlap_entropy)]:
    
    for file in files:
        file_path = os.path.join(folder, file)
        signal = read_s1_column(file_path)
        if len(signal) == 0:
            print(f"Skipping {file} due to empty signal.")
            continue
        
        #Sample Entropy
        entropy_value = sample_entropy(signal)
        entropy_list.append(entropy_value)
        
        # Epoching without overlap
        non_overlap_epochs = segment_epochs(signal, fs, epoch_duration=2, overlap=0)
        non_overlap_entropy = [sample_entropy(epoch) for epoch in non_overlap_epochs]
        avg_non_overlap_entropy = np.nanmean(non_overlap_entropy) if len(non_overlap_entropy) > 0 else np.nan
        non_overlap_list.append(avg_non_overlap_entropy)
        
        # Epoching with overlap
        overlap_epochs = segment_epochs(signal, fs, epoch_duration=2, overlap=1)
        overlap_entropy = [sample_entropy(epoch) for epoch in overlap_epochs]
        avg_overlap_entropy = np.nanmean(overlap_entropy) if len(overlap_entropy) > 0 else np.nan
        overlap_list.append(avg_overlap_entropy)
        
        print(f"{category} - {file}: sE = {entropy_value:.4f}, Non-Overlap sE = {avg_non_overlap_entropy:.4f}, Overlap sE = {avg_overlap_entropy:.4f}")
        
      

def compute_average_entropy(entropy_list):
    return np.nanmean(entropy_list) if len(entropy_list) > 0 else np.nan

avg_preterm_entropy = compute_average_entropy(preterm_entropy)
avg_term_entropy = compute_average_entropy(term_entropy)

avg_preterm_non_overlap_entropy = compute_average_entropy(preterm_non_overlap_entropy)
avg_term_non_overlap_entropy = compute_average_entropy(term_non_overlap_entropy)

avg_preterm_overlap_entropy = compute_average_entropy(preterm_overlap_entropy)
avg_term_overlap_entropy = compute_average_entropy(term_overlap_entropy)

print("\nAverage Entropy Values:")
print(f"Preterm - Sample Entropy: {avg_preterm_entropy:.4f}, Non-Overlap Epoch Entropy: {avg_preterm_non_overlap_entropy:.4f}, Overlap Epoch Entropy: {avg_preterm_overlap_entropy:.4f}")
print(f"Term - Sample Entropy: {avg_term_entropy:.4f}, Non-Overlap Epoch Entropy: {avg_term_non_overlap_entropy:.4f}, Overlap Epoch Entropy: {avg_term_overlap_entropy:.4f}")

#Sample Entropy
plt.figure(figsize=(10, 5))
plt.bar(preterm_files, preterm_entropy, color='blue', label='Preterm')
plt.bar(term_files, term_entropy, color='green', label='Term')
plt.xlabel('Files')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy for Preterm and Term Files')
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

#Non-Overlapping Epoch Entropy
plt.figure(figsize=(10, 5))
plt.bar(preterm_files, preterm_non_overlap_entropy, color='blue', label='Preterm')
plt.bar(term_files, term_non_overlap_entropy, color='green', label='Term')
plt.xlabel('Files')
plt.ylabel('Non-Overlapping Epoch Entropy')
plt.title('Non-Overlapping Epoch Entropy for Preterm and Term Files')
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

#Overlapping Epoch Entropy
plt.figure(figsize=(10, 5))
plt.bar(preterm_files, preterm_overlap_entropy, color='blue', label='Preterm')
plt.bar(term_files, term_overlap_entropy, color='green', label='Term')
plt.xlabel('Files')
plt.ylabel('Overlapping Epoch Entropy')
plt.title('Overlapping Epoch Entropy for Preterm and Term Files')
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()
