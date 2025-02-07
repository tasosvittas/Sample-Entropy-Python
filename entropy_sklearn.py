import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.spatial.distance import pdist

# Ορισμός φακέλων δεδομένων
ehg_folder_path = "EHGs"
preterm_folder = os.path.join(ehg_folder_path, "preterm")
term_folder = os.path.join(ehg_folder_path, "term")

# Λίστα αρχείων
preterm_files = sorted(os.listdir(preterm_folder))[:10]  # Επιλογή 10 αρχείων
term_files = sorted(os.listdir(term_folder))[:10]  # Επιλογή 10 αρχείων

# Συνάρτηση ανάγνωσης της 4ης στήλης
def read_s1_column(file_path):
    data = np.loadtxt(file_path, usecols=3)  # 4η στήλη (S1 κανάλι)
    return data

# Συνάρτηση Sample Entropy
def sample_entropy(signal, m=2, r=0.2):
    N = len(signal)
    r *= np.std(signal)  # Καθορισμός του r σε σχέση με την τυπική απόκλιση
    
    def _phi(m):
        X = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum(pdist(X, metric='chebyshev') <= r) / (N - m + 1)
        return C

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return np.nan  # Αποφυγή log(0)

    return -np.log(phi_m1 / phi_m)

# Ανάγνωση και υπολογισμός της Sample Entropy
sampen_preterm = []
sampen_term = []

for file in preterm_files:
    signal = read_s1_column(os.path.join(preterm_folder, file))
    signal = signal[::10]  # Downsampling (κάθε 10ο δείγμα)
    sampen_preterm.append(sample_entropy(signal))

for file in term_files:
    signal = read_s1_column(os.path.join(term_folder, file))
    signal = signal[::10]  # Downsampling (κάθε 10ο δείγμα)
    sampen_term.append(sample_entropy(signal))

# Προβολή των αποτελεσμάτων
print("Sample Entropy για Preterm:", sampen_preterm)
print("Sample Entropy για Term:", sampen_term)

# Δημιουργία bar plots
plt.figure(figsize=(10,5))

# Bar plot για Preterm
plt.subplot(1, 2, 1)
plt.bar(range(1, 11), sampen_preterm, color='blue')
plt.xlabel('Sample Index')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy - Preterm')

# Bar plot για Term
plt.subplot(1, 2, 2)
plt.bar(range(1, 11), sampen_term, color='green')
plt.xlabel('Sample Index')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy - Term')

plt.tight_layout()
plt.show()

# Στατιστικός έλεγχος t-test
t_stat, p_value = ttest_ind(sampen_preterm, sampen_term, nan_policy='omit')
print(f"t-test p-value: {p_value:.5f}")

if p_value < 0.05:
    print("Υπάρχει στατιστικά σημαντική διαφορά μεταξύ των δύο ομάδων.")
else:
    print("Δεν υπάρχει στατιστικά σημαντική διαφορά μεταξύ των δύο ομάδων.")
