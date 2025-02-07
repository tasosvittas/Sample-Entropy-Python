import numpy as np
import os
import matplotlib.pyplot as plt

def read_s1_column(file_path):
    """Reads the 4th column (index 4) from a text file."""
    data = np.loadtxt(file_path, usecols=4)
    return data

ehg_folder_path = "EHGs"  
preterm_folder = os.path.join(ehg_folder_path, "preterm") 
term_folder = os.path.join(ehg_folder_path, "term")  

preterm_files = sorted(os.listdir(preterm_folder))[:10]  
term_files = sorted(os.listdir(term_folder))[:10]  

# Plot signals for preterm files
for file_name in preterm_files:
    file_path = os.path.join(preterm_folder, file_name)
    signal = read_s1_column(file_path)
    
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label=f"Preterm - {file_name}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"S1 Signal from {file_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot signals for term files
for file_name in term_files:
    file_path = os.path.join(term_folder, file_name)
    signal = read_s1_column(file_path)
    
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label=f"Term - {file_name}", color='orange')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"S1 Signal from {file_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
