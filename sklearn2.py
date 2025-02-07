import os
import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd

# Helper function to extract signals from a directory
class EHGSignalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, directory, column_index=4):  # 4η στήλη -> index 3
        self.directory = directory
        self.column_index = column_index

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        signals = []
        labels = []
        for subdir in ['term', 'preterm']:
            subdir_path = os.path.join(self.directory, subdir)
            if os.path.exists(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith('.txt'):
                        try:
                            data = np.loadtxt(os.path.join(subdir_path, file_name))
                            signals.append(data[:, self.column_index])
                            labels.append(subdir)
                        except Exception as e:
                            print(f"Error loading {file_name}: {e}")
        return signals, labels

# Wavelet entropy function
def wavelet_entropy(signal, wavelet='db4'):
    coeffs = pywt.wavedec(signal, wavelet, level=4)
    energy = np.array([np.sum(np.square(c)) for c in coeffs])
    total_energy = np.sum(energy)
    if total_energy == 0:
        return 0  # Αποφυγή division by zero
    entropy = -np.sum((energy / total_energy) * np.log2(energy / total_energy + 1e-10))  # Αποφυγή log(0)
    return entropy

# Transformer for wavelet entropy
class WaveletEntropyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([wavelet_entropy(signal) for signal in X])

# Create the processing pipeline
def create_pipeline(directory):
    return Pipeline([
        ('extractor', EHGSignalExtractor(directory=directory)),  # Μόνο εξαγωγή σημάτων
        ('entropy', WaveletEntropyTransformer())  # Υπολογισμός Wavelet Entropy
    ])

def main(input_directory):
    # Run the pipeline
    pipeline = create_pipeline(input_directory)
    signals, labels = pipeline.named_steps['extractor'].transform(None)
    entropies = pipeline.named_steps['entropy'].transform(signals)
    
    print(f"Computed entropies for {len(entropies)} signals.")
    
    # Save to file
    df = pd.DataFrame({'Entropy': entropies, 'Category': labels})
    df.to_csv('wavelet_entropies.csv', index=False)
    print("Wavelet entropies saved to 'wavelet_entropies.csv' with category labels.")
    
    # Check entropy values
    print("\nEntropy Value Check:")
    print(f"Min Entropy: {df['Entropy'].min():.4f}")
    print(f"Max Entropy: {df['Entropy'].max():.4f}")
    print(f"Mean Entropy (Term): {df[df['Category'] == 'term']['Entropy'].mean():.4f}")
    print(f"Mean Entropy (Preterm): {df[df['Category'] == 'preterm']['Entropy'].mean():.4f}")
    
    # Bar plot for each sample
    plt.figure(figsize=(15, 6))
    
    # Χρώματα ανά κατηγορία
    colors = ['blue' if label == 'term' else 'red' for label in labels]
    
    # Δημιουργία bar plot
    plt.bar(range(len(df)), df['Entropy'], color=colors, alpha=0.7)
    
    # Διαμόρφωση plot
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Wavelet Entropy", fontsize=12)
    plt.title("Wavelet Entropy per EHG Signal (Term vs Preterm)", fontsize=14)
    plt.xticks(range(len(df)), rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Term'),
        Patch(facecolor='red', alpha=0.7, label='Preterm')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    input_directory = 'EHGs'  # Adjust path as necessary
    main(input_directory)