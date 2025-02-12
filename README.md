# EHG Signal Processing - Term vs Preterm Classification

## Project Overview
This repository contains data and scripts related to the analysis of **Electrohysterogram (EHG) signals** from the **Term-Preterm EHG Database (TPEHG DB)**. The goal is to apply **entropy-based methods** to distinguish between term and preterm pregnancies using nonlinear signal processing techniques.

## Dataset
The dataset used in this project is sourced from the **PhysioNet TPEHG DB**: [PhysioNet TPEHG DB](https://www.physionet.org/physiobank/database/tpehgdb/).

### **Dataset Structure**
- **EHGs/** → Contains the extracted `.dat` files for analysis.
  - **preterm/** → Contains signals from pregnancies that ended in preterm birth.
  - **term/** → Contains signals from pregnancies that resulted in full-term birth.
- **2015 - Acharya - Entropy review.pdf** → Reference paper on entropy-based signal analysis.

### **Data Description**
- **300 uterine EMG records** (one per pregnancy).
- **262 records** correspond to **term deliveries** (>37 weeks).
- **38 records** correspond to **preterm deliveries** (≤37 weeks).
- Each record contains **three channels** recorded from four electrodes:
  - **S1 = E2 – E1** (First channel)
  - **S2 = E2 – E3** (Second channel)
  - **S3 = E4 – E3** (Third channel)
- Sampling rate: **20 Hz** with **16-bit resolution**.

### **Filtered Signals**
Each channel is available in filtered and unfiltered formats:
- **0.08 Hz – 4 Hz**
- **0.3 Hz – 3 Hz (Used in this analysis)**
- **0.3 Hz – 4 Hz**

## Methodology
### **1. Preprocessing**
- Extracting the **S1 channel** from each file.
- Filtering signals using **Butterworth band-pass filter (0.3 Hz – 3 Hz)**.
- Removing the first and last **180 seconds** due to transient filter effects.

### **2. Feature Extraction**
The following nonlinear features are computed:
- **Sample Entropy (sE):** Measures complexity in the EHG signals.
- **RMS (Root Mean Square):** Measures signal energy.
- **MAV (Mean Absolute Value):** Computes signal magnitude.
- **Variance:** Captures signal variability.
- **Zero-Crossing Rate (ZCR):** Counts the number of times the signal changes sign.

### **3. Visualization**
- **Bar plots of Sample Entropy** for term and preterm groups.
- **Comparative analysis** of entropy variations between groups.

## 🛠 Usage Instructions
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/ehg-analysis.git
   cd ehg-analysis
   ```
2. **Run the Python script for feature extraction and visualization**:
   ```sh
   python analyze_ehg.py
   ```

## 📌 Reference
- **PhysioNet TPEHG Database**: [Link](https://www.physionet.org/physiobank/database/tpehgdb/)
- **Acharya et al. (2015) - Entropy Review**: Published research on entropy-based analysis.

## 📝 License
This project is for academic and research purposes. Please cite the original TPEHG database and Acharya et al. (2015) when using this work.

---
💡 **For questions, feel free to reach out or open an issue!** 🚀

