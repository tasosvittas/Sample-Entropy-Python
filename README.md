# EHG Signal Processing - Term vs Preterm Classification

## Project Overview

This project focuses on analyzing uterine electromyography (EMG) signals from the TPEHG DB dataset. The dataset contains 300 uterine EMG records from 300 pregnancies, categorized into term and preterm deliveries. The analysis involves calculating the **Sample Entropy** of the filtered signals, specifically focusing on the first channel (S1), which has been filtered using a 4th-order Butterworth filter with a passband from 0.3 Hz to 3 Hz.

For this analysis, **only 20 files** from the dataset are used:
- **10 files** from the **term deliveries** group.
- **10 files** from the **preterm deliveries** group.

The project includes two main Python scripts:
1. **`sample_entropy.py`**: Computes the sample entropy for preterm and term signals, both for the entire signal and for segmented epochs (with and without overlap).
2. **`signals_read.py`**: Reads and visualizes the S1 channel signals from the dataset.

## Dataset Description

The TPEHG DB dataset contains 300 uterine EMG records, divided into:
- **Term deliveries**: 262 records (gestation > 37 weeks).
- **Preterm deliveries**: 38 records (gestation ≤ 37 weeks).

Each record consists of three channels (S1, S2, S3) recorded from four electrodes placed around the navel. The signals were digitized at 20 samples per second with 16-bit resolution and filtered using Butterworth filters.

### Filtered Signals
The dataset provides signals filtered using three different Butterworth filters:
1. **0.08 Hz to 4 Hz**
2. **0.3 Hz to 3 Hz**
3. **0.3 Hz to 4 Hz**

In this project, we focus on the **S1 channel** filtered with the **0.3 Hz to 3 Hz** Butterworth filter.

### Important
- The first and last 180 seconds of the filtered signals should be ignored due to transient effects of the filters.
- The dataset includes clinical information such as pregnancy duration, maternal age, and other relevant details.

## Project Structure

### Files
- **`signals_read.py`**:
  - Reads the S1 channel from the dataset.
  - Plots the S1 signals for both preterm and term records.
    
- **`sample_entropy.py`**: 
  - Computes the sample entropy for the entire signal and for segmented epochs.
  - Segments the signal into non-overlapping and overlapping epochs.
  - Plots the sample entropy for preterm and term signals.

## Sample Entropy Calculation

Sample entropy is a measure of the complexity or irregularity of a signal. In this project, we calculate the sample entropy for:
1. The entire signal.
2. Non-overlapping epochs (2-second segments).
3. Overlapping epochs (2-second segments with 1-second overlap).

The results are visualized using bar plots for both preterm and term signals.

### Prerequisites
- Python 3.x
- Required Python libraries: `numpy`, `scipy`, `matplotlib`
