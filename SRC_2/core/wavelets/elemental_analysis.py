import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame  # Explicit import of DataFrame

# Load your data from Excel
excel_file = 'nifty50.xlsx'
data = pd.read_excel(excel_file, parse_dates=['Date'], sheet_name='Sheet1')
data.set_index('Date', inplace=True)

# Parameters for Analysis
column_to_analyze = 'LastPrice'
wavelet_type = 'db4'
levels = 5  # Control depth of wavelet decomposition 

# Filter out rows with empty values in the column to analyze
data_for_analysis = data.dropna(subset=[column_to_analyze])[column_to_analyze]

# Check if there's still data to analyze after filtering
if data_for_analysis.empty:
    print("No data available for analysis.")
else:
    # Discrete Wavelet Transform (DWT)
    coeffs = pywt.wavedec(data_for_analysis, wavelet_type, level=levels)

    # Scaleogram Visualization
    def plot_wavelet_scalogram(coeffs, wavelet, time_index):
        scales = np.arange(1, len(coeffs) + 1)
        
        # Reshape coefficient data if necessary
        if isinstance(coeffs[0], np.ndarray):
            coefficient_data = np.abs(np.array(coeffs).T)
        else:
            coefficient_data = np.abs(np.array([coeffs]).T)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(coefficient_data, cmap='viridis', aspect='auto', origin='lower',
                       extent=[time_index.min(), time_index.max(), 0, levels])

        ax.set_ylabel('Scale')
        ax.set_xlabel('Time (days)')
        ax.set_title(f'Scaleogram of {column_to_analyze} (Wavelet: {wavelet})')
        fig.colorbar(im, ax=ax, label='Coefficient Magnitude')
        plt.show()

    plot_wavelet_scalogram(coeffs, wavelet_type, data_for_analysis.index)
