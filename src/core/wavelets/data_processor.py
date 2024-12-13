import pandas as pd
import pywt
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read Excel data into a DataFrame
file_path = 'sp_500.xlsx'
df = pd.read_excel(file_path)

# Step 2: Specify the wavelet function
wavelet = 'morl'

# Step 3: Extract data 
pe_ratio_values = df['value'].values  # Assuming 'value' is your column name
years = pd.to_datetime(df['Date']).dt.year.values

# Perform wavelet analysis (corrected scales)
scales = np.arange(1, 128)
coeffs, freqs = pywt.cwt(pe_ratio_values, scales, wavelet)

# Plot the results
plt.figure(figsize=(12, 10))

# Plot the original PE ratio
plt.subplot(3, 1, 1)
plt.plot(pe_ratio_values, color='blue')  # No need for x_values here
plt.title('Value - S&P 500')
plt.xlabel('Year')
plt.ylabel('Value')
plt.xticks(np.arange(0, len(years), 50), years[::50], rotation=45, ha='right') 

# Plot the wavelet coefficients
plt.subplot(3, 1, 2)
plt.imshow(abs(coeffs), aspect='auto', origin='lower', 
           extent=[0, len(pe_ratio_values), freqs[0], freqs[-1]], cmap='viridis') 
plt.title('Wavelet Analysis - Value')
plt.xlabel('Year')
plt.ylabel('Frequency (Hz)')
plt.xticks(np.arange(0, len(years), 50), years[::50], rotation=45, ha='right')
plt.colorbar(label='Magnitude')

# Plot the scaleogram
plt.subplot(3, 1, 3)
plt.imshow(abs(coeffs)**2, aspect='auto', origin='lower', 
           extent=[0, len(pe_ratio_values), freqs[0], freqs[-1]], cmap='viridis') 
plt.title('Scaleogram - Value')
plt.xlabel('Year')
plt.ylabel('Frequency (Hz)') # Consider changing the label if interpreting in terms of scale 
plt.yticks(np.linspace(freqs[0], freqs[-1], 10))  # Adjust number of ticks if needed
plt.xticks(np.arange(0, len(years), 50), years[::50], rotation=45, ha='right')
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.show()