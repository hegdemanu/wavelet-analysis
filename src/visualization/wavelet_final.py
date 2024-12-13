import pandas as pd
import pywt
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read Excel data into a DataFrame
file_path = 'nifty50.xlsx'
df = pd.read_excel(file_path, sheet_name=0)  # Assuming data is in the first sheet

# Step 2: Specify the wavelet function
wavelet = 'morl'  # Use the Morlet wavelet

# Step 3: Extract data for the entire dataset
last_price_values = df['LastPrice'].values
dates = pd.to_datetime(df['Date'])

# Determine the length of the dataset
data_length = len(last_price_values)

# Determine the range of scales based on the length of the dataset
max_scale = min(128, data_length)  # Limit the maximum scale to avoid exceeding the length of the dataset
scales = np.arange(1, max_scale)

# Perform wavelet analysis
coeffs, freqs = pywt.cwt(last_price_values, scales=scales, wavelet=wavelet)

# Create common x-values for plotting
x_values = np.arange(data_length)

# Plot the wavelet analysis results
plt.figure(figsize=(12, 10))

# Plot the original LastPrice with dates on the x-axis
plt.subplot(3, 1, 1)
plt.plot(dates, last_price_values, color='blue')
plt.title('LastPrice - Nifty 50')
plt.xlabel('Date')
plt.ylabel('LastPrice')

# Plot the wavelet coefficients
plt.subplot(3, 1, 2)
plt.imshow(np.abs(coeffs), aspect='auto', extent=[dates[0], dates[-1], freqs[0], freqs[-1]], cmap='viridis', interpolation='bilinear')
plt.title('Wavelet Analysis - LastPrice')
plt.xlabel('Date')
plt.ylabel('Frequency (Hz)')

# Plot the scaleogram
plt.subplot(3, 1, 3)
plt.imshow(abs(coeffs)**2, aspect='auto', extent=[dates[0], dates[-1], scales[0], scales[-1]], cmap='jet', origin='lower')
plt.title('Scaleogram - LastPrice')
plt.xlabel('Date')
plt.ylabel('Scale')
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.show()
