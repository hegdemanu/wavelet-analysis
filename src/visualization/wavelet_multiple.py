import pandas as pd
import pywt
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read Excel data into a DataFrame
file_path = 'sp_500.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Step 2: Specify the wavelet function
wavelet = 'db1'  # You can choose a different wavelet function if needed

# Step 3: Extract data for the entire dataset
window_size = 250
window_values = df['value'].tolist()
window_dates = pd.to_datetime(df['date']).dt.year.tolist()  # Extract years from dates

# Perform wavelet analysis
coeffs = pywt.wavedec(window_values, wavelet)

# Create common x-values for plotting
x_values = np.arange(len(window_values))

# Plot the scaleogram for the entire dataset
plt.figure(figsize=(12, 8))

# Plot the original signal with dates on the x-axis
plt.subplot(2, 1, 1)
plt.plot(x_values, window_values, color='blue')
plt.title('Original Signal - Entire Dataset')
plt.xlabel('Year')
plt.ylabel('Value')
plt.xticks(x_values[::50], window_dates[::50], rotation=45, ha='right')  # Show every 50th year for better visibility

# Plot the scaleogram with the 'default' scale
plt.subplot(2, 1, 2)
plt.specgram(window_values, NFFT=256, Fs=1, noverlap=128, cmap='viridis', vmin=-50, vmax=20)
plt.title('Scaleogram')
plt.xlabel('Year')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude (dB)')

plt.tight_layout()
plt.show()
