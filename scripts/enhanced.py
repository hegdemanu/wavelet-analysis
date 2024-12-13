import pandas as pd
import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial  # Import factorial function

# Function for noise estimation
def estimate_noise_std(coefficients, scales):
    low_activity_scales = scales[:5]
    noise_region = coefficients[:5, :]
    noise_std = np.std(noise_region)
    return noise_std, noise_region

# Function for Morse wavelet
def morse_wavelet(t, beta, gamma):
    return (2**(beta/gamma)) * (gamma**(beta)) * t**(beta - 1) * \
           np.exp(-gamma * t) * np.exp(-0.5 * t**2) / factorial(beta - 1)

# Function for maxima detection
def find_maxima(coefficients, scales, noise_std=None, percentile_threshold=None):
    maxima_points = []

    if noise_std:
        threshold = noise_std * 3
        maxima_points = np.where(np.abs(coefficients) > threshold)

    elif percentile_threshold:
        threshold = np.percentile(np.abs(coefficients), percentile_threshold)
        maxima_points = np.where(np.abs(coefficients) > threshold)

    else:
        print("Error: Specify either a noise_std or percentile_threshold")

    return maxima_points

# Step 1: Read Excel data into a DataFrame
file_path = 'sp_500.xlsx'
df = pd.read_excel(file_path)

# Step 2: Specify the wavelet function with parameters
wavelet = 'cmor1.5-1.0'  # Construct a valid wavelet name

# Step 3: Extract data for the entire dataset
pe_ratio_values = df['value'].tolist()
years = pd.to_datetime(df['Date']).dt.year.tolist()

# Downsample the data to reduce computation time
downsample_factor = 2
pe_ratio_values_downsampled = pe_ratio_values[::downsample_factor]
years_downsampled = years[::downsample_factor]  # Downsample years for consistency


# Perform wavelet analysis with reduced scales
coeffs, freqs = pywt.cwt(pe_ratio_values_downsampled, scales=np.arange(1, 64, 2), wavelet=wavelet)

# Noise estimation
noise_std, noise_region = estimate_noise_std(coeffs, freqs)

# Maxima detection
maxima_points = find_maxima(coeffs, freqs, noise_std=noise_std)

# Adjust other plotting parameters accordingly
x_values_downsampled = np.arange(0, len(pe_ratio_values), downsample_factor)
years_downsampled = years[::downsample_factor]

# Plot the wavelet analysis results
plt.figure(figsize=(14, 12))

# Plot the original PE ratio with years on the x-axis
plt.subplot(4, 1, 1)
plt.plot(x_values_downsampled, pe_ratio_values_downsampled, color='blue')
plt.title('Value - S&P 500')
plt.xlabel('Year')
plt.ylabel('Value')
plt.xticks(x_values_downsampled[::50], years_downsampled[::50], rotation=45, ha='right')

# Plot the wavelet coefficients
plt.subplot(4, 1, 2)
plt.imshow(np.abs(coeffs), aspect='auto', extent=[0, len(pe_ratio_values_downsampled), freqs[-1], freqs[0]], cmap='viridis', interpolation='bilinear')
plt.title('Wavelet Analysis - Value')
plt.xlabel('Year')
plt.ylabel('Frequency (Hz)')

# Plot the magnitude of the estimated noise region
plt.subplot(4, 1, 3)
plt.imshow(np.abs(noise_region), aspect='auto', extent=[0, len(pe_ratio_values_downsampled), freqs[4], freqs[0]], cmap='viridis', interpolation='bilinear')
plt.title('Magnitude of Estimated Noise Region')
plt.xlabel('Year')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude')
plt.xticks(x_values_downsampled[::50], years_downsampled[::50], rotation=45, ha='right')

# Ridge Plot
plt.subplot(4, 1, 4)
ridge_levels = np.linspace(0, np.max(np.abs(coeffs)), 20)
plt.contourf(np.abs(coeffs), levels=ridge_levels, colors='black', extent=[0, len(pe_ratio_values_downsampled), freqs[-1], freqs[0]])
plt.title('Ridge Plot')
plt.xlabel('Year')
plt.ylabel('Frequency (Hz)')

# Overlay Wavelet Skeleton with Maxima
plt.plot(maxima_points[1], freqs[maxima_points[0]], 'ro', label='Maxima')

for i in range(len(maxima_points[0])):
    scale = freqs[maxima_points[0][i]]
    wavelet_shape = morse_wavelet(np.arange(-10, 11), beta=3, gamma=2)
    time_points = maxima_points[1][i] + np.arange(-10, 11)

    plt.plot(time_points, wavelet_shape, 'k--', alpha=0.5)
    plt.plot(maxima_points[1][i], scale, 'ko')

# Adjust layout
plt.tight_layout()
plt.show()
