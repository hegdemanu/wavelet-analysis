import matplotlib.pyplot as plt
import numpy as np

def plot_wavelet_analysis(x_values, data_values, years, coeffs, freqs, noise_region=None):
    """
    Create standard wavelet analysis plot with multiple subplots.
    
    Args:
        x_values (np.ndarray): X-axis values
        data_values (list/np.ndarray): Original data values
        years (list): Years for x-axis labeling
        coeffs (np.ndarray): Wavelet coefficients
        freqs (np.ndarray): Frequencies
        noise_region (np.ndarray, optional): Noise region coefficients
    """
    plt.figure(figsize=(15, 10))

    # Original Data
    plt.subplot(3, 1, 1)
    plt.plot(x_values, data_values, color='blue')
    plt.title('Original Data')
    plt.xlabel('Year')
    plt.xticks(x_values[::50], years[::50], rotation=45)

    # Wavelet Coefficients
    plt.subplot(3, 1, 2)
    plt.imshow(np.abs(coeffs), aspect='auto', 
               extent=[0, len(data_values), freqs[-1], freqs[0]], 
               cmap='viridis', interpolation='bilinear')
    plt.title('Wavelet Coefficients')
    plt.xlabel('Year')
    plt.ylabel('Frequency')

    # Noise Region or Scaleogram
    plt.subplot(3, 1, 3)
    if noise_region is not None:
        plt.imshow(np.abs(noise_region), aspect='auto', 
                   extent=[0, len(data_values), freqs[4], freqs[0]], 
                   cmap='viridis', interpolation='bilinear')
        plt.title('Estimated Noise Region')
    else:
        plt.imshow(abs(coeffs)**2, aspect='auto', 
                   extent=[0, len(data_values), freqs[0], freqs[-1]], 
                   cmap='jet', origin='lower')
        plt.title('Scaleogram')
    
    plt.xlabel('Year')
    plt.ylabel('Frequency/Scale')
    plt.colorbar(label='Magnitude')

    plt.tight_layout()
    plt.show()