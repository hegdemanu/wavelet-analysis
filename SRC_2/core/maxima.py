import numpy as np

def find_maxima(coefficients, scales, noise_std=None, percentile_threshold=None):
    """
    Detect maxima points in wavelet coefficients.
    
    Args:
        coefficients (np.ndarray): Wavelet coefficients
        scales (np.ndarray): Scales used in wavelet transform
        noise_std (float, optional): Standard deviation of noise
        percentile_threshold (float, optional): Percentile threshold for detection
    
    Returns:
        tuple: Indices of maxima points
    """
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