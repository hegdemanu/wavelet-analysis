import numpy as np
from scipy.special import factorial

def estimate_noise_std(coefficients, scales):
    """Estimate noise standard deviation from wavelet coefficients."""
    low_activity_scales = scales[:5]
    noise_region = coefficients[:5, :]
    noise_std = np.std(noise_region)
    return noise_std, noise_region

def morse_wavelet(t, beta, gamma):
    """Generate Morse wavelet with specified parameters."""
    return (2**(beta/gamma)) * (gamma**(beta)) * t**(beta - 1) * \
           np.exp(-gamma * t) * np.exp(-0.5 * t**2) / factorial(beta - 1)