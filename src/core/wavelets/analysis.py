import pywt
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from wavelet import morse
from scipy.special import factorial  # Use factorial for efficiency 


# --- Data Loading ---
data = pd.read_excel("sp_500.xlsx", parse_dates=["Date"]) 


def morse_wavelet(t, beta, gamma):
    return (2**(beta/gamma)) * (gamma**(beta)) * t**(beta - 1) * \
           np.exp(-gamma * t) * np.exp(-0.5 * t**2) / factorial(beta - 1)  

# --- Scale Selection --- 
short_scales = np.logspace(0, 1.5, num=10, base=2) 
medium_scales = np.logspace(1.5, 2.5, num=15, base=2) 
long_scales = np.logspace(2.5, 3.5, num=10, base=2)  
scales = np.concatenate((short_scales, medium_scales, long_scales)
coefficients, scales = pywt.cwt(data['value'], scales, wavelet)  


# --- Continuous Wavelet Transform ---
wavelet = pywt.Wavelet('morse', beta=3, gamma=2)  # Default beta, gamma
coefficients, scales = pywt.cwt(data['value'], scales, wavelet) 

# --- Maxima Detection---
def find_maxima(coefficients, scales, noise_std=None, percentile_threshold=None):
    # ... (Zero-crossing calculations) ... 

    maxima_points = []  
    if noise_std:
        threshold = noise_std * 3 
    elif percentile_threshold:
        threshold = np.percentile(np.abs(coefficients), percentile_threshold) 
        maxima_points = np.where(np.abs(coefficients) > threshold)

    return maxima_points

# --- Noise Estimation ---
def estimate_noise_std(coefficients, scales):
    low_activity_scales = scales[:5]  # Assuming noise dominates these scales 
    noise_region = coefficients[low_activity_scales, :]
    
