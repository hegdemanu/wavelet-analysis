import pandas as pd
import pywt
import matplotlib.pyplot as plt

# Step 1: Read Excel data into a DataFrame
file_path = 'sp_500.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Step 2: Extract data for the first year (rows 2 to 12)
first_year_data = df.iloc[1:13]

# Step 3: Perform wavelet analysis
wavelet = 'db1'  # You can choose a different wavelet function if needed
coeffs = pywt.wavedec(first_year_data['value'], wavelet)

# Step 4: Plot the original and wavelet transformed signals
plt.figure(figsize=(10, 6))

# Plot the original signal
plt.subplot(2, 1, 1)
plt.plot(first_year_data['value'])
plt.title('Original Signal')

# Plot the wavelet coefficients
plt.subplot(2, 1, 2)
for i in range(len(coeffs)):
    plt.plot(coeffs[i], label=f'Level {i+1}')

plt.title('Wavelet Coefficients')
plt.legend()
plt.tight_layout()
plt.show()



