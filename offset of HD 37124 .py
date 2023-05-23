import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file
data = np.loadtxt("HD 37124.txt")
x = data[:, 1]  # X-values
y = data[:, 2]  # Y-values
ye = data[:, 3].T  # Error values

# Calculate the average of the Y-values
ybar = sum(y) / len(y)
print("Average of Y-values:", ybar)

# Plot a horizontal line at the average value
plt.plot([300, x[len(x)-1]], [ybar, ybar], 'g', label='Average')

# Plot the data with error bars
plt.errorbar(x, y, yerr=ye, fmt="o", capsize=3, ecolor='r', label='Data with error bars')

# Set the labels and title for the plot
plt.xlabel('JD-2450000 (days)', fontsize=12)
plt.ylabel('Radial velocity (m/s)', fontsize=12)
plt.title('Radial velocity offset of HD 37124', fontsize=15)

# Display the legend
plt.legend()

# Show the plot
plt.show()

