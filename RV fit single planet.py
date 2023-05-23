import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import least_squares

# Load the data from the file
data = np.loadtxt("HD 37124.txt")
x = data[:, 1]  # X-values
y = data[:, 2]  # Y-values
ye = data[:, 3].T  # Error values

# Print the average of the Y-values
print("Average of Y-values:", sum(y) / len(y))

# Define the Keplerian model function
def keplerian_model(params, t, yerr):
    v, k, P, e, w, tp = params
    n = 2 * np.pi / P
    M = n * (t - tp)
    E = np.zeros_like(M)
    for i in range(len(M)):
        E[i] = kepler_eq_solver(M[i], e)
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return v + k * (np.cos(f + w) + e * np.cos(w))

# Define the Kepler equation solver
def kepler_eq_solver(M, e):
    E = M
    while True:
        delta = E - e * np.sin(E) - M
        if np.abs(delta) < 1e-8:
            break
        E = E - delta / (1 - e * np.cos(E))
    return E

# Define the residuals function for least squares optimization
def keplerian_residuals(params, t, y, ye):
    return (y - keplerian_model(params, t, ye)) / ye

# Generate time values for plotting
t = np.linspace(400, 2500, 1000)

# Initial parameter values for the optimization
params_init = [2.21, 28, 153, 0.1, np.pi / 2, 0]

# Perform least squares optimization to fit the model to the data
params_opt = least_squares(keplerian_residuals, params_init, args=(x, y, ye))

# Print the optimized parameters
print("Optimized parameters:", params_opt)

# Plot the data with error bars and the fitted model
plt.errorbar(x, y, yerr=ye, fmt="o", capsize=3, ecolor='r', label='Data with error bars')
plt.plot(t, keplerian_model(params_opt.x, t, ye), label='Fit')
plt.xlabel('JD-2450000 (days)', fontsize=12)
plt.ylabel('Radial velocity (m/s)', fontsize=12)
plt.title('Radial velocity data for HD 37124', fontsize=15)

plt.legend()
plt.show()

