import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define a function to model the radial velocity data for a single planet
def single_planet_model(t, K, P, e, w, v0):
    M = 2 * np.pi * t / P + w
    E = M + e * np.sin(M)
    f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(E/2))
    v = K * (np.cos(f + w) + e * np.cos(w)) + v0
    return v

# Define a function to model the radial velocity data for a double planet
def double_planet_model(t, K1, P1, e1, w1, K2, P2, e2, w2, v0):
    v1 = single_planet_model(t, K1, P1, e1, w1, 0)
    v2 = single_planet_model(t, K2, P2, e2, w2, 0)
    v = v1 + v2 + v0
    return v
# Define a function to model the radial velocity data for a trible planet
def trible_planet_model(t, K1, P1, e1, w1, K2, P2, e2, w2,K3,P3,e3,w3, v0):
    v1 = single_planet_model(t, K1, P1, e1, w1, 0)
    v2 = single_planet_model(t, K2, P2, e2, w2, 0)
    v3 = single_planet_model(t, K3, P3, e3, w3, 0)
    v = v1 + v2 +v3 +v0
    return v
# Load the radial velocity data
data = np.loadtxt('HD 37124.txt')
t = data[:, 1] # time
v= data[:, 2] # radial velocity
e = data[:, 3] # uncertainty in radial velocity


# Fit the single planet model to the data using curve_fit
p0_single = [28, 153, 0.1, np.pi/2, 2.21]
popt_single, pcov_single = curve_fit(single_planet_model, t, v, p0_single, e)

# Fit the double planet model to the data using curve_fit
p0_double = [28, 153, 0.1, np.pi/2, 15, 885, 0.1, np.pi/2, 2.21]
popt_double, pcov_double = curve_fit(double_planet_model, t, v, p0_double, e)

# Fit the double planet model to the data using curve_fit
p0_trible = [28, 153, 0.1, np.pi/2, 15, 885, 0.1, np.pi/2,12,2295,0.2,np.pi/2, 2.21]
popt_trible, pcov_trible= curve_fit(trible_planet_model, t, v, p0_trible, e)

# Calculate the reduced chi-square values for both models
residuals_single = v - single_planet_model(t, *popt_single)
chisq_single = np.sum(residuals_single**2 / e**2)
dof_single = len(v) - len(popt_single)
redchisq_single = chisq_single / dof_single

residuals_double = v - double_planet_model(t, *popt_double)
chisq_double = np.sum(residuals_double**2 / e**2)
dof_double = len(v) - len(popt_double)
redchisq_double = chisq_double / dof_double

residuals_trible = v - trible_planet_model(t, *popt_trible)
chisq_trible = np.sum(residuals_trible**2 / e**2)
dof_trible = len(v) - len(popt_trible)
redchisq_trible = chisq_trible / dof_trible


# Print the reduced chi-square values
print('Reduced chi-square for single planet model: {:.2f}'.format(redchisq_single))
print('Reduced chi-square for double planet model: {:.2f}'.format(redchisq_double))
print('Reduced chi-square for triple planet model: {:.2f}'.format(redchisq_trible))

