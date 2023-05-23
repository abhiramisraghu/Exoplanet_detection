import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import leastsq
# Define the double Keplerian model
def keplerian_model(params, t):
    K1, K2, P1, P2, e, omega1,omega2, gamma = params
    M1 = 2*np.pi*t/P1 + omega1
    M2 = 2*np.pi*t/P2 + omega2
    E1 = kepler(M1, e)
    E2 = kepler(M2, e)
    V1 = K1*(np.cos(E1) - e)
    V2 = K2*(np.cos(E2) - e)
    return gamma + V1 + V2

# Define the Kepler solver
def kepler(M, e):
    E0 = M
    E1 = M + e*np.sin(E0)
    while (abs(E1 - E0) > 1e-8).all():
        E0 = E1
        E1 = M + e*np.sin(E0)
    return E1
# Define the residual function to minimize
def residuals(params, t, y):
    return y - keplerian_model(params, t)
# Load the radial velocity data

data=np.loadtxt("HD 37124.txt")
t=data[:,1]
y=data[:,2]
err=data[:,3].T
params0 = [28, 15, 154, 885, 0.1, np.pi/2,np.pi/2, 2.21]
# Fit the model to the data using least-squares
params_fit, flag = leastsq(residuals, params0, args=(t, y))
# Calculate the orbital parameters
K1, K2, P1, P2, e, omega1,omega2, gamma = params_fit

print('K1 =', K1, 'm/s')
print('K2 =', K2, 'm/s')
print('P1 =', P1, 'days')
print('P2 =', P2, 'days')
print('e =', e)
print('omega1 =', omega1, 'rad')
print('omega2 =', omega2, 'rad')
print('gamma =', gamma, 'm/s')

x=np.linspace(250,2500,2250)
# Plot the data and the model fit
plt.errorbar(t, y, yerr=err, fmt='o', label='Data')
plt.plot(x, keplerian_model(params_fit, x), label='Model')
plt.xlabel('JD-2450000 (days)',fontsize=12)
plt.ylabel('Radial velocity (m/s)',fontsize=12)
plt.show()