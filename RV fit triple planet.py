import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# Define the double Keplerian model
def keplerian_model(params, t):
    K1, K2,K3, P1, P2,P3, e, omega1,omega2,omega3, gamma = params
    M1 = 2*np.pi*t/P1 + omega1
    M2 = 2*np.pi*t/P2 + omega2
    M3=2*np.pi*t/P3 + omega3
    E1 = kepler(M1, e)
    E2 = kepler(M2, e)
    E3 = kepler(M3, e)
    V1 = K1*(np.cos(E1) - e)
    V2 = K2*(np.cos(E2) - e)
    V3 = K3*(np.cos(E3) - e)
    return gamma + V1 + V2+V3

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

data=np.loadtxt("HD 37124.txt")
t=data[:,1]
y=data[:,2]
err=data[:,3].T


params0 = [28, 15,12.2, 154, 885,2295, 0.1, np.pi/2,np.pi/2,np.pi/2, 2.21]

# Fit the model to the data using least-squares
params_fit, flag = leastsq(residuals, params0, args=(t, y))
# Calculate the orbital parameters
K1, K2,K3, P1, P2,P3, e,omega1,omega2,omega3, gamma= params_fit

print('K1 =', K1, 'm/s')
print('K2 =', K2, 'm/s')
print('K3=', K3, 'm/s')
print('P1 =', P1, 'days')
print('P2 =', P2, 'days')
print('P3=', P3, 'days')
print('e=', e)
print('omega1 =', omega1, 'degrees')
print('omega2 =', omega2, 'degrees')
print('omega3 =', omega3, 'degrees')
print('gamma =', gamma, 'm/s')

x=np.linspace(250,2500,2250)

# Plot the data and the model fit
plt.errorbar(t, y, yerr=err, fmt='o', label='Data')
plt.plot(x, keplerian_model(params_fit, x), label='Model')
plt.xlabel('Time (days)')
plt.ylabel('Radial velocity (m/s)')
plt.legend()
plt.show()