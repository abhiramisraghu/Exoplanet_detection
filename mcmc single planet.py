import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import scipy.optimize as op
# Define the model function for radial velocity data
def model(params, t):
    # Extract parameters
    K, P, e, omega ,gamma= params
    # Compute mean anomaly
    M = 2 * np.pi * t / P
    # Compute eccentric anomaly using Newton-Raphson iteration
    E = M
    for i in range(10):
        E = M + e * np.sin(E)
    # Compute true anomaly
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    # Compute radial velocity
    v = K * (np.cos(nu + omega) + e * np.cos(omega))
    # Add systemic velocity
    v += gamma
    return v
# Define the log-likelihood function
def log_likelihood(params, t, y, yerr):
    # Compute model predictions
    model_pred = model(params, t)
    # Compute log-likelihood using Gaussian noise assumption
    return -0.5 * np.sum(((y - model_pred) / yerr)**2 + np.log(2 * np.pi * yerr**2))

# Define the log-prior function
def log_prior(params):
    # Extract parameters
    K, P, e, omega,gamma= params
    # Check parameter bounds
    if 0 < K < 100 and 0 < P < 1500 and 0 < e < 1 and -np.pi < omega < np.pi and -100 < gamma <100 :
        return 0
    else:
        return -np.inf
    
    
# Define the log-posterior function
def log_posterior(params, t, y, yerr):
    return log_prior(params) + log_likelihood(params, t, y, yerr)    


data=np.loadtxt("HD 37124.txt")
t=data[:,1]
y=data[:,2]
yerr=data[:,3].T

# Plot data
plt.errorbar(t, y, yerr=yerr, fmt='.k')
plt.xlabel('Time (days)')
plt.ylabel('Radial velocity (m/s)')
plt.show()


# Define the initial parameter values and number of walkers
nwalkers = 100
K0,P0,e0,omega0,gamma0=28, 153, 0.1,np.pi/2,2.21
params_init =np.array([K0,P0,e0,omega0,gamma0])
ndim = len(params_init)
bounds = ((0, 100), (0, 1500),(0,1), (-np.pi, np.pi),(-10,10))

# Optimize initial parameter values
nll = lambda *args: -log_likelihood(*args)
result = op.minimize(nll, params_init, args=(t, y, yerr), bounds=bounds)
theta_ml = result.x

# Define the MCMC sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, y, yerr))
pos = [theta_ml + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
sampler.run_mcmc(pos, 10000, progress=True)

# Extract the chain and discard the burn-in
chain = sampler.chain[:, 1000:, :].reshape((-1, ndim))

# Plot the posterior distributions
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ['K', 'P', 'e', 'omega','gamma']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], 'k', alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('step')
plt.show()

# Compute the median parameter values and uncertainties
params_med = np.median(chain, axis=0)
params_std = np.std(chain, axis=0)
print(f"K = {params_med[0]:.2f} +/- {params_std[0]:.2f}")
print(f"P = {params_med[1]:.2f}+/- {params_std[1]:.2f}")
print(f"e = {params_med[2]:.2f} +/- {params_std[2]:.2f}")
print(f"omega = {params_med[3]:.2f} +/- {params_std[3]:.2f}")
print(f"gamma = {params_med[4]:.2f} +/- {params_std[4]:.2f}")

# Plot the histograms of the parameters
fig, axes = plt.subplots(ndim, figsize=(10, 20))
samples = sampler.get_chain()
labels = ['K (m/s)', 'P (days)', 'e', 'omega (rad)', 'gamma']
for i in range(ndim):
    ax = axes[i]
    ax.hist(chain[:, i], bins=30, density=True, histtype='step', color='k')
    ax.set_xlabel(labels[i])
    ax.set_ylabel('Density')
plt.show()


t_plot = np.linspace(200, 2500, 1000)
y_plot = model(params_med, t_plot)
plt.errorbar(t, y, yerr=yerr, fmt='o', color='k', ms=5, capsize=0, zorder=10)
plt.plot(t_plot, y_plot, '-', color='r', lw=2, zorder=5)

plt.xlabel('JD-2450000 (days)')
plt.ylabel('Radial velocity (m/s)')
plt.show()

# Extract samples from the chain
samples = sampler.get_chain(discard=100, flat=True)

# Define parameter labels and ranges
labels = ['K', 'P', 'e', 'omega', 'gamma']
ranges = [(np.percentile(samples[:, i], 1), np.percentile(samples[:, i], 99)) for i in range(ndim)]

# Plot the 2D parameter distribution with confidence intervals
fig = corner.corner(samples, labels=labels, range=ranges, quantiles=[0.95, 0.98,0.99], fill_contours=True, show_titles=True)
plt.show()
