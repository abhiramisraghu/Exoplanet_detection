import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# Define the log likelihood function
def log_likelihood(theta, t, y, yerr):
    P1, K1, e1, w1, P2, K2, e2, w2, P3, K3, e3, w3, v0 = theta
    # Calculate the model radial velocity curve
    model = v0 + K1 * np.sin(2*np.pi*t/P1 + w1) + K2 * np.sin(2*np.pi*t/P2 + w2) + K3 * np.sin(2*np.pi*t/P3 + w3)
    # Calculate the log likelihood using Gaussian errors
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

# Define the log prior function
def log_prior(theta):
    P1, K1, e1, w1, P2, K2, e2, w2, P3, K3, e3, w3, v0 = theta
    # Enforce some prior constraints on the parameters
    if (1 < P1 < 1000 and 0 < K1 < 100 and 0 <= e1 < 1 and 0 <= w1 < 2*np.pi and
        1 < P2 < 1000 and 0 < K2 < 100 and 0 <= e2 < 1 and 0 <= w2 < 2*np.pi and
        1 < P3 < 3000 and 0 < K3 < 100 and 0 <= e3 < 1 and 0 <= w3 < 2*np.pi and
        -100 < v0 < 100):
        return 0.0
    return -np.inf

# Define the log posterior function
def log_posterior(theta, t, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y, yerr)

data=np.loadtxt("HD 37124.txt")
t=data[:,1]
y=data[:,2]
yerr=data[:,3].T

# Set up the MCMC sampler
ndim, nwalkers = 13, 100
pos = np.array([154, 28, 0.16, np.pi/2, 885, 15, 0.4, 0, 2295, 12.2, 0.2, 0, 2.21]) + 1e-4*np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, y, yerr))

# Burn in the sampler
pos, _, _ = sampler.run_mcmc(pos, 1000)
sampler.reset()

# Run the MCMC sampler
sampler.run_mcmc(pos, 5000, progress=True)

# Plot the results
samples = sampler.get_chain(discard=1000, flat=True)
labels = ['P1', 'K1', 'e1', 'w1', 'P2', 'K2', 'e2', 'w2', 'P3', 'K3', 'e3', 'w3', 'v0']

# Compute the median values and the uncertainties
med = np.median(samples, axis=0)
lower = np.percentile(samples, 16, axis=0)
upper = np.percentile(samples, 84, axis=0)

P1, K1, e1,w1, P2, K2, e2, w2, P3,K3,e3,w3,v0= np.median(samples, axis=0)
print(f"P1 = {P1:.2f} ± {np.std(samples[:, 0]):.2f} days")
print(f"K1 = {K1:.2f} ± {np.std(samples[:, 1]):.2f} m/s")
print(f"e1 = {e1:.2f} ± {np.std(samples[:, 2]):.2f}")
print(f"w1 = {w1:.2f} ± {np.std(samples[:, 3]):.2f}")
print(f"P2 = {P2:.2f} ± {np.std(samples[:, 4]):.2f} days")
print(f"K2 = {K2:.2f} ± {np.std(samples[:, 5]):.2f} m/s")
print(f"e2 = {e2:.2f} ± {np.std(samples[:, 6]):.2f}")
print(f"w2 = {w2:.2f} ± {np.std(samples[:, 7]):.2f}")
print(f"P3 = {P3:.2f} ± {np.std(samples[:, 8]):.2f} days")
print(f"K3 = {K3:.2f} ± {np.std(samples[:, 9]):.2f} m/s")
print(f"e3 = {e3:.2f} ± {np.std(samples[:, 10]):.2f}")
print(f"w3 = {w3:.2f} ± {np.std(samples[:, 11]):.2f}")
print(f"v0= {v0:.2f} ± {np.std(samples[:, 12]):.2f} ")

t_fit = np.linspace(min(t), max(t), 1000)
params_fit = np.median(samples, axis=0)
model = v0 + K1 * np.sin(2*np.pi*t_fit/P1 + w1) + K2 * np.sin(2*np.pi*t_fit/P2 + w2) + K3 * np.sin(2*np.pi*t_fit/P3 + w3)


# Plot the observed data with error bars
plt.errorbar(t, y, yerr=yerr, fmt='o', color='k', ms=4)

# Plot the best-fit model
plt.plot(t_fit,model, color='r', lw=2)

# Label the axes
plt.xlabel('JD-2450000 (days)')
plt.ylabel('Radial velocity (m/s)')

# Show the plot
plt.show()


fig = corner.corner(samples, labels=labels,label_kwargs={"fontsize":25}, quantiles=[0.95, 0.98, 0.99], show_titles=True, title_kwargs={"fontsize": 20})
plt.show()
