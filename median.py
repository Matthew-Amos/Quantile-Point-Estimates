#%% Dependencies.
from turtle import width
import numpy as np
import matplotlib.pyplot as plt

#%% Data Generating Process.
np.random.seed(7979797)

"""
Data generating process. Draws `n` realizations from the distribution
function `dist`, which takes `n` as a sole parameter.
"""
def dgp(dist, n):
    return dist(n)

dist_normal = lambda n: np.random.normal(0, 5, size=n)
dist_gamma  = lambda n: np.random.gamma(1, size=n)

#%% Simulation Setup.

# Sample size of the parent data.
N = 10000

# Number of simulation iterations.
B = 1000

# Percentiles to calculate.
P = [0.20, 0.40, 0.50, 0.60, 0.80]

# Sample sizes to simulate.
sizes = np.concatenate((
    np.arange(10, 31),
    np.arange(30, 101, step=10),
    np.arange(100, 1001, step=100),
    np.arange(1000, 10001, step=1000)
))

"""
Permutative draw of size `size` from `X` for `B` iterations.
Each row represents a random realization. There is no replacement
across rows, but there is replacement across columns.
"""
def draw(X, size, B):
    return np.asarray([np.random.permutation(X)[0:size] for i in range(B)])

"""
Compute a mean and standard deviation of point estimates based on
varying sample sizes.
"""
def simulate(X, P, sizes, B):
    res = np.empty((len(sizes), len(P), 2), np.float64)

    for i in range(len(sizes)):
        xb = draw(X, sizes[i], B)
        pb = np.percentile(xb, P, axis=1)
        res[i, :, 0] = np.mean(pb, axis=1)
        res[i, :, 1] = np.std(pb, axis=1)

    return res

#%% Gamma Simulation.
Xgamma = dgp(dist_gamma, N)
Xgamma_p = np.percentile(Xgamma, P)
Xgamma_sim = simulate(Xgamma, P, sizes, B)

#%% Normal Simulation.
Xnormal= dgp(dist_normal, N)
Xnormal_p = np.percentile(Xnormal, P)
Xnormal_sim = simulate(Xnormal, P, sizes, B)

#%% Summarise.
def build_plot(sim, truth, idx, P, title, sizes):
    y_scaled = sim[:, idx, 0] / truth[idx]
    e_scaled = sim[:, idx, 1] / truth[idx]
    plt.errorbar(sizes, y_scaled, e_scaled, marker='^')
    plt.xlabel('Sample Size')
    plt.ylabel('Estimate/Truth')
    plt.ylim((-3, 3))
    plt.hlines([1.10, 0.90],
        xmin=sizes[0],
        xmax=sizes[len(sizes)-1],
        linestyles='dashed',
        linewidth=1,
        color='lightgray')
    plt.vlines([30, 100, 500, 1000, 5000],
        ymin=-3,
        ymax=3,
        linestyles='dotted',
        linewidth=1,
        color='lightgray')
    plt.title(f"{title} {round(100*P[idx])}th Percentile")
    plt.show()

#%%
build_plot(Xgamma_sim, Xgamma_p, 4, P, 'Gamma', sizes)

#%%
build_plot(Xnormal_sim, Xnormal_p, 3, P, 'Normal', sizes)
