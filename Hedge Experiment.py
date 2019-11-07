import torch
import numpy as np
import multiprocess
np.seterr(divide='ignore', invalid='ignore')

"""
----------------
Description:
Discrete delta hedging strategy for a down-and-out barrier option in the Black-Scholes model.
----------------
"""

### Store the number of cores to allow for parallel processing
nCores = multiprocess.cpu_count()

### Set up parameters for the Monte Carlo simulation
Nrep = 100
nHedges = 25
dt = 1.0 / nHedges
volatility = 0.15
rate = 0.05

B_t = np.ones(Nrep)
spotT = np.ones(Nrep)
spotT[spotT == 1.0] = 100.0
initialoutlay = 15.4675

pfValue = np.ones(Nrep)
pfValue[pfValue == 1.0] = initialoutlay

hSTOCK = up_and_out_BS_AVCV(100.0, 90.0, 75.0, 0.15, 0.05, 1.0, 5000)
hBANK = (pfValue - spotT * hSTOCK) / B_t

### Update the portfolio holdings for each hedge point
for i in range(1, nHedges):

    pool = multiprocess.Pool(processes=nCores, initargs=(i, dt))

    z = np.random.normal(loc=0, scale=1, size=Nrep)
    dW = volatility * z * np.sqrt(dt)
    paths = (rate - volatility * volatility / 2) * dt + dW
    spotT = spotT * np.exp(paths)
    B_t = B_t * np.exp(rate * dt)
    pfValue = hSTOCK * spotT + hBANK * B_t
    hSTOCK = np.array(pool.map(lambda x: up_and_out_BS_AVCV(x, 90.0, 75.0, 0.15, 0.05, 1.0 - dt * (i - 1), 5000), spotT))
    hBANK = (pfValue - hSTOCK * spotT) / B_t

    pool.close()
    pool.join()

z = np.random.normal(loc=0, scale=1, size=Nrep)
dW = volatility * z * np.sqrt(dt)
paths = (rate - volatility * volatility / 2) * dt + dW
B_t = B_t * np.exp(rate * dt)
spotT = spotT * np.exp(paths)
pfValue = hSTOCK * spotT + hBANK * B_t

print(list(pfValue))
print(list(spotT))
