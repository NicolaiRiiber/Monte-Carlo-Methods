import torch
import numpy as np

def up_and_out_BS_AV(spot, strike, barrier, volatility, rate, maturity, nPaths):

    """
    -----------------
    Description:
    This function calculates the price of a european up-and-out barrier option
    using Monte Carlo simulation in a Black-Scholes model with antithetic variates.
    -----------------

    Input:
    :param spot: the current price of the underlying asset
    :param strike: the strike price of the option
    :param barrier: the barrier level
    :param volatility: volatility of the underlying asset
    :param rate: the risk free interest rate
    :param maturity: the expiry date of the option
    :param nPaths: number of simulated paths
    -----------------

    Return:
    A list containing a vector of payoffs and a variable containing the sum of payoffs.
    -----------------
    """

    ### Convert all input parameters to tensors.
    torch.manual_seed(2)
    spot = torch.tensor(float(spot), requires_grad=True)
    strike = torch.tensor(float(strike), requires_grad=True)
    volatility = torch.tensor(float(volatility), requires_grad=True)
    rate = torch.tensor(float(rate), requires_grad=True)
    barrier = torch.tensor(float(barrier), requires_grad=True)
    time_to_maturity = torch.tensor(float(maturity), requires_grad=True)

    ### Set up simulations paramters.
    dt = torch.tensor(float(1 / 252))
    nSteps = int(maturity / dt)
    z = torch.randn(size=(nPaths, nSteps))
    z = (z - torch.mean(z)) / torch.std(z)  # Matching the first and second moment.
    dW = volatility * z * torch.sqrt(dt)
    paths = (rate - volatility * volatility / 2) * dt + dW
    paths = spot * torch.exp(torch.cumsum(paths, dim=1))
    AV_paths = (rate - volatility * volatility / 2) * dt - dW
    AV_paths = spot * torch.exp(torch.cumsum(AV_paths, dim=1))
    discount_rate = torch.exp(-rate * time_to_maturity)

    ### Determine if the stock prices crosses the barrier.
    alive = torch.ones(size=(nPaths,))
    AV_alive = torch.ones(size=(nPaths,))
    smoothing_spread = torch.tensor(float(spot * 0.05))
    max_st = torch.max(paths, dim=1)
    max_st_AV = torch.max(AV_paths, dim=1)
    for i in range(0, nPaths):
        if max_st[0][i] > barrier + smoothing_spread:
            alive[i,] = 0.0
            continue
        if max_st[0][i] > barrier - smoothing_spread:
            alive[i,] = ((barrier + smoothing_spread) - max_st[0][i]) / (2.0 * smoothing_spread)
            continue
        if max_st_AV[0][i] > barrier + smoothing_spread:
            AV_alive[i,] = 0.0
            continue
        if max_st_AV[0][i] > barrier - smoothing_spread:
            AV_alive[i,] = ((barrier + smoothing_spread) - max_st_AV[0][i]) / (2.0 * smoothing_spread)
            continue

    ### Calculate the present value of future payoffs.
    payoff = torch.max(paths[:, -1] - strike, torch.zeros(size=(nPaths,))) * alive
    payoff_AV = torch.max(AV_paths[:, -1] - strike, torch.zeros(size=(nPaths,))) * AV_alive
    payoff_estimate = discount_rate * (payoff + payoff_AV) / 2
    cummulative = torch.cumsum(payoff_estimate, dim=0)

    return [cummulative.detach().numpy(), payoff_estimate.detach().numpy()]

data = up_and_out_BS_AV(100.0, 90.0, 140.0, 0.15, 0.05, 1.0, 75000)

print("Estimate :", round(np.mean(data[1]), 4))
print("95% Confidence Interval: ", [round(data[0][-1] / 75000 - 1.96 * np.std(data[1]) / np.sqrt(75000), 4), round(data[0][-1] / 75000 + 1.96 * np.std(data[1]) / np.sqrt(75000), 5)])
print("Variance: ", round(np.var(data[1]), 4))
print("Standard Error: ", round(np.std(data[1])/np.sqrt(75000), 4))
