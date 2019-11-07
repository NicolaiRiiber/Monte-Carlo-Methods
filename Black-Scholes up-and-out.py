import torch
import numpy as np

def BSuao(spot, strike, barrier, volatility, rate, maturity, nPaths):

    """
    Description:
    This function calculates the price of a european up-and-out barrier option
    using Monte Carlo simulation in a Black-Scholes model.

    Input:
    @param spot: the current price of the underlying asset
    @strike: the strike price of the option
    @barrier: the barrier level
    @volatility: volatility of the underlying asset
    @rate: the risk free interest rate
    @maturity: the expiry date of the option
    @nPaths: number of simulated paths

    Return:
    A list containing a vector of payoffs and a variable containing the sum of payoffs.
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
    z = (z - torch.mean(z)) / torch.std(z)  # First and second Moment matching
    paths = (rate - volatility * volatility / 2) * dt + volatility * z * torch.sqrt(dt)
    paths = spot * torch.exp(torch.cumsum(paths, dim=1))
    discount_rate = torch.exp(-rate * time_to_maturity)

    ### Determine if the stock prices crosses the barrier.
    alive = torch.ones(size=(nPaths,))
    smoothing_spread = torch.tensor(float(spot * 0.05))
    max_st = torch.max(paths, dim=1)
    for i in range(0, nPaths):
        if max_st[0][i] > barrier + smoothing_spread:
            alive[i,] = 0.0
            continue
        if max_st[0][i] > barrier - smoothing_spread:
            alive[i,] *= ((barrier + smoothing_spread) - max_st[0][i]) / (2.0 * smoothing_spread)
            continue

    payoff = torch.max(paths[:, -1] - strike, torch.zeros(size=(nPaths,)))
    price = discount_rate * torch.mean(payoff * alive)

    payoffs = discount_rate * payoff * alive
    cummulative = torch.cumsum(payoffs, dim=0)
    sample_sd = torch.std(payoffs)

    return [cummulative.detach().numpy(), payoffs.detach().numpy()]

% time data = BSuao(100.0, 90.0, 140.0, 0.15, 0.05, 1.0, 75000)
print("Estimate :", round(data[0][-1] / 75000, 3))
print("95% Confidence Interval: ", [round(data[0][-1] / 75000 - 1.96 * np.std(data[1]) / np.sqrt(75000), 4),
                                    round(data[0][-1] / 75000 + 1.96 * np.std(data[1]) / np.sqrt(75000), 4)])
print("Variance: ", round(np.var(data[1]), 4))
print("Standard Error: ", round(np.std(data[1]) / np.sqrt(75000), 4))

