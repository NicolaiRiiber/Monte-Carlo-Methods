import torch
import numpy as np


def BS_Call(spot, strike, T, r, sigma, q):
    """
    -----------------
    Description:
    This function calculates the price of a european call option using the Black-Scholes formula.
    -----------------
    """
    gaussian = torch.distributions.normal.Normal(0.0, 1.0)
    d1 = (torch.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    return spot * gaussian.cdf(d1) - strike * torch.exp(-r * T) * gaussian.cdf(d2)


def up_and_out_BS_AVCV_AAD(spot, strike, barrier, volatility, rate, maturity, nPaths):

    """
    -----------------
    Description:
    This function calculates the price and greeks (using AAD) of a european up-and-out barrier option
    using Monte Carlo simulation in a Black-Scholes model using antithetic variates and a european call option
    as control variate.
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
    A list containing the price, delta, vega, theta, rho and gamma.
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

    ### Set up simulations parameters.
    dt = torch.tensor(float(1 / 252))
    nSteps = int(maturity / dt)
    z = torch.randn(size=(nPaths, nSteps))
    z = z - torch.mean(z)
    z = z / torch.std(z)
    dW = volatility * z * torch.sqrt(dt)

    ### Perform the Monte Carlo simulation.
    control_variate_price = BS_Call(spot, strike, time_to_maturity, rate, volatility, torch.tensor(0.0))
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
            # alive[i, ] = 1 / (1 + torch.exp(max_st[0][i] - barrier))
            alive[i,] = ((barrier + smoothing_spread) - max_st[0][i]) / (2.0 * smoothing_spread)
            # continue
    for i in range(0, nPaths):
        if max_st_AV[0][i] > barrier + smoothing_spread:
            AV_alive[i,] = 0.0
            continue
        if max_st_AV[0][i] > barrier - smoothing_spread:
            # AV_alive[i, ] = 1 / (1 + torch.exp(max_st_AV[0][i]) - barrier)
            AV_alive[i,] = ((barrier + smoothing_spread) - max_st_AV[0][i]) / (2.0 * smoothing_spread)
            # continue

    ### Calculate the present value of future payoffs.
    payoff = torch.max(paths[:, -1] - strike, torch.zeros(size=(nPaths,))) * alive
    payoff_AV = torch.max(AV_paths[:, -1] - strike, torch.zeros(size=(nPaths,))) * AV_alive
    payoff_control_variate = torch.max(paths[:, -1] - strike, torch.zeros(size=(nPaths,)))
    payoff_control_variate_AV = torch.max(AV_paths[:, -1] - strike, torch.zeros(size=(nPaths,)))
    payoff_estimate = discount_rate * (payoff + payoff_AV) / 2
    payoff_estimate_control_variate = discount_rate * (payoff_control_variate + payoff_control_variate_AV) / 2

    c = - np.cov(payoff_estimate.detach().numpy(), payoff_estimate_control_variate.detach().numpy()) / np.var(
        payoff_estimate_control_variate.detach().numpy())
    c[np.isnan(c)] = 0.0
    option_prices = payoff_estimate + torch.tensor(c[0, 1]) * (payoff_estimate_control_variate - control_variate_price)
    price = torch.mean(option_prices)

    ### Perform a backward sweep to calculate all first order greeks.
    gradient = torch.autograd.grad(price, [spot, volatility, time_to_maturity, rate], create_graph=True)
    delta = gradient[0]
    vega = gradient[1].detach().numpy()
    theta = gradient[2].detach().numpy()
    rho = gradient[3].detach().numpy()

    ### Perform a second backward sweep to calculate all second order greeks.
    delta.backward(retain_graph=True)
    gamma = spot.grad.detach().numpy()
    delta = delta.detach().numpy()

    return [price, delta, vega, theta, rho, gamma]


N = 25000
% timeit data = up_and_out_BS_AVCV_AAD(100.0, 90.0, 140.0, 0.15, 0.05, 1.0, N)
print(data)
