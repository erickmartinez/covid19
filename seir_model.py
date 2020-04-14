# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:42:55 2020

@author: Erick
"""

import numpy as np
from scipy.integrate import solve_ivp

# https://www.idmod.org/docs/hiv/model-seir.html
# The SIR model differential equations. (without vital kinetics)
def seir(t, y, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def seir_wvd(t, y, N, beta, gamma, sigma, nu, mu):
    S, E, I, R = y
    dSdt = mu*N - nu*S - beta * S * I / N
    dEdt = beta * S * I / N - (nu + sigma)*E 
    dIdt = sigma * E - (gamma + nu) * I
    dRdt = gamma * I - nu*R
    return [dSdt, dEdt, dIdt, dRdt]

def seir_model(t: np.ndarray, N: int, beta: float, gamma: float, sigma: float,
               **kwargs):
    """
    Solves the Susceptible-Exposed-Infected-Removed ODE
    
    Parameters
    ----------
    t: np.ndarray
        The time (days)
    N: int
        The total population
    beta: float
        The contact rate beta (1/days)
    gamma: float
        The mean recovery rate (1/days)
    kwargs: keyword arguments
        I0: int
            The initial number of infected individuals
        R0: int
            The initial number of recovered individuals
        E0: int
            The initial number of exposed individuals
    """

    I0 = kwargs.get('I0',1)
    R0 = kwargs.get('R0',0)
    E0 = kwargs.get('E0',1)

#    # Total population, N.
#    N = 1000
#    # Initial number of infected and recovered individuals, I0 and R0.
#    I0, R0 = 1, 0
#    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - E0
#    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#    beta, gamma = 0.2, 1./10 
#    # A grid of time points (in days)

    # Initial conditions vector
    y0 = S0, E0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    #ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    sol = solve_ivp(seir, [np.amin(t), np.amax(t)], y0, # t_eval=t,
                    args=(N, beta, gamma, sigma), 
                    method='DOP853',
                    dense_output=True)
    
    
    #S, I, R = ret.T
#    y = sol.sol(t)
#    S, I, R = y
    return sol

def seir_model_wvd(t: np.ndarray, N: int, beta: float, gamma: float, sigma: float,
                   mu: float, nu: float, **kwargs):
    """
    Solves the Susceptible-Exposed-Infected-Removed ODE
    
    Parameters
    ----------
    t: np.ndarray
        The time (days)
    N: int
        The total population
    beta: float
        The contact rate beta (1/days)
    gamma: float
        The mean recovery rate (1/days)
    mu: float
        The mean birth rate (1/days)
    nu: float
        The mean death rate (1/days)
    kwargs: keyword arguments
        I0: int
            The initial number of infected individuals
        R0: int
            The initial number of recovered individuals
        E0: int
            The initial number of exposed individuals
    """

    I0 = kwargs.get('I0',1)
    R0 = kwargs.get('R0',0)
    E0 = kwargs.get('E0',1)

#    # Total population, N.
#    N = 1000
#    # Initial number of infected and recovered individuals, I0 and R0.
#    I0, R0 = 1, 0
#    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - E0
#    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#    beta, gamma = 0.2, 1./10 
#    # A grid of time points (in days)

    # Initial conditions vector
    y0 = S0, E0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    #ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    sol = solve_ivp(seir_wvd, [np.amin(t), np.amax(t)], y0, # t_eval=t,
                    args=(N, beta, gamma, sigma, mu, nu), 
                    method='DOP853',
                    dense_output=True)
    
    
    #S, I, R = ret.T
#    y = sol.sol(t)
#    S, I, R = y
    return sol

