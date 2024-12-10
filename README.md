# FRE-Project

FRE 6811: Final Project - Planning the Software

Problem Definition
In this project we aim to calculate the prices of options European and Asian options on
stocks by implementing Monte Carlo simulation techniques based on the Geometric Brownian
Motion to model asset prices. The primary objective is to simulate the underlying asset’s price
path under realistic assumptions and price these different types of options accurately.
Additionally, we will calculate the Greeks (Delta, Gamma, Vega, Theta, and Rho) for each
option to provide insight into the option’s sensitivity to various risk factors such as the
underlying asset price, time decay, volatility, and interest rates.
To accomplish this, we’ll develop a Monte Carlo simulation using the Geometric
Brownian Motion to capture the price path of the underlying asset. We will then implement
pricing algorithms specific to each option type, accounting for payoff structures and barrier
conditions where applicable. We will calculate the Greeks and develop a delta-hedging strategy
by applying a finite differences method. This hedging strategy will be backtested over historical
price data to evaluate its effectiveness in managing risk.

Software Requirements Specification (SRS)
1. Introduction
a. Purpose:
i. The purpose of this document is to define the requirements for a
Python-based application that calculates the prices of European, Asian,
and up-and-out options using Monte Carlo simulation with geometric
Brownian motion. Additionally, the application will calculate the Greeks
for each option and implement a delta hedging strategy for backtesting.
b. Scope:
i. This project aims to simulate stock price paths, price options, calculate
their sensitivities (Greeks), and test delta hedging strategies. It will be
used by financial engineers or quantitative researchers to explore and
validate option pricing models in a simulated environment.
2. General Description
a. Functions
i. Monte Carlo Simulation: Simulate stock price paths based on GBM.
ii. Option Pricing: Calculate the prices of European, Asian, and up-and-out
options.
iii. Calculation of Greeks: Compute Delta, Gamma, Vega, Theta, and Rho for
each option.
iv. Delta Hedging: Backtest a delta-hedging strategy over historical data.
b. User Community
i. The primary users are expected to have knowledge of finance, especially
derivatives pricing, and some familiarity with Python programming.
3. Functional Requirements
a. Monte Carlo Simulation
i. Description: Generate multiple stock price paths using GBM.
ii. Inputs: Initial stock price, volatility, time to maturity, risk-free rate, and
number of simulations.
iii. Outputs: A set of simulated stock price paths for each option type.
b. Option Pricing
i. Description: Calculate the option prices for European, Asian, and
up-and-out options.
ii. Inputs: Simulated stock price paths, option type, option strike price,
risk-free rate, volatility, and time to maturity.
iii. Outputs: Option prices.
c. Greeks Calculation
i. Description: Estimate Delta, Gamma, Vega, Theta, and Rho for each
option using finite difference methods.
ii. Inputs: Simulated stock paths, option parameters, and a small perturbation
value used to calculate sensitivity (epsilon).
iii. Outputs: Numerical values for each Greek.
d. Delta Hedging Strategy
i. Description: Implement a delta hedging strategy based on the calculated
Delta, adjust the hedge periodically, and backtest performance.
ii. Inputs: Delta values, simulated stock prices, and rebalancing frequency.
iii. Outputs: Hedging PnL and performance metrics.
4. User Interface Requirements
a. User Interfaces
i. The application will run in a command-line interface (CLI) with outputs
displayed as console logs and possibly graphical charts using matplotlib.
There’s also potential for integrating the solution into a streamlined python
application if time permits for better user experience.
b. Software Interfaces
i. Libraries: Python libraries such as Pandas, NumPy, SciPy, and matplotlib.
ii. Data Sources: Bloomberg
5. Non-Functional Requirements
a. Performance Requirements
i. The Monte Carlo simulation should be optimized to handle large numbers
of simulations (e.g., 10,000 paths) within reasonable processing time.
b. Usability Requirements
i. The application should be easy to set up and run by following installation
instructions. Clear error messages should be provided for incorrect or
missing inputs.
c. Reproducibility
i. The simulation and calculations should produce consistent results across
different runs with the same random seed.
d. Scalability
i. The system should be able to scale to run more simulations or price
additional types of options with minimal adjustments.
6. Other Requirements
a. Documentation
i. Include a README file with setup instructions, usage examples, and
descriptions of the core components. We will create a GitHub repository to
store our code and collaborate on writing the code to complete the
objectives of the project.
Data Sources
Bloomberg: Used to gather underlying asset data to price the options. Will also use
Bloomberg to calculate option prices to compare our option pricing models to those within the
Bloomberg terminal to assess accuracy.
