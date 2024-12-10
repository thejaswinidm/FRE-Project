# -*- coding: utf-8 -*-
"""FRE-greeks+deltaipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KMGee0oPP7mebFxzTHnYOGOrECXYY0Qu
"""

from scipy.stats import norm
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import streamlit as st

class Option:
    """
    Represents an option contract with various pricing methods and Greek calculations.
    """
    def __init__(self, option_type, ticker, spot, strike, implied_vol, T, r, q=0, day_counts=252, call_option=True, position=0, barrier=None):
        if option_type not in ["european", "asian", "up-and-out", "down-and-in"]:
            raise ValueError("Invalid option type.")

        self.option_type = option_type
        self.ticker = ticker
        self.spot = spot
        self.strike = strike
        self.implied_vol = implied_vol
        self.T = T
        self.r = r
        self.q = q
        self.day_counts = day_counts
        self.call_option = call_option
        self.position = position
        self.barrier = barrier

        self.original_price = self.price_option(num_simulations=10000)
        self.current_price = self.original_price

    """
        Price the option based on its type using Monte Carlo simulation.
        """
    def price_option(self, num_simulations):
        if self.option_type == "european":
            price = self._price_european(num_simulations)
        elif self.option_type == "asian":
            price = self._price_asian(num_simulations)
        elif self.option_type == "up-and-out":
            price = self._price_up_and_out(num_simulations)
        elif self.option_type == "down-and-in":
            price = self._price_down_and_in(num_simulations)

        self.current_price = price
        return price
        
    def calculate_delta(self, num_simulations=1000, epsilon=1e-4):
        """
        Calculate delta using finite difference method.
        """
        original_spot = self.spot
        self.spot += epsilon
        price_up = self.price_option(num_simulations)
        self.spot -= 2 * epsilon
        price_down = self.price_option(num_simulations)
        self.spot = original_spot
        self.price_option(num_simulations)
        return (price_up - price_down) / (2 * epsilon)

    def calculate_gamma(self, num_simulations=1000, epsilon=1e-4):
        """
        Calculate gamma using finite difference method.
        """
        original_spot = self.spot
        self.spot += epsilon
        delta_up = self.calculate_delta(num_simulations, epsilon)
        self.spot -= 2 * epsilon
        delta_down = self.calculate_delta(num_simulations, epsilon)
        self.spot = original_spot
        #self.price_option(num_simulations)
        return (delta_up - delta_down) / (2 * epsilon)

    def calculate_vega(self, num_simulations=1000, epsilon=0.01):
        """
        Calculate vega using finite difference method.
        """
        original_vol = self.implied_vol
        self.implied_vol += epsilon
        price_up = self.price_option(num_simulations)
        self.implied_vol -= 2 * epsilon
        price_down = self.price_option(num_simulations)
        self.implied_vol = original_vol
        #self.price_option(num_simulations)
        return (price_up - price_down) / (2 * epsilon) * 0.01

    def calculate_theta(self, num_simulations=1000, epsilon=1/252):
        """
        Calculate theta using finite difference method.
        """
        original_T = self.T
        self.T -= epsilon
        price_down = self.price_option(num_simulations)
        self.T = original_T
        self.price_option(num_simulations)
        return -(self.current_price - price_down) * (1/252)

    def calculate_rho(self, num_simulations=1000, epsilon=0.0001):
        """
        Calculate rho using finite difference method.
        """
        original_r = self.r
        self.r += epsilon
        price_up = self.price_option(num_simulations)
        self.r -= 2 * epsilon
        price_down = self.price_option(num_simulations)
        self.r = original_r
        ##self.price_option(num_simulations)
        return (price_up - price_down) / (2 * epsilon) * 0.01

    def calculate_greeks(self, num_simulations=1000):
        """
        Calculate all Greeks for the option.
        """
        return {
            'delta': self.calculate_delta(num_simulations),
            'gamma': self.calculate_gamma(num_simulations),
            'vega': self.calculate_vega(num_simulations),
            'theta': self.calculate_theta(num_simulations),
            'rho': self.calculate_rho(num_simulations)
        }

    """
        Price European option using Monte Carlo simulation.
        """
    def _price_european(self, num_simulations):
        sim_prices = self._price_modeling(self.spot, self.implied_vol, num_simulations, self.T, self.r, self.q, self.day_counts)
        print(sim_prices.iloc[:, -1].head())
        print(self.strike)

        if self.call_option:
            payoffs = np.maximum((sim_prices.iloc[:, -1].values - self.strike), 0)
        else:
            payoffs = np.maximum(self.strike - sim_prices.iloc[:, -1], 0)
        return np.mean(payoffs) * np.exp(-self.r * self.T / self.day_counts)
    
    def _price_asian(self, num_simulations):
        """
        Price Asian option using Monte Carlo simulation.
        """
        sim_prices = self._price_modeling(self.spot, self.implied_vol, num_simulations, self.T, self.r, self.q, self.day_counts)
        avg_prices = sim_prices.mean(axis=1)
        if self.call_option:
            payoffs = np.maximum(avg_prices - self.strike, 0)
        else:
            payoffs = np.maximum(self.strike - avg_prices, 0)
        return np.mean(payoffs) * np.exp(-self.r * self.T / self.day_counts)
    
    def _price_up_and_out(self, num_simulations):
        """
        Price Up-and-Out barrier option using Monte Carlo simulation.
        """
        sim_paths = self._price_modeling(self.spot, self.implied_vol, num_simulations, self.T, self.r, self.q, self.day_counts)
        path_max = sim_paths.max(axis=1).values  # Max price along each path
        in_barrier = path_max < self.barrier  # Check if the barrier is never breached
        final_prices = sim_paths.iloc[:, -1].values  # Prices at maturity
        if self.call_option:
            payoffs = np.maximum(final_prices - self.strike, 0)
        else:
            payoffs = np.maximum(self.strike - final_prices, 0)
        payoffs = payoffs * in_barrier  # Only keep payoffs where barrier is not breached
        return np.mean(payoffs) * np.exp(-self.r * self.T / self.day_counts)

    def _price_down_and_in(self, num_simulations):
        """
        Price Down-and-In barrier option using Monte Carlo simulation.
        """
        # Simulate price paths
        sim_paths = self._price_modeling(self.spot, self.implied_vol, num_simulations, self.T, self.r, self.q, self.day_counts)
        path_min = sim_paths.min(axis=1)  # Minimum price for each path
        in_barrier = path_min <= self.barrier  # True if the path crosses or touches the barrier
        final_prices = sim_paths.iloc[:, -1]  # Last column of the DataFrame (prices at maturity)
        if self.call_option:
            payoffs = np.maximum(final_prices - self.strike, 0)
        else:
            payoffs = np.maximum(self.strike - final_prices, 0)
        payoffs = payoffs * in_barrier
        return np.mean(payoffs) * np.exp(-self.r * self.T / self.day_counts)

    def _price_modeling(self, S0, iv, N, T, r, q, day_counts):
        """
        Simulate price paths using geometric Brownian motion.
        """
        t = T / day_counts
        n_steps = max(int(T), 1)  # Ensure at least one step
        dt = t / n_steps if n_steps > 0 else t  # Avoid division by zero
        prices = {}
        np.random.seed(42)
        for i in range(N):
            W_t = 0
            daily_prices = [S0]
            for j in range(n_steps + 1):
                W_t += np.random.normal(0, 1)
                if dt > 0:
                    S_t = S0 * np.exp((r - q - 0.5 * iv**2) * dt + iv * np.sqrt(dt) * W_t)
                else:
                    S_t = S0  # If dt is zero, price doesn't change
                daily_prices.append(S_t)
            prices[i] = daily_prices
        return pd.DataFrame.from_dict(prices, orient='index')


def calculate_portfolio_over_time(sp500_data, portfolio):
    results = []
    for date in sp500_data.index:
        for option in portfolio[:10]:
            option.spot = sp500_data.loc[date, option.ticker]['Close']
            greeks = option.calculate_greeks()
            results.append({
                "Date": date,
                "Ticker": option.ticker,
                "Option Price": option.current_price,
                "Delta": greeks['delta'],
                "Gamma": greeks['gamma'],
                "Vega": greeks['vega'],
                "Theta": greeks['theta'],
                "Rho": greeks['rho']
            })
    return pd.DataFrame(results)


import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz


def calculate_implied_vol(ticker, start_date=None):
    american_tz = pytz.timezone('America/New_York')  # You can change this to your specific timezone
    end_date = datetime.now().date()
    if end_date > datetime(2024, 12, 8).date():
        end_date = datetime(2024, 12, 8).date()

    # Adjust end_date if it's a weekend
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=2)

    # Use provided start_date or calculate it as 3 years prior to end_date
    if start_date is None:
        start_date = end_date - timedelta(days=3*365)
    else:
        start_date = pd.to_datetime(start_date).date()
        start_date = start_date - timedelta(days=3*365)

    # Ensure start_date is a business day
    while start_date.weekday() >= 5:
        start_date += timedelta(days=2)

    # if start_date is None:
    #     start_date = end_date - timedelta(days=3*365)  # 3 years prior
    # else:
    #     start_date = pd.to_datetime(start_date).date()

    # Ensure start_date is not in the future
    # start_date = min(start_date, end_date - timedelta(days=(3*365)))

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data fetched for ticker {ticker} between {start_date} and {end_date}.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


    close_prices = data['Close']
    
    # Calculate daily returns
    returns = close_prices.pct_change().dropna()

    # Parameters for EWMA
    lambda_ = 0.94  # decay factor commonly used in finance

    # Compute EWMA of squared returns
    ewma_variance = returns.pow(2).ewm(alpha=1-lambda_).mean()
    
    # Calculate the square root of the variance estimate
    # Multiply by sqrt(252) to annualize the volatility (assuming 252 trading days per year)
    implied_volatility = np.sqrt(ewma_variance * 252)
    
    return implied_volatility.iloc[-1]

# portfolio_df = calculate_portfolio_over_time(backtest, portfolio)
# print("Portfolio DataFrame:", portfolio_df.head(10))

# portfolio_df


# Streamlit app
st.title("Option Pricing App")

# Streamlit app with multiple pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Portfolio Management", "Backtesting", "Term Structure"])

# Global portfolio list (stored in session state)
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if page == "Home":
    # Initialize session state variables
    if "price_button_clicked" not in st.session_state:
        st.session_state.price_button_clicked = False
    if "option_price" not in st.session_state:
        st.session_state.option_price = None

    # Input fields for user to specify option details
    ticker = st.text_input("Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", dt.date.today())
    time_to_maturity = st.number_input("Time to Maturity (in days)", min_value=1, step=1, value=23)

    
    data = yf.download(ticker, start=start_date, end=start_date + dt.timedelta(days=1))
    
    spot_price = data["Close"].iloc[-1] if not data.empty else None
    if isinstance(spot_price, pd.Series):
        spot = spot_price.iloc[0] if not spot_price.empty else 150.0
    elif isinstance(spot_price, float) or isinstance(spot_price, int):
        spot = spot_price
    else:
        spot = 150.0 
    strike = spot * 1.05
    implied_vol = calculate_implied_vol(ticker, start_date)
    interest_rate = 0.05

    spot = st.number_input("Spot Price", value=float(spot), step=0.001, format="%.5f")
    strike = st.number_input("Strike Price", value=float(strike), step=0.001, format="%.5f")
    if implied_vol is not None:
        implied_vol_input = st.number_input("Implied Volatility", value=float(implied_vol.iloc[0]), step=0.001, format="%.5f")
    else:
        implied_vol_input = st.number_input("Implied Volatility", value=0.0, step=0.001, format="%.5f")    
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, step=0.001, format="%.5f")
    dividend_rate = st.number_input("Dividend Rate", value=0.0, step=0.001, format="%.5f")

    option_type = st.selectbox("Option Type", ["european", "asian", "up-and-out", "down-and-in"])
    if option_type == "up-and-out":
        st.info("Up-and-Out options can only be Calls.")
        call_option = True  # Force Call for Up-and-Out
    elif option_type == "down-and-in":
        st.info("Down-and-In options can only be Puts.")
        call_option = False  # Force Put for Down-and-In
    else:
        # Allow the user to select Call or Put for other types
        call_option = st.selectbox("Option Style", ["Call", "Put"]) == "Call"

    barrier = None
    if option_type in ["up-and-out", "down-and-in"]:
        barrier = st.number_input("Barrier Level", min_value=0.0, value=float(strike * (1.2 if option_type == "up-and-out" else 0.8)))

        if option_type == "up-and-out" and strike > barrier:
            st.error("For an up-and-out call, the strike price cannot exceed the barrier.")
        elif option_type == "down-and-in" and strike < barrier:
            st.error("For a down-and-in put, the strike price cannot be less than the barrier.")
    

    
    if st.button("Price Option"):
        # Create an Option instance
        option = Option(
            option_type=option_type,
            ticker=ticker,
            spot=spot,
            strike=strike,
            implied_vol=implied_vol,
            T=time_to_maturity,
            r=interest_rate,
            q=dividend_rate,
            call_option=call_option,
            barrier=barrier
        )

        # Price the option
        #price = option.price_option()
        st.success(f"The option price is: ${option.original_price:.5f}")
        st.success(f"The delta is: {option.calculate_delta():.5f}")
        st.success(f"The gamma is: {option.calculate_gamma():.5f}")
        st.success(f"The vega is: {option.calculate_vega():.5f}")
        st.success(f"The theta is: {option.calculate_theta():.5f}")
        st.success(f"The rho is: {option.calculate_rho():.5f}")



elif page == "Portfolio Management":
    st.header("Manage Your Portfolio")

    # Input fields for user to specify option details
    ticker = st.text_input("Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", dt.date.today())
    time_to_maturity = st.number_input("Time to Maturity (in days)", min_value=1, step=1, value=30)

    
    data = yf.download(ticker, start=start_date, end=start_date + dt.timedelta(days=1))
    spot_price = data["Close"].iloc[-1] if not data.empty else None
    if isinstance(spot_price, pd.Series):
        spot = spot_price.iloc[0] if not spot_price.empty else 150.0
    elif isinstance(spot_price, float) or isinstance(spot_price, int):
        spot = spot_price
    else:
        spot = 150.0 
    strike = spot * 1.05
    implied_vol = calculate_implied_vol(ticker, start_date).any() or 0
    interest_rate = 0.05

    spot = st.number_input("Spot Price", value=float(spot), step=0.001)
    strike = st.number_input("Strike Price", value=float(strike), step=0.001)
    if implied_vol is not None:
        implied_vol_input = st.number_input("Implied Volatility", value=float(implied_vol), step=0.001, format="%.5f")
    else:
        implied_vol_input = st.number_input("Implied Volatility", value=0.0, step=0.001, format="%.5f")    
    # interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, step=0.001, format="%.5f")
    interest_rate = st.number_input("Risk-Free Interest Rate", value=float(interest_rate), step=0.001)
    dividend_rate = st.number_input("Dividend Rate", value=float(0), step=0.001)

    option_type = st.selectbox("Option Type", ["european", "asian", "up-and-out", "down-and-in"])
    if option_type == "up-and-out":
        st.info("Up-and-Out options can only be Calls.")
        call_option = True  # Force Call for Up-and-Out
    elif option_type == "down-and-in":
        st.info("Down-and-In options can only be Puts.")
        call_option = False  # Force Put for Down-and-In
    else:
        # Allow the user to select Call or Put for other types
        call_option = st.selectbox("Option Style", ["Call", "Put"]) == "Call"

    if option_type in ["up-and-out", "down-and-in"]:
        barrier = st.number_input("Barrier Level", min_value=0.0, value=float(strike * (1.2 if option_type == "up-and-out" else 0.8)))

        if option_type == "up-and-out" and strike > barrier:
            st.error("For an up-and-out call, the strike price cannot exceed the barrier.")
        elif option_type == "down-and-in" and strike < barrier:
            st.error("For a down-and-in put, the strike price cannot be less than the barrier.")

        if st.button("Add to Portfolio"):
            new_option = Option(
                option_type=option_type,
                ticker=ticker,
                spot=spot,
                strike=strike,
                implied_vol=implied_vol,
                T=time_to_maturity,
                r=interest_rate,
                q=dividend_rate,
                call_option=call_option,
                barrier=barrier
            )
            st.session_state.portfolio.append(new_option)
            st.success(f"Added {ticker} option to portfolio!")
    else:
        if st.button("Add to Portfolio"):
            new_option = Option(
                option_type=option_type,
                ticker=ticker,
                spot=spot,
                strike=strike,
                implied_vol=implied_vol,
                T=time_to_maturity,
                r=interest_rate,
                q=dividend_rate,
                call_option=call_option
            )
            st.session_state.portfolio.append(new_option)
            st.success(f"Added {ticker} option to portfolio!")

    # Display portfolio
    st.subheader("Current Portfolio")
    if st.session_state.portfolio:
        portfolio_data = [{
            "Ticker": option.ticker,
            "Option Type": option.option_type,
            "Call/Put": "Call" if option.call_option else "Put",
            "Spot Price": option.spot,
            "Strike Price": option.strike,
            "Implied Volatility": option.implied_vol,
            "Time to Maturity (years)": option.T,
            "Risk-Free Rate": option.r,
            "Original Option Price": option.original_price,
            "Delta": option.calculate_delta(),
            "Gamma": option.calculate_gamma(),
            "Vega": option.calculate_vega(),
            "Thea": option.calculate_theta(),
            "Rho": option.calculate_rho()
        } for option in st.session_state.portfolio]
        st.table(pd.DataFrame(portfolio_data))
    else:
        st.write("No options in portfolio yet.")

elif page == "Backtesting":
    st.header("Backtest Portfolio Greeks")  # Section header for backtesting

    # Check if there are options in the portfolio
    if not st.session_state.portfolio:
        st.warning("No options in portfolio! Go to the Portfolio Management page to add options.")
    else:
        st.subheader("Current Portfolio")  # Display portfolio details
        
        # Create a structured table of portfolio details
        portfolio_data = [{
            "Ticker": option.ticker,
            "Option Type": option.option_type,
            "Call/Put": "Call" if option.call_option else "Put",
            "Spot Price": option.spot,
            "Strike Price": option.strike,
            "Implied Volatility": option.implied_vol,
            "Time to Maturity (years)": option.T,
            "Risk-Free Rate": option.r,
            "Original Option Price": option.original_price,
            "Delta": option.calculate_delta(),
            "Gamma": option.calculate_gamma(),
            "Vega": option.calculate_vega(),
            "Thea": option.calculate_theta(),
            "Rho": option.calculate_rho()
        } for option in st.session_state.portfolio]
        st.table(pd.DataFrame(portfolio_data))  # Display portfolio in a table
        
        st.subheader("Enter Backtesting Time Window")  # Prompt user for backtesting inputs
        # Input for historical data range
        start_date = st.date_input("Start Date", dt.date.today())
        end_date = st.date_input("End Date", dt.date.today())

        if st.button("Backtest"):
            # Fetch historical market data for the portfolio's tickers
            tickers = [option.ticker for option in st.session_state.portfolio]
            historical_data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker")
            print(historical_data)  # Debug: print downloaded data
            
            if len(tickers) > 1:
                # Filter to only Close prices for multi-ticker data
                historical_data = historical_data.loc[:, (slice(None), "Close")]
            
            # Calculate portfolio Greeks over the selected time period
            results_df = calculate_portfolio_over_time(historical_data, st.session_state.portfolio)

            # Display the backtest results
            st.write("Backtest Results:")
            st.dataframe(results_df)

            # Plot the Greeks over time
            greek_pivot = pd.DataFrame()
            for greek in ["Delta", "Vega", "Rho"]:
                if len(tickers) > 1:
                    greek_data = results_df[["Date", "Ticker", greek]]
                    greek_pivot = greek_data.pivot(index="Date", columns="Ticker", values=greek).fillna(0)
                    dict_greek = greek_pivot.to_dict(orient='dict')
                    greek_pivot = pd.DataFrame(dict_greek)
                else:
                    print(results_df)  # Debug: print single-ticker results
                
                # Display the line chart for each Greek
                st.subheader(f"{greek} Over Time")
                st.line_chart(greek_pivot)

elif page == "Term Structure":
    st.title("Term Structure of Volatility")  # Section header for term structure

    # Input fields for user to specify option details
    ticker = st.text_input("Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", dt.date.today())
    
    # Fetch spot price for the given ticker and date
    data = yf.download(ticker, start=start_date, end=start_date + dt.timedelta(days=1))
    spot_price = data["Close"].iloc[-1] if not data.empty else None
    if isinstance(spot_price, pd.Series):
        spot = spot_price.iloc[0] if not spot_price.empty else 150.0
    elif isinstance(spot_price, float) or isinstance(spot_price, int):
        spot = spot_price
    else:
        spot = 150.0  # Default value if no data is available
    strike = spot * 1.05  # Strike price as 5% above spot
    implied_vol = calculate_implied_vol(ticker, start_date).iloc[0]  # Calculate implied volatility
    interest_rate = calculate_implied_vol(ticker, start_date) / 100  # Calculate risk-free rate

    # Inputs for various option parameters
    spot = st.number_input("Spot Price", value=float(spot))
    strike = st.number_input("Strike Price", value=float(strike))
    implied_vol = st.number_input("Implied Volatility", value=float(implied_vol))
    interest_rate = st.number_input("Risk-Free Interest Rate", value=float(interest_rate.iloc[0]))
    dividend_rate = st.number_input("Dividend Rate", value=float(0))

    # Select option type and enforce conditions for barrier options
    option_type = st.selectbox("Option Type", ["european", "asian", "up-and-out", "down-and-in"])
    if option_type == "up-and-out":
        st.info("Up-and-Out options can only be Calls.")
        call_option = True  # Force Call for Up-and-Out
    elif option_type == "down-and-in":
        st.info("Down-and-In options can only be Puts.")
        call_option = False  # Force Put for Down-and-In
    else:
        # Allow the user to select Call or Put for other types
        call_option = st.selectbox("Option Style", ["Call", "Put"]) == "Call"
    
    # Generate range of time to maturities for the term structure plot
    time_to_maturities = np.linspace(21, 504, 24)  # 1 month to 2 years in 24 steps
    
    if st.button('Plot Term Structure'):
        term_structure = []
        for t in time_to_maturities:
            # Create Option object and calculate price
            option = Option(
                option_type=option_type,
                ticker=ticker,
                spot=spot,
                strike=strike,
                implied_vol=implied_vol,
                T=t/365,  # Convert days to years
                r=interest_rate,
                q=dividend_rate,
                call_option=call_option
            )
            price = option.price_option(1000)  # Calculate option price
            term_structure.append({"Maturity (Days)": t, "Option Price": price})

        # Create and display the term structure DataFrame
        term_structure_df = pd.DataFrame(term_structure)
        st.subheader("Term Structure Data")
        st.dataframe(term_structure_df)

        # Plot the term structure
        st.subheader("Term Structure of Option Prices")
        fig, ax = plt.subplots()
        ax.plot(term_structure_df["Maturity (Days)"], term_structure_df["Option Price"], marker='o')
        ax.set_xlabel("Time to Maturity (Days)")
        ax.set_ylabel("Option Price")
        ax.set_title("Term Structure of Option Prices")
        ax.grid(True)
        st.pyplot(fig)