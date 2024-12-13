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

    def price_option(self, num_simulations):
        """
        Price the option based on its type using Monte Carlo simulation.
        """
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

    def calculate_gamma(self, num_simulations=1000, epsilon=1e-2):
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

    def calculate_theta(self, num_simulations=1000, epsilon=1):
        """
        Calculate theta using finite difference method.
        """
        original_T = self.T
        self.T -= epsilon
        price_down = self.price_option(num_simulations)
        self.T = original_T
        self.price_option(num_simulations)
        print(self.current_price)
        print(price_down)
        return -(self.current_price - price_down) / epsilon

    def calculate_rho(self, num_simulations=1000, epsilon=0.01):
        """
        Calculate rho using finite difference method.
        """
        original_r = self.r
        self.r += epsilon
        price_up = self.price_option(num_simulations)
        self.r -= 2 * epsilon
        price_down = self.price_option(num_simulations)
        self.r = original_r
        print(price_up, price_down)
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

    def _price_european(self, num_simulations):
        """
        Price European option using Monte Carlo simulation.
        """
        sim_prices = self._price_modeling(self.spot, self.implied_vol, num_simulations, self.T, self.r, self.q, self.day_counts)

        if self.call_option:
            payoffs = np.maximum(sim_prices.iloc[:, -1] - self.strike, 0)
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
        n_steps = int(T)  # Convert T to an integer
        dt = t / n_steps
        prices = {}
        np.random.seed(42)
        for i in range(N):
            W_t = 0
            daily_prices = [S0]
            for j in range(n_steps + 1):
                W_t += np.random.normal(0, 1)
                S_t = S0 * np.exp((r - q - 0.5 * iv**2) * dt + iv * np.sqrt(dt) * W_t)
                daily_prices.append(S_t)
            prices[i] = daily_prices
        return pd.DataFrame.from_dict(prices, orient='index')

def calculate_portfolio_over_time(sp500_data, portfolio):
    results = []
    for date in sp500_data.index:
        for option in portfolio[:10]:
            option.spot = sp500_data.loc[date, option.ticker]['Close']
            greeks = option.calculate_greeks()
            option.T -= 1
            results.append({
                "Date": date,
                "Ticker": option.ticker,
                "Option Price": option.current_price,
                "Time to Maturity": option.T,
                "Delta": greeks['delta'],
                "Gamma": greeks['gamma'],
                "Vega": greeks['vega'],
                "Theta": greeks['theta'],
                "Rho": greeks['rho']
            })
    return pd.DataFrame(results)

def calculate_portfolio_over_time_v2(sp500_data, portfolio):
    results = []
    for date in sp500_data.index:
        for option in portfolio[:10]:
            option.spot = float(sp500_data.loc[date, option.ticker])
            greeks = option.calculate_greeks()
            print(greeks)
            option.T -= 1
            results.append({
                "Date": date,
                "Ticker": option.ticker,
                "Option Price": option.current_price,
                "Time to Maturity": option.T,
                "Delta": greeks['delta'],
                "Gamma": greeks['gamma'],
                "Vega": greeks['vega'],
                "Theta": greeks['theta'],
                "Rho": greeks['rho']
            })
    return pd.DataFrame(results)

def calculate_implied_vol(ticker, start_date):
    end_date = pd.to_datetime(start_date)
    if end_date.weekday() >= 5:  # Adjust if it's a weekend
        end_date -= pd.offsets.BDay(1)

    # Calculate start_date as 3 years prior
    start_date = end_date - pd.DateOffset(years=3)
    if start_date.weekday() >= 5:  # Adjust if it's a weekend
        start_date -= pd.offsets.BDay(1)

    # Fetch historical close prices
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        # If API fails, fallback to sp500historical.csv
        try:
            sp500_data = pd.read_csv('sp500historical.csv', parse_dates=["Date"])
            sp500_data.set_index("Date", inplace=True)
            sp500_data = sp500_data[ticker]
            close_prices = sp500_data[(sp500_data.index >= start_date) & (sp500_data.index <= end_date)]
        except Exception as e:
            print(f"Error while fetching data for {ticker}: {e}")
            return None
    else:
        # Use API data if available
        close_prices = data["Close"]

    
    returns = close_prices.pct_change().dropna()
    # Parameters for EWMA
    lambda_ = 0.94  # decay factor commonly used in finance
    # Compute EWMA of squared returns
    ewma_variance = returns.pow(2).ewm(alpha=1-lambda_).mean()
    # Calculate the square root of the variance estimate
    # Multiply by sqrt(252) to annualize the volatility (assuming 252 trading days per year)
    implied_volatility = np.sqrt(ewma_variance * 252)
    

    try:
        return implied_volatility[-1]
    except:
        return 0


class OptionPricingApp:
    """
    A Streamlit application for option pricing and portfolio management.
    """
    def run(self):
        # Streamlit app
        st.title("Option Pricing App")

        # Streamlit app with multiple pages
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", ["Home", "Portfolio Management", "Backtesting", "Term Structure"])

        # Global portfolio list (stored in session state)
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = []
            

        if page == "Home":
            """
            Renders the home page of the application.
            Allows users to input option details and price an individual option.
            """
            # Initialize session state variables
            if "price_button_clicked" not in st.session_state:
                st.session_state.price_button_clicked = False
            if "option_price" not in st.session_state:
                st.session_state.option_price = None

            # Input fields for user to specify option details
            ticker = st.text_input("Stock Ticker", "AAPL")
            start_date = st.date_input("Start Date", dt.date.today())
            time_to_maturity = st.number_input("Time to Maturity (in trading days)", min_value=1, step=1, value=63)

            
            data = yf.download(ticker, start=start_date, end=start_date + dt.timedelta(days=1), proxy=None)
            if data.empty:
                try:
                    sp500_data = pd.read_csv('sp500historical.csv', parse_dates=["Date"])
                    sp500_data.set_index("Date", inplace=True)
                    sp500_data = sp500_data[ticker]
                    sp500_data = sp500_data[(sp500_data.index == pd.to_datetime(start_date))]
                    spot_price = sp500_data[0]
                except Exception as e:
                    print(f"Error while fetching data for {ticker}: {e}")
            else:
                # Use API data if available
                close_prices = data["Close"]
                spot_price = data["Close"].iloc[-1] if not data.empty else None

            iv = calculate_implied_vol(ticker, start_date)
            spot = spot_price or 150.0
            strike = spot * 1.05
            implied_vol = iv or 0.33
            interest_rate = 0.05

            spot = st.number_input("Spot Price", value=spot, step=0.001, format="%.5f")
            strike = st.number_input("Strike Price", value=strike, step=0.001, format="%.5f")
            implied_vol = st.number_input("Implied Volatility", value=float(implied_vol), step=0.001, format="%.5f")
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
            """
            Renders the portfolio management page.
            Allows users to add options to their portfolio and view current holdings.
            """

            st.header("Manage Your Portfolio")

            # Input fields for user to specify option details
            ticker = st.text_input("Stock Ticker", "AAPL")
            start_date = st.date_input("Start Date", dt.date.today())
            time_to_maturity = st.number_input("Time to Maturity (in trading days)", min_value=1, step=1, value=63)

            
            data = yf.download(ticker, start=start_date, end=start_date + dt.timedelta(days=1), proxy=None)
            if data.empty:
                try:
                    sp500_data = pd.read_csv('sp500historical.csv', parse_dates=["Date"])
                    sp500_data.set_index("Date", inplace=True)
                    sp500_data = sp500_data[ticker]
                    sp500_data = sp500_data[(sp500_data.index == pd.to_datetime(start_date))]
                    spot_price = sp500_data[0]
                except Exception as e:
                    print(f"Error while fetching data for {ticker}: {e}")
            else:
                # Use API data if available
                close_prices = data["Close"]
                spot_price = data["Close"].iloc[-1] if not data.empty else None

            iv = calculate_implied_vol(ticker, start_date)
            spot = spot_price or 150.0
            strike = spot * 1.05
            implied_vol = iv or 0.33
            interest_rate = 0.05

            spot = st.number_input("Spot Price", value=float(spot), step=0.001)
            strike = st.number_input("Strike Price", value=float(strike), step=0.001)
            implied_vol = st.number_input("Implied Volatility", value=float(implied_vol), step=0.001)
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
                    "Time to Maturity (in trading days)": option.T,
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
            """
            Renders the backtesting page.
            Allows users to backtest their portfolio's performance over a specified time period.
            """
            st.header("Backtest Portfolio Greeks")



            if not st.session_state.portfolio:
                st.warning("No options in portfolio! Go to the Portfolio Management page to add options.")
            else:
                st.subheader("Current Portfolio")
            
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
                
                st.subheader("Enter Backtesting Time Window")
                # Input for historical data range
                start_date = st.date_input("Start Date", dt.date.today())
                end_date = st.date_input("End Date", dt.date.today())
                

                if st.button("Backtest"):
                    # Fetch historical data
                    tickers = [option.ticker for option in st.session_state.portfolio]
                    historical_data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker")
                    
                    if historical_data.empty:
                        sp500 = pd.read_csv('sp500historical.csv', parse_dates=["Date"])
                        sp500.set_index("Date", inplace=True)
                        sp500 = sp500[tickers]
                        sp500 = sp500[(sp500.index >= pd.to_datetime(start_date)) & (sp500.index <= pd.to_datetime(end_date))]
                        results_df = calculate_portfolio_over_time_v2(sp500, st.session_state.portfolio)
                        st.write("Backtest Results:")
                        st.dataframe(results_df)

                        for greek in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
                            if len(tickers) > 1:
                                greek_data = results_df[["Date", "Ticker", greek]]
                                greek_pivot = greek_data.pivot(index="Date", columns="Ticker", values=greek).fillna(0)
                                dict_greek = greek_pivot.to_dict(orient='dict')
                                greek_pivot = pd.DataFrame(dict_greek)
                            else:
                                print(results_df)
                            st.subheader(f"{greek} Over Time")
                            st.line_chart(greek_pivot)
                    
                    else:
                        if len(tickers) > 1:
                            historical_data = historical_data.loc[:, (slice(None), "Close")]
                    
                        # Calculate Greeks for the portfolio
                        results_df = calculate_portfolio_over_time(historical_data, st.session_state.portfolio)

                        # Display results
                        st.write("Backtest Results:")
                        st.dataframe(results_df)

                        # Plot Greeks
                        for greek in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
                            if len(tickers) > 1:
                                greek_data = results_df[["Date", "Ticker", greek]]
                                greek_pivot = greek_data.pivot(index="Date", columns="Ticker", values=greek).fillna(0)
                                dict_greek = greek_pivot.to_dict(orient='dict')
                                greek_pivot = pd.DataFrame(dict_greek)
                            else:
                                print(results_df)
                            st.subheader(f"{greek} Over Time")
                            st.line_chart(greek_pivot)
        elif page =="Term Structure":
            """
            Renders the term structure page.
            Allows users to visualize the term structure of option prices for different maturities.
            """

            st.title("Term Structure of Volatility")

            # Input fields for user to specify option details
            ticker = st.text_input("Stock Ticker", "AAPL")
            start_date = st.date_input("Start Date", dt.date.today())
            
            
            data = yf.download(ticker, start=start_date, end=start_date + dt.timedelta(days=1))
            spot_price = data["Close"].iloc[-1] if not data.empty else None
            spot = spot_price or 150.0
            strike = spot * 1.05
            implied_vol = calculate_implied_vol(ticker, start_date)[0]
            interest_rate = calculate_implied_vol(ticker, start_date)[1] / 100

            spot = st.number_input("Spot Price", value=float(spot))
            strike = st.number_input("Strike Price", value=float(strike))
            implied_vol = st.number_input("Implied Volatility", value=float(implied_vol))
            interest_rate = st.number_input("Risk-Free Interest Rate", value=float(interest_rate))
            dividend_rate = st.number_input("Dividend Rate", value=float(0))

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
            
            # Time to maturity values (e.g., 1 month to 2 years)
            time_to_maturities = np.linspace(21, 504, 24)  # 1 month to 2 years in 20 steps
            
            if st.button('Plot Term Structure'):
                # Calculate option prices or volatilities for each time to maturity
                term_structure = []
                for t in time_to_maturities:
                    # Create an Option object for each maturity
                    option = Option(
                        option_type=option_type,
                        ticker=ticker,
                        spot=spot,
                        strike=strike,
                        implied_vol=implied_vol,
                        T=t,
                        r=interest_rate,
                        q=dividend_rate,
                        call_option=call_option
                    )
                    price = option.price_option(1000)
                    term_structure.append({"Maturity (Years)": t, "Option Price": price})

                # Convert to DataFrame
                term_structure_df = pd.DataFrame(term_structure)

                # Display DataFrame
                st.subheader("Term Structure Data")
                st.dataframe(term_structure_df)

                # Plot the term structure
                st.subheader("Term Structure of Option Prices")
                fig, ax = plt.subplots()
                ax.plot(term_structure_df["Maturity (Years)"], term_structure_df["Option Price"], label="Option Price")
                ax.set_xlabel("Time to Maturity (Years)")
                ax.set_ylabel("Option Price")
                ax.set_title("Term Structure of Volatility")
                ax.grid()
                st.pyplot(fig)


if __name__ == "__main__":
    app = OptionPricingApp()
    app.run()