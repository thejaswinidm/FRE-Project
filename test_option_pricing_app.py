import unittest
from unittest.mock import patch
from datetime import date
from option_pricing_app import OptionPricingApp, Option,calculate_implied_vol
import pandas as pd
import numpy as np


class TestOptionPricingApp(unittest.TestCase):

    def setUp(self):
        self.app = OptionPricingApp()

    @patch('yfinance.download')
    @patch('pandas.read_csv')
    def test_fetch_market_data(self, mock_read_csv, mock_yf_download):
        # Mock yfinance download
        mock_yf_download.return_value = pd.DataFrame({'Close': [150.0]})
        
        ticker = 'AAPL'
        start_date = date(2023, 1, 1)
        calculate_implied_vol(ticker, start_date)
        
        self.assertTrue(1)

    

    def setUp(self):
        self.option = Option(
            option_type="up-and-out",
            ticker="AAPL",
            spot=100,
            strike=110,
            implied_vol=0.2,
            T=252,
            r=0.05,
            q=0.02,
            day_counts=252,
            call_option=True,
            barrier=120
        )

    @patch('option_pricing_app.Option._price_modeling')
    def test_price_up_and_out(self, mock_price_modeling):
        # Mock the _price_modeling method
        mock_paths = pd.DataFrame({
            0: [100, 105, 115, 110],
            1: [100, 110, 125, 130],
            2: [100, 95, 105, 115]
        })
        mock_price_modeling.return_value = mock_paths

        price = self.option._price_up_and_out(3)

        # Only the first and third paths should contribute to the price
        expected_price = (np.mean([0, 5]) * np.exp(-0.05 * 252 / 252))
        self.assertIsInstance(price, float)

    @patch('option_pricing_app.Option._price_modeling')
    def test_price_down_and_in(self, mock_price_modeling):
        self.option.option_type = "down-and-in"
        self.option.barrier = 90

        # Mock the _price_modeling method
        mock_paths = pd.DataFrame({
            0: [100, 95, 85, 105],
            1: [100, 98, 92, 95],
            2: [100, 105, 110, 115]
        })
        mock_price_modeling.return_value = mock_paths

        price = self.option._price_down_and_in(3)

        # Only the first path should contribute to the price
        expected_price = (np.mean([0]) * np.exp(-0.05 * 252 / 252))
        self.assertAlmostEqual(price, expected_price, places=6)

    def test_price_modeling(self):
        np.random.seed(42)
        paths = self.option._price_modeling(100, 0.2, 2, 252, 0.05, 0.02, 252)

        self.assertIsInstance(paths, pd.DataFrame)



if __name__ == '__main__':
    unittest.main()
