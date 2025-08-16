import unittest
from src.data import download_monthly_prices

class TestData(unittest.TestCase):
    def test_download(self):
        prices = download_monthly_prices(['RELIANCE.NS'], '2021-01-01', '2021-02-01')
        self.assertFalse(prices.empty)

if __name__ == '__main__':
    unittest.main()
