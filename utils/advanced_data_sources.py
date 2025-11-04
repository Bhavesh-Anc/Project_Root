"""
Advanced Data Sources Module
Provides company fundamentals, economic indicators, insider trading, and institutional data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from functools import lru_cache
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EconomicIndicators:
    """Fetch and process economic indicators"""

    def __init__(self, cache_duration_hours: int = 24):
        self.cache_duration_hours = cache_duration_hours
        self.cache = {}

    def get_treasury_rates(self) -> Dict[str, float]:
        """Get US Treasury rates (risk-free rates)"""
        try:
            # Using yfinance tickers for treasury yields
            treasury_tickers = {
                '3_month': '^IRX',    # 13 Week Treasury Bill
                '2_year': '^2YR',     # Placeholder (not always available)
                '10_year': '^TNX',    # 10 Year Treasury
                '30_year': '^TYX'     # 30 Year Treasury
            }

            rates = {}
            for period, ticker in treasury_tickers.items():
                try:
                    data = yf.Ticker(ticker).history(period='5d')
                    if not data.empty:
                        # Treasury yields are in percentage points
                        rates[period] = float(data['Close'].iloc[-1])
                except:
                    rates[period] = None

            # Fallback: use fixed rates if not available
            if not any(rates.values()):
                rates = {
                    '3_month': 5.25,
                    '10_year': 4.50,
                    '30_year': 4.75
                }
                logger.warning("Using fallback treasury rates")

            return rates

        except Exception as e:
            logger.error(f"Error fetching treasury rates: {e}")
            return {'3_month': 5.25, '10_year': 4.50, '30_year': 4.75}

    def get_market_indices(self) -> Dict[str, Dict[str, float]]:
        """Get major market indices for market regime detection"""
        try:
            indices = {
                'SP500': '^GSPC',      # S&P 500
                'NIFTY50': '^NSEI',    # Nifty 50
                'VIX': '^VIX',         # Volatility Index
                'DXY': 'DX-Y.NYB',     # Dollar Index
                'GOLD': 'GC=F',        # Gold Futures
                'OIL': 'CL=F'          # Crude Oil
            }

            results = {}
            for name, ticker in indices.items():
                try:
                    data = yf.Ticker(ticker).history(period='1mo')
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        month_ago_price = float(data['Close'].iloc[0])
                        monthly_return = (current_price - month_ago_price) / month_ago_price * 100

                        # Calculate volatility
                        returns = data['Close'].pct_change().dropna()
                        volatility = float(returns.std() * np.sqrt(252) * 100)

                        results[name] = {
                            'current_price': current_price,
                            'monthly_return': monthly_return,
                            'volatility': volatility
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch {name}: {e}")

            return results

        except Exception as e:
            logger.error(f"Error fetching market indices: {e}")
            return {}

    def get_economic_calendar(self) -> Dict[str, Any]:
        """Get upcoming economic events (simplified version)"""
        # This would typically connect to an economic calendar API
        # For now, return a placeholder structure
        return {
            'next_fed_meeting': 'Check Federal Reserve website',
            'next_earnings_season': 'Quarterly',
            'major_events': []
        }

    def calculate_market_regime(self, market_data: Dict) -> str:
        """Determine current market regime"""
        try:
            if not market_data:
                return 'UNKNOWN'

            sp500_return = market_data.get('SP500', {}).get('monthly_return', 0)
            vix_level = market_data.get('VIX', {}).get('current_price', 20)

            # Simple regime classification
            if sp500_return > 2 and vix_level < 15:
                return 'BULL_LOW_VOL'
            elif sp500_return > 0 and vix_level < 20:
                return 'BULL_NORMAL_VOL'
            elif sp500_return < -2 and vix_level > 30:
                return 'BEAR_HIGH_VOL'
            elif sp500_return < 0 and vix_level > 20:
                return 'BEAR_NORMAL_VOL'
            else:
                return 'NEUTRAL'

        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return 'UNKNOWN'


class FundamentalData:
    """Fetch and process company fundamental data"""

    def __init__(self, cache_db: str = 'data/fundamentals_cache.db'):
        self.cache_db = cache_db
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamentals_cache (
                    ticker TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing fundamentals cache: {e}")

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive company information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract key fundamental data
            fundamentals = {
                # Valuation Metrics
                'market_cap': info.get('marketCap', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'ev_to_revenue': info.get('enterpriseToRevenue', None),
                'ev_to_ebitda': info.get('enterpriseToEbitda', None),

                # Profitability Metrics
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'gross_margin': info.get('grossMargins', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),

                # Financial Health
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'total_debt': info.get('totalDebt', None),
                'total_cash': info.get('totalCash', None),
                'free_cash_flow': info.get('freeCashflow', None),

                # Growth Metrics
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None),

                # Dividend Information
                'dividend_yield': info.get('dividendYield', None),
                'payout_ratio': info.get('payoutRatio', None),
                'dividend_rate': info.get('dividendRate', None),

                # Share Statistics
                'shares_outstanding': info.get('sharesOutstanding', None),
                'float_shares': info.get('floatShares', None),
                'shares_short': info.get('sharesShort', None),
                'short_ratio': info.get('shortRatio', None),
                'short_percent_float': info.get('shortPercentOfFloat', None),

                # Trading Information
                'beta': info.get('beta', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', None),

                # Company Information
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'employees': info.get('fullTimeEmployees', None),

                # Analyst Recommendations
                'recommendation': info.get('recommendationKey', 'none'),
                'target_price': info.get('targetMeanPrice', None),
                'num_analysts': info.get('numberOfAnalystOpinions', None),
            }

            # Calculate derived metrics
            fundamentals['debt_to_assets'] = self._calculate_debt_to_assets(fundamentals)
            fundamentals['cash_ratio'] = self._calculate_cash_ratio(fundamentals)
            fundamentals['working_capital'] = self._calculate_working_capital(info)

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {}

    def _calculate_debt_to_assets(self, fundamentals: Dict) -> Optional[float]:
        """Calculate debt to assets ratio"""
        try:
            if fundamentals.get('total_debt') and fundamentals.get('market_cap'):
                total_assets = fundamentals['total_debt'] + fundamentals['market_cap']
                return fundamentals['total_debt'] / total_assets
        except:
            pass
        return None

    def _calculate_cash_ratio(self, fundamentals: Dict) -> Optional[float]:
        """Calculate cash ratio"""
        try:
            if fundamentals.get('total_cash') and fundamentals.get('market_cap'):
                return fundamentals['total_cash'] / fundamentals['market_cap']
        except:
            pass
        return None

    def _calculate_working_capital(self, info: Dict) -> Optional[float]:
        """Calculate working capital"""
        try:
            current_assets = info.get('totalCurrentAssets', 0)
            current_liabilities = info.get('totalCurrentLiabilities', 0)
            if current_assets and current_liabilities:
                return current_assets - current_liabilities
        except:
            pass
        return None

    def get_financial_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get income statement, balance sheet, and cash flow"""
        try:
            stock = yf.Ticker(ticker)

            return {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_financials': stock.quarterly_financials
            }

        except Exception as e:
            logger.error(f"Error fetching financial statements for {ticker}: {e}")
            return {}

    def calculate_fundamental_score(self, fundamentals: Dict) -> float:
        """Calculate a composite fundamental score (0-100)"""
        try:
            score = 50  # Start with neutral score

            # Valuation (cheaper is better)
            if fundamentals.get('pe_ratio'):
                if fundamentals['pe_ratio'] < 15:
                    score += 10
                elif fundamentals['pe_ratio'] < 25:
                    score += 5
                elif fundamentals['pe_ratio'] > 40:
                    score -= 10

            # Profitability (higher is better)
            if fundamentals.get('roe'):
                if fundamentals['roe'] > 0.20:  # 20%+ ROE
                    score += 10
                elif fundamentals['roe'] > 0.15:
                    score += 5
                elif fundamentals['roe'] < 0:
                    score -= 15

            if fundamentals.get('profit_margin'):
                if fundamentals['profit_margin'] > 0.20:
                    score += 10
                elif fundamentals['profit_margin'] > 0.10:
                    score += 5
                elif fundamentals['profit_margin'] < 0:
                    score -= 10

            # Growth (higher is better)
            if fundamentals.get('revenue_growth'):
                if fundamentals['revenue_growth'] > 0.15:  # 15%+ growth
                    score += 10
                elif fundamentals['revenue_growth'] > 0.05:
                    score += 5
                elif fundamentals['revenue_growth'] < -0.05:
                    score -= 10

            # Financial Health (stronger is better)
            if fundamentals.get('current_ratio'):
                if fundamentals['current_ratio'] > 2:
                    score += 5
                elif fundamentals['current_ratio'] < 1:
                    score -= 10

            if fundamentals.get('debt_to_equity'):
                if fundamentals['debt_to_equity'] < 0.5:
                    score += 10
                elif fundamentals['debt_to_equity'] > 2:
                    score -= 10

            # Dividend (bonus for dividend payers)
            if fundamentals.get('dividend_yield') and fundamentals['dividend_yield'] > 0.02:
                score += 5

            # Analyst sentiment
            if fundamentals.get('recommendation') in ['strong_buy', 'buy']:
                score += 5
            elif fundamentals.get('recommendation') in ['sell', 'strong_sell']:
                score -= 5

            # Clamp score between 0 and 100
            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 50


class InsiderTradingData:
    """Fetch and analyze insider trading activity"""

    def get_insider_transactions(self, ticker: str) -> pd.DataFrame:
        """Get recent insider transactions"""
        try:
            stock = yf.Ticker(ticker)
            insider_transactions = stock.insider_transactions

            if insider_transactions is not None and not insider_transactions.empty:
                return insider_transactions
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching insider transactions for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_insider_sentiment(self, ticker: str, days: int = 90) -> Dict[str, Any]:
        """Calculate insider sentiment score"""
        try:
            transactions = self.get_insider_transactions(ticker)

            if transactions.empty:
                return {'sentiment': 'neutral', 'score': 0, 'transaction_count': 0}

            # Filter recent transactions
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_transactions = transactions[
                pd.to_datetime(transactions.index) > cutoff_date
            ] if hasattr(transactions.index, 'to_pydatetime') else transactions

            if recent_transactions.empty:
                return {'sentiment': 'neutral', 'score': 0, 'transaction_count': 0}

            # Calculate buy/sell ratio
            buys = recent_transactions[recent_transactions['Shares'] > 0]['Shares'].sum()
            sells = abs(recent_transactions[recent_transactions['Shares'] < 0]['Shares'].sum())

            total_transactions = len(recent_transactions)

            if sells == 0 and buys > 0:
                sentiment = 'bullish'
                score = 10
            elif buys > sells:
                ratio = buys / (sells + 1)
                sentiment = 'bullish' if ratio > 1.5 else 'neutral'
                score = min(10, ratio * 2)
            elif sells > buys:
                ratio = sells / (buys + 1)
                sentiment = 'bearish' if ratio > 1.5 else 'neutral'
                score = -min(10, ratio * 2)
            else:
                sentiment = 'neutral'
                score = 0

            return {
                'sentiment': sentiment,
                'score': score,
                'transaction_count': total_transactions,
                'buy_volume': buys,
                'sell_volume': sells
            }

        except Exception as e:
            logger.error(f"Error calculating insider sentiment for {ticker}: {e}")
            return {'sentiment': 'neutral', 'score': 0, 'transaction_count': 0}


class InstitutionalData:
    """Fetch institutional holdings and ownership data"""

    def get_institutional_holders(self, ticker: str) -> pd.DataFrame:
        """Get institutional holders"""
        try:
            stock = yf.Ticker(ticker)
            institutional_holders = stock.institutional_holders

            if institutional_holders is not None and not institutional_holders.empty:
                return institutional_holders
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching institutional holders for {ticker}: {e}")
            return pd.DataFrame()

    def get_major_holders(self, ticker: str) -> Dict[str, float]:
        """Get major holders breakdown"""
        try:
            stock = yf.Ticker(ticker)
            major_holders = stock.major_holders

            if major_holders is not None and not major_holders.empty:
                # Parse the major holders data
                holders_dict = {}
                for idx, row in major_holders.iterrows():
                    holders_dict[row[1]] = float(row[0].strip('%')) if isinstance(row[0], str) else row[0]
                return holders_dict
            else:
                return {}

        except Exception as e:
            logger.error(f"Error fetching major holders for {ticker}: {e}")
            return {}

    def calculate_institutional_ownership_score(self, ticker: str) -> float:
        """Calculate score based on institutional ownership (0-10)"""
        try:
            major_holders = self.get_major_holders(ticker)

            if not major_holders:
                return 5  # Neutral if no data

            # Higher institutional ownership generally positive (but not too high)
            institutional_pct = 0
            for key, value in major_holders.items():
                if 'institutions' in key.lower():
                    institutional_pct = value
                    break

            # Optimal range: 40-70% institutional ownership
            if 40 <= institutional_pct <= 70:
                score = 8
            elif 30 <= institutional_pct < 40 or 70 < institutional_pct <= 80:
                score = 6
            elif institutional_pct < 30:
                score = 4  # Low institutional interest
            else:  # > 80%
                score = 5  # Too concentrated

            return score

        except Exception as e:
            logger.error(f"Error calculating institutional ownership score for {ticker}: {e}")
            return 5


class AdvancedDataAggregator:
    """Main class to aggregate all advanced data sources"""

    def __init__(self):
        self.economic = EconomicIndicators()
        self.fundamentals = FundamentalData()
        self.insider = InsiderTradingData()
        self.institutional = InstitutionalData()

    def get_complete_data(self, ticker: str) -> Dict[str, Any]:
        """Get all advanced data for a single ticker"""
        logger.info(f"Fetching advanced data for {ticker}")

        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'fundamentals': {},
            'insider_activity': {},
            'institutional': {},
            'economic_context': {}
        }

        try:
            # Get fundamental data
            result['fundamentals'] = self.fundamentals.get_company_info(ticker)
            result['fundamental_score'] = self.fundamentals.calculate_fundamental_score(
                result['fundamentals']
            )

            # Get insider trading data
            result['insider_activity'] = self.insider.calculate_insider_sentiment(ticker)

            # Get institutional data
            result['institutional']['holders'] = self.institutional.get_institutional_holders(ticker)
            result['institutional']['major_holders'] = self.institutional.get_major_holders(ticker)
            result['institutional']['ownership_score'] = self.institutional.calculate_institutional_ownership_score(ticker)

        except Exception as e:
            logger.error(f"Error aggregating data for {ticker}: {e}")

        return result

    def get_batch_data(self, tickers: List[str], max_workers: int = 4) -> Dict[str, Dict]:
        """Get advanced data for multiple tickers in parallel"""
        logger.info(f"Fetching advanced data for {len(tickers)} tickers")

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.get_complete_data, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    results[ticker] = data
                    logger.info(f"✓ Completed {ticker}")
                except Exception as e:
                    logger.error(f"✗ Failed {ticker}: {e}")
                    results[ticker] = None

        return results

    def get_market_context(self) -> Dict[str, Any]:
        """Get overall market context"""
        logger.info("Fetching market context")

        context = {
            'treasury_rates': self.economic.get_treasury_rates(),
            'market_indices': self.economic.get_market_indices(),
        }

        context['market_regime'] = self.economic.calculate_market_regime(
            context['market_indices']
        )

        return context

    def create_fundamental_features(self, ticker_data: Dict) -> pd.DataFrame:
        """Convert fundamental data to features for ML models"""
        try:
            fundamentals = ticker_data.get('fundamentals', {})

            features = {
                # Valuation features
                'pe_ratio': fundamentals.get('pe_ratio'),
                'forward_pe': fundamentals.get('forward_pe'),
                'pb_ratio': fundamentals.get('price_to_book'),
                'ps_ratio': fundamentals.get('price_to_sales'),

                # Profitability features
                'profit_margin': fundamentals.get('profit_margin'),
                'operating_margin': fundamentals.get('operating_margin'),
                'roe': fundamentals.get('roe'),
                'roa': fundamentals.get('roa'),

                # Growth features
                'revenue_growth': fundamentals.get('revenue_growth'),
                'earnings_growth': fundamentals.get('earnings_growth'),

                # Financial health features
                'current_ratio': fundamentals.get('current_ratio'),
                'debt_to_equity': fundamentals.get('debt_to_equity'),

                # Other features
                'dividend_yield': fundamentals.get('dividend_yield', 0),
                'beta': fundamentals.get('beta'),
                'short_percent': fundamentals.get('short_percent_float', 0),

                # Composite scores
                'fundamental_score': ticker_data.get('fundamental_score', 50),
                'insider_score': ticker_data.get('insider_activity', {}).get('score', 0),
                'institutional_score': ticker_data.get('institutional', {}).get('ownership_score', 5),
            }

            # Create DataFrame
            df = pd.DataFrame([features])

            # Fill NaN with neutral values
            df = df.fillna({
                'pe_ratio': 20,
                'forward_pe': 20,
                'pb_ratio': 3,
                'ps_ratio': 2,
                'profit_margin': 0.10,
                'operating_margin': 0.10,
                'roe': 0.15,
                'roa': 0.05,
                'revenue_growth': 0.05,
                'earnings_growth': 0.05,
                'current_ratio': 1.5,
                'debt_to_equity': 1.0,
                'dividend_yield': 0,
                'beta': 1.0,
                'short_percent': 0,
                'fundamental_score': 50,
                'insider_score': 0,
                'institutional_score': 5
            })

            return df

        except Exception as e:
            logger.error(f"Error creating fundamental features: {e}")
            return pd.DataFrame()


# Convenience functions for easy integration

def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Quick function to get fundamentals for a single ticker"""
    aggregator = AdvancedDataAggregator()
    return aggregator.get_complete_data(ticker)


def get_fundamentals_batch(tickers: List[str]) -> Dict[str, Dict]:
    """Quick function to get fundamentals for multiple tickers"""
    aggregator = AdvancedDataAggregator()
    return aggregator.get_batch_data(tickers)


def get_market_overview() -> Dict[str, Any]:
    """Quick function to get market overview"""
    aggregator = AdvancedDataAggregator()
    return aggregator.get_market_context()


if __name__ == "__main__":
    # Test the module
    print("Testing Advanced Data Sources Module")
    print("=" * 50)

    # Test with a sample ticker
    test_ticker = "RELIANCE.NS"

    print(f"\n1. Testing fundamental data for {test_ticker}...")
    aggregator = AdvancedDataAggregator()
    data = aggregator.get_complete_data(test_ticker)

    print(f"\nFundamental Score: {data.get('fundamental_score', 'N/A')}")
    print(f"Insider Sentiment: {data.get('insider_activity', {}).get('sentiment', 'N/A')}")
    print(f"Institutional Score: {data.get('institutional', {}).get('ownership_score', 'N/A')}")

    print(f"\n2. Testing market context...")
    market_context = aggregator.get_market_context()
    print(f"Market Regime: {market_context.get('market_regime', 'N/A')}")
    print(f"10Y Treasury Rate: {market_context.get('treasury_rates', {}).get('10_year', 'N/A')}%")

    print("\n[OK] All tests completed!")
