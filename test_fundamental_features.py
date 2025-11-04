"""
Comprehensive Test Script for Advanced Fundamental Analysis Features
Tests all new modules and integration points
"""

import sys
import pandas as pd
from datetime import datetime

print("=" * 70)
print("AI STOCK ADVISOR PRO - FUNDAMENTAL ANALYSIS TEST SUITE")
print("=" * 70)
print(f"Test started at: {datetime.now()}")
print()

# Test 1: Import all new modules
print("Test 1: Testing module imports...")
try:
    from utils.advanced_data_sources import (
        AdvancedDataAggregator,
        EconomicIndicators,
        FundamentalData,
        InsiderTradingData,
        InstitutionalData,
        get_fundamentals,
        get_market_overview
    )
    print("  [OK] advanced_data_sources imported successfully")
except ImportError as e:
    print(f"  [FAIL] Failed to import advanced_data_sources: {e}")
    sys.exit(1)

try:
    from utils.feature_engineer import (
        add_fundamental_features,
        engineer_features_with_fundamentals
    )
    print("  [OK] feature_engineer updates imported successfully")
except ImportError as e:
    print(f"  [FAIL] Failed to import feature_engineer: {e}")
    sys.exit(1)

try:
    from config import ADVANCED_DATA_CONFIG
    print("  [OK] ADVANCED_DATA_CONFIG imported successfully")
except ImportError as e:
    print(f"  [FAIL] Failed to import config: {e}")
    sys.exit(1)

print()

# Test 2: Test Economic Indicators
print("Test 2: Testing Economic Indicators...")
try:
    econ = EconomicIndicators()

    # Test treasury rates
    treasury_rates = econ.get_treasury_rates()
    print(f"  [OK] Treasury rates fetched: {len(treasury_rates)} rates")
    print(f"       10Y Treasury: {treasury_rates.get('10_year', 'N/A')}%")

    # Test market indices
    market_indices = econ.get_market_indices()
    print(f"  [OK] Market indices fetched: {len(market_indices)} indices")

    # Test market regime
    market_regime = econ.calculate_market_regime(market_indices)
    print(f"  [OK] Market regime: {market_regime}")

except Exception as e:
    print(f"  [FAIL] Economic indicators test failed: {e}")

print()

# Test 3: Test Fundamental Data
print("Test 3: Testing Fundamental Data Fetching...")
try:
    fund = FundamentalData()

    # Test with a well-known stock
    test_ticker = "RELIANCE.NS"
    print(f"  Testing with ticker: {test_ticker}")

    company_info = fund.get_company_info(test_ticker)

    if company_info:
        print(f"  [OK] Company info fetched")
        print(f"       Sector: {company_info.get('sector', 'N/A')}")
        print(f"       P/E Ratio: {company_info.get('pe_ratio', 'N/A')}")
        print(f"       ROE: {company_info.get('roe', 'N/A')}")

        # Test fundamental score
        fund_score = fund.calculate_fundamental_score(company_info)
        print(f"  [OK] Fundamental score calculated: {fund_score}/100")
    else:
        print(f"  [WARN] No data returned for {test_ticker}")

except Exception as e:
    print(f"  [FAIL] Fundamental data test failed: {e}")

print()

# Test 4: Test Insider Trading Data
print("Test 4: Testing Insider Trading Data...")
try:
    insider = InsiderTradingData()

    test_ticker = "RELIANCE.NS"
    insider_sentiment = insider.calculate_insider_sentiment(test_ticker)

    print(f"  [OK] Insider sentiment calculated")
    print(f"       Sentiment: {insider_sentiment.get('sentiment', 'N/A')}")
    print(f"       Score: {insider_sentiment.get('score', 0)}")
    print(f"       Transactions: {insider_sentiment.get('transaction_count', 0)}")

except Exception as e:
    print(f"  [FAIL] Insider trading test failed: {e}")

print()

# Test 5: Test Institutional Data
print("Test 5: Testing Institutional Data...")
try:
    institutional = InstitutionalData()

    test_ticker = "RELIANCE.NS"
    inst_score = institutional.calculate_institutional_ownership_score(test_ticker)

    print(f"  [OK] Institutional score calculated: {inst_score}/10")

except Exception as e:
    print(f"  [FAIL] Institutional data test failed: {e}")

print()

# Test 6: Test Complete Data Aggregation
print("Test 6: Testing Complete Data Aggregation...")
try:
    aggregator = AdvancedDataAggregator()

    test_ticker = "RELIANCE.NS"
    complete_data = aggregator.get_complete_data(test_ticker)

    if complete_data:
        print(f"  [OK] Complete data aggregated for {test_ticker}")
        print(f"       Fundamental Score: {complete_data.get('fundamental_score', 'N/A')}")
        print(f"       Insider Activity: {complete_data.get('insider_activity', {}).get('sentiment', 'N/A')}")
        print(f"       Institutional Score: {complete_data.get('institutional', {}).get('ownership_score', 'N/A')}")
    else:
        print(f"  [WARN] No complete data returned")

except Exception as e:
    print(f"  [FAIL] Data aggregation test failed: {e}")

print()

# Test 7: Test Batch Data Fetching
print("Test 7: Testing Batch Data Fetching...")
try:
    aggregator = AdvancedDataAggregator()

    test_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    print(f"  Testing with {len(test_tickers)} tickers...")

    batch_data = aggregator.get_batch_data(test_tickers, max_workers=2)

    successful = sum(1 for data in batch_data.values() if data is not None)
    print(f"  [OK] Batch data fetched: {successful}/{len(test_tickers)} successful")

    for ticker, data in batch_data.items():
        if data:
            score = data.get('fundamental_score', 'N/A')
            print(f"       {ticker}: Score = {score}")

except Exception as e:
    print(f"  [FAIL] Batch fetching test failed: {e}")

print()

# Test 8: Test Market Context
print("Test 8: Testing Market Context...")
try:
    market_context = get_market_overview()

    print(f"  [OK] Market context fetched")
    print(f"       Market Regime: {market_context.get('market_regime', 'N/A')}")
    print(f"       Treasury Rates: {len(market_context.get('treasury_rates', {}))} rates")
    print(f"       Market Indices: {len(market_context.get('market_indices', {}))} indices")

except Exception as e:
    print(f"  [FAIL] Market context test failed: {e}")

print()

# Test 9: Test Fundamental Features Integration
print("Test 9: Testing Fundamental Features Integration...")
try:
    # Create sample price data
    sample_data = pd.DataFrame({
        'Close': [2500, 2510, 2520, 2530, 2540],
        'Open': [2490, 2500, 2510, 2520, 2530],
        'High': [2520, 2530, 2540, 2550, 2560],
        'Low': [2480, 2490, 2500, 2510, 2520],
        'Volume': [1000000, 1100000, 1050000, 1200000, 1150000]
    })

    test_ticker = "RELIANCE.NS"

    # Add fundamental features
    df_with_fundamentals = add_fundamental_features(sample_data, test_ticker)

    # Check if fundamental columns were added
    expected_cols = ['pe_ratio', 'roe', 'fundamental_score', 'insider_score']
    cols_added = all(col in df_with_fundamentals.columns for col in expected_cols)

    if cols_added:
        print(f"  [OK] Fundamental features added to DataFrame")
        print(f"       Total columns: {len(df_with_fundamentals.columns)}")
        print(f"       Sample features: {', '.join(expected_cols)}")
    else:
        missing = [col for col in expected_cols if col not in df_with_fundamentals.columns]
        print(f"  [WARN] Some features missing: {missing}")

except Exception as e:
    print(f"  [FAIL] Features integration test failed: {e}")

print()

# Test 10: Test Configuration
print("Test 10: Testing Configuration...")
try:
    print(f"  [OK] ADVANCED_DATA_CONFIG loaded")
    print(f"       Fundamentals enabled: {ADVANCED_DATA_CONFIG['enable_fundamentals']}")
    print(f"       Economic indicators enabled: {ADVANCED_DATA_CONFIG['enable_economic_indicators']}")
    print(f"       Insider trading enabled: {ADVANCED_DATA_CONFIG['enable_insider_trading']}")
    print(f"       Cache duration: {ADVANCED_DATA_CONFIG['cache_duration_hours']} hours")
    print(f"       Parallel fetching: {ADVANCED_DATA_CONFIG['parallel_fetching']}")

    # Test weights
    total_weight = (
        ADVANCED_DATA_CONFIG['weight_fundamental_score'] +
        ADVANCED_DATA_CONFIG['weight_technical_score'] +
        ADVANCED_DATA_CONFIG['weight_sentiment_score']
    )

    if abs(total_weight - 1.0) < 0.01:
        print(f"  [OK] Scoring weights sum to 1.0")
    else:
        print(f"  [WARN] Scoring weights sum to {total_weight} (should be 1.0)")

except Exception as e:
    print(f"  [FAIL] Configuration test failed: {e}")

print()

# Test 11: Test Error Handling
print("Test 11: Testing Error Handling...")
try:
    # Test with invalid ticker
    invalid_ticker = "INVALID_TICKER_XYZ"
    aggregator = AdvancedDataAggregator()

    print(f"  Testing with invalid ticker: {invalid_ticker}")
    invalid_data = aggregator.get_complete_data(invalid_ticker)

    # Should return data with defaults, not crash
    if invalid_data:
        print(f"  [OK] Graceful handling of invalid ticker")
        print(f"       Returned default values instead of crashing")
    else:
        print(f"  [WARN] No data returned for invalid ticker")

except Exception as e:
    print(f"  [FAIL] Error handling test failed: {e}")

print()

# Test Summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Test completed at: {datetime.now()}")
print()
print("All tests completed! Check results above for any failures.")
print()
print("Next Steps:")
print("  1. Review any [FAIL] or [WARN] messages above")
print("  2. Check FUNDAMENTAL_ANALYSIS_INTEGRATION_GUIDE.md for integration steps")
print("  3. Add the new features to your app.py using the guide")
print("  4. Test with real stock data in your Streamlit app")
print()
print("=" * 70)
