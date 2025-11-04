"""
Prediction Validation Utility
Detects when models are returning default/fallback values instead of real predictions
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_predictions(predictions_df: pd.DataFrame, price_targets_df: pd.DataFrame) -> Dict:
    """
    Validate predictions to detect if they're real or defaults

    Args:
        predictions_df: DataFrame with predictions
        price_targets_df: DataFrame with price targets

    Returns:
        Dictionary with validation results
    """

    validation_results = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'using_defaults': False,
        'affected_stocks': [],
        'confidence': 'high'
    }

    if predictions_df.empty and price_targets_df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("No predictions generated")
        validation_results['confidence'] = 'none'
        return validation_results

    # Check price targets for default 5.0% pattern
    if not price_targets_df.empty:
        default_count = 0
        total_count = len(price_targets_df)

        for _, row in price_targets_df.iterrows():
            # Check for telltale signs of defaults
            expected_return = row.get('expected_return', row.get('percentage_change', row.get('expected_change', 0)))
            confidence = row.get('confidence', 0)
            direction = row.get('direction', '')

            # Exact 5.0% return, 50.0% confidence, NEUTRAL direction = default
            if (abs(expected_return - 5.0) < 0.01 and
                abs(confidence - 50.0) < 0.01 and
                direction == 'NEUTRAL'):
                default_count += 1
                validation_results['affected_stocks'].append(row.get('ticker', 'Unknown'))

        if default_count > 0:
            validation_results['using_defaults'] = True
            default_percentage = (default_count / total_count) * 100

            if default_count == total_count:
                # ALL stocks using defaults - critical issue
                validation_results['is_valid'] = False
                validation_results['confidence'] = 'none'
                validation_results['issues'].append(
                    f"ALL {total_count} stocks showing default values (5.0% / 50.0%)"
                )
                validation_results['issues'].append(
                    "Models failed to generate real predictions"
                )
            elif default_percentage > 50:
                # More than half using defaults
                validation_results['confidence'] = 'low'
                validation_results['warnings'].append(
                    f"{default_count}/{total_count} stocks ({default_percentage:.0f}%) using default values"
                )
            else:
                # Some using defaults
                validation_results['confidence'] = 'medium'
                validation_results['warnings'].append(
                    f"{default_count} stocks using fallback predictions"
                )

    # Check predictions for variance
    if not predictions_df.empty and 'confidence' in predictions_df.columns:
        confidences = predictions_df['confidence'].values

        # If all confidences are exactly the same, likely defaults
        if len(set(confidences)) == 1 and confidences[0] == 50.0:
            validation_results['warnings'].append(
                "All stocks have identical 50% confidence - may indicate model issues"
            )
            validation_results['confidence'] = 'low'

        # Check for low confidence across the board
        avg_confidence = confidences.mean()
        if avg_confidence < 55:
            validation_results['warnings'].append(
                f"Low average confidence ({avg_confidence:.1f}%) - models may need retraining"
            )
            validation_results['confidence'] = 'low'

    return validation_results


def display_validation_warnings(validation_results: Dict):
    """Display validation warnings in Streamlit UI"""

    if not validation_results['is_valid']:
        st.error("🚨 **CRITICAL: Predictions Failed**")

        st.markdown("""
        ### Why This Happened:
        """)

        for issue in validation_results['issues']:
            st.error(f"❌ {issue}")

        st.markdown("""
        ### 🔧 How to Fix:

        **Most Common Cause:** Models were trained with different features than current data.

        **Solution Steps:**
        1. **Delete the model cache:**
           ```bash
           # Windows
           rmdir /s model_cache

           # Mac/Linux
           rm -rf model_cache
           ```

        2. **Run the analysis again** - models will retrain automatically

        3. **Wait for training to complete** (may take 2-5 minutes)

        **Alternative:** Select stocks with at least **1 year of trading history**
        """)

        if validation_results['affected_stocks']:
            with st.expander("🔍 Affected Stocks"):
                for stock in validation_results['affected_stocks']:
                    st.caption(f"• {stock}")

        return False

    elif validation_results['using_defaults']:
        st.warning("⚠️ **Some Predictions Using Default Values**")

        for warning in validation_results['warnings']:
            st.warning(f"⚠️ {warning}")

        if validation_results['confidence'] == 'low':
            st.info("""
            **💡 Recommendation:**
            - Delete `model_cache` folder and re-run analysis
            - Models will retrain with fresh data
            - This usually resolves the issue
            """)

        return True

    elif validation_results['warnings']:
        for warning in validation_results['warnings']:
            st.info(f"ℹ️ {warning}")

        return True

    else:
        st.success("✅ **All predictions validated successfully**")
        st.caption(f"Confidence Level: {validation_results['confidence'].upper()}")
        return True


def get_cache_clear_instructions() -> str:
    """Get platform-specific cache clear instructions"""
    import platform

    os_type = platform.system()

    if os_type == "Windows":
        return """
        **Windows:**
        ```bash
        # Open Command Prompt or PowerShell in project directory
        rmdir /s /q model_cache
        rmdir /s /q feature_cache_v2
        ```
        """
    else:
        return """
        **Mac/Linux:**
        ```bash
        # Open Terminal in project directory
        rm -rf model_cache
        rm -rf feature_cache_v2
        ```
        """


def diagnose_prediction_failure(models: Dict, featured_data: Dict, predictions_df: pd.DataFrame) -> List[str]:
    """
    Diagnose why predictions might be failing

    Returns:
        List of diagnostic messages
    """

    diagnostics = []

    # Check if models exist
    if not models or len(models) == 0:
        diagnostics.append("❌ No models were trained")
        diagnostics.append("   → Check if stock data has sufficient history (need 200+ days)")
        return diagnostics

    # Check featured data
    if not featured_data or len(featured_data) == 0:
        diagnostics.append("❌ No feature data available")
        diagnostics.append("   → Feature engineering may have failed")
        return diagnostics

    # Check feature count consistency
    feature_counts = {}
    for ticker, df in featured_data.items():
        if not df.empty:
            exclude_cols = ['target', 'future_return', 'date', 'Date', 'ticker']
            features = [col for col in df.columns if col not in exclude_cols and not col.startswith('Target_')]
            feature_counts[ticker] = len(features)

    if feature_counts:
        unique_counts = set(feature_counts.values())
        if len(unique_counts) > 1:
            diagnostics.append(f"⚠️ Inconsistent feature counts across stocks: {unique_counts}")
            diagnostics.append("   → Some stocks may have different data available")

        # Check if feature count is reasonable
        avg_features = sum(feature_counts.values()) / len(feature_counts)
        if avg_features < 20:
            diagnostics.append(f"⚠️ Low feature count ({avg_features:.0f} features)")
            diagnostics.append("   → May need more historical data for better features")

    # Check model count
    total_models = sum(len(ticker_models) for ticker_models in models.values())
    expected_models = len(featured_data) * 3  # Assuming 3 model types

    if total_models < expected_models * 0.5:
        diagnostics.append(f"⚠️ Only {total_models}/{expected_models} models trained successfully")
        diagnostics.append("   → Training may have failed for some stocks/model types")

    # Check predictions
    if predictions_df.empty:
        diagnostics.append("❌ Prediction dataframe is empty")
        diagnostics.append("   → Ensemble prediction generation failed")
    else:
        if len(predictions_df) < len(models):
            diagnostics.append(f"⚠️ Predictions for only {len(predictions_df)}/{len(models)} stocks")
            diagnostics.append("   → Some predictions failed to generate")

    if not diagnostics:
        diagnostics.append("✅ No obvious issues detected")
        diagnostics.append("   → Models and data appear normal")

    return diagnostics


if __name__ == "__main__":
    print("Prediction Validator - Ready")

    # Test with mock data
    test_df = pd.DataFrame({
        'ticker': ['STOCK1', 'STOCK2', 'STOCK3'],
        'expected_return': [5.0, 5.0, 7.5],
        'confidence': [50.0, 50.0, 65.0],
        'direction': ['NEUTRAL', 'NEUTRAL', 'BULLISH']
    })

    results = validate_predictions(pd.DataFrame(), test_df)
    print(f"Validation Results: {results}")
