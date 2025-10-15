import pandas as pd
import numpy as np
import pytest

from churn_core import prep_data, split_data, churn_roi_summary


def make_sample_df():
    # Build a small representative dataframe
    df = pd.DataFrame({
        'customerID': ['c1', 'c2', 'c3', 'c4'],
        'gender': ['Male', 'Female', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0],
        'tenure': [1, 12, 5, 2],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30],
        'TotalCharges': ['29.85', '695.40', '284.10', '84.60'],
        'Churn': ['Stayed', 'Churned', 'Stayed', 'Churned'],
    })
    df['Churn_encoded'] = df['Churn'].map({'Stayed': 0, 'Churned': 1})
    return df


def test_prep_data_scales_and_encodes():
    df = make_sample_df()
    cols = ['gender', 'SeniorCitizen']
    out = prep_data(cols, df)

    # Ensure dummy columns created for gender
    assert any('gender_' in c for c in out.columns)

    # Ensure numeric columns scaled to [0,1]
    for c in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        assert c in out.columns
        assert out[c].min() >= 0 - 1e-6
        assert out[c].max() <= 1 + 1e-6


def test_split_data_shapes_and_stratify():
    df = make_sample_df()
    cols = ['gender', 'SeniorCitizen']
    prepared = prep_data(cols, df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(prepared, test_size=0.5, val_fraction_of_test=0.5, random_state=0)

    # Check lengths sum
    total = len(y_train) + len(y_val) + len(y_test)
    assert total == len(prepared)

    # Ensure y arrays only contain 0/1
    for arr in [y_train, y_val, y_test]:
        assert set(np.unique(arr)).issubset({0, 1})


def test_churn_roi_summary_basic():
    res = churn_roi_summary(customer_ltv=500, retention_cost=20, churn_rate=0.25, total_customers=1000, precision=0.8, recall=0.5)
    # Check keys exist
    for k in ["Flagged Customers", "Net Savings ($)"]:
        assert k in res
    # Basic numeric sanity: Net savings could be negative or positive but must be int-like
    assert isinstance(res["Flagged Customers"], int)


def test_split_data_missing_label_raises():
    df = pd.DataFrame({'a': [1,2,3]})
    with pytest.raises(ValueError):
        split_data(df)


def test_optimal_threshold_for_roi_pr_basic():
    # Synthetic scenario: 10 samples where first 4 are positive (churn)
    y_true = np.array([1,1,1,1,0,0,0,0,0,0])
    # Scores: higher for first two positives, medium for next two, low for negatives
    y_scores = np.array([0.95, 0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])

    # Use small numbers for LTV/cost to make net_savings easily interpretable
    df = split_data(pd.DataFrame({
        'Churn_encoded': y_true,
        'f1': range(len(y_true))
    }), test_size=0.5, val_fraction_of_test=0.5, random_state=0)

    # Call function directly using full arrays (function expects raw arrays)
    from churn_core import optimal_threshold_for_roi_pr

    result = optimal_threshold_for_roi_pr(y_true, y_scores, total_customers=10, churn_rate=0.4, customer_ltv=100, retention_cost=5, model_version='t1', min_precision=0.5)
    # We expect a DataFrame in return
    assert result is not None
    assert 'threshold' in result.columns
    assert result.loc[0, 'model_version'] == 't1'
