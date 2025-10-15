import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prep_data(category_cols: list, data: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns, scale numeric columns, and clean data.

    This function mirrors the preprocessing in `churn_genai.py` but avoids
    side-effects and external I/O so it is easier to test.
    """
    data_encoded = pd.get_dummies(data, columns=category_cols)

    # Handle TotalCharges potentially being a space-filled string
    if 'TotalCharges' in data_encoded.columns:
        data_encoded['TotalCharges'] = data_encoded['TotalCharges'].replace(' ', np.nan)
        data_encoded['TotalCharges'] = pd.to_numeric(data_encoded['TotalCharges'], errors='coerce')

    drop_cols = [c for c in ['Unnamed: 0', 'customerID'] if c in data_encoded.columns]
    if drop_cols:
        data_encoded = data_encoded.drop(drop_cols, axis=1)

    data_encoded = data_encoded.dropna()

    # Scale selected numeric columns if they exist
    scaler = MinMaxScaler()
    num_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in data_encoded.columns]
    if num_cols:
        data_encoded[num_cols] = scaler.fit_transform(data_encoded[num_cols])

    return data_encoded


def split_data(df: pd.DataFrame, test_size: float = 0.3, val_fraction_of_test: float = 0.5, random_state: int = 42):
    """Split data into train, validation, and test sets.

    Expects `Churn_encoded` column and drops `Churn`/`Churn_encoded` from X.
    Returns X_train, X_val, X_test, y_train, y_val, y_test
    """
    if 'Churn_encoded' not in df.columns:
        raise ValueError('Dataframe must contain Churn_encoded column')

    y = df['Churn_encoded'].values
    X = df.drop(columns=[c for c in ['Churn', 'Churn_encoded'] if c in df.columns], axis=1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_fraction_of_test, random_state=random_state, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test


def churn_roi_summary(customer_ltv, retention_cost, churn_rate, total_customers, precision, recall):
    """Compute a simple ROI summary for retention campaigns.

    Returns a dict of computed statistics.
    """
    actual_churners = int(total_customers * churn_rate)
    tp = int(recall * actual_churners)
    fn = actual_churners - tp
    # estimate false positives from precision: precision = tp / (tp + fp)
    fp = int(tp * (1 / precision - 1)) if precision > 0 else 0
    flagged = tp + fp

    campaign_cost = flagged * retention_cost
    revenue_protected = tp * customer_ltv
    revenue_lost = fn * customer_ltv
    net_savings = revenue_protected - campaign_cost

    return {
        "Flagged Customers": flagged,
        "Revenue Protected ($)": revenue_protected,
        "Campaign Cost ($)": campaign_cost,
        "Revenue Lost ($)": revenue_lost,
        "Net Savings ($)": net_savings,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3)
    }


def optimal_threshold_for_roi_pr(y_true, y_scores, total_customers, churn_rate, customer_ltv, retention_cost,
                                 model_version='model', min_precision=0.8):
    """Find the threshold that maximizes net savings using precision-recall curve.

    This is a lightweight, testable variant inspired by the notebook.
    Returns a dict with best threshold and stats (or None if no threshold meets min_precision).
    """
    from sklearn.metrics import precision_recall_curve

    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
    # thresholds length is len(prec)-1; append 1.0 to align
    thresholds = np.append(thresholds, 1.0)

    best_savings = -np.inf
    best = None

    actual_churners = int(total_customers * churn_rate)
    actual_non_churners = total_customers - actual_churners

    for p, r, t in zip(prec, rec, thresholds):
        if p < min_precision:
            continue

        true_positives = int(r * actual_churners)
        false_negatives = actual_churners - true_positives
        # derive false positives from precision
        false_positives = int(true_positives * (1 / p - 1)) if p > 0 else 0
        flagged_customers = true_positives + false_positives

        campaign_cost = flagged_customers * retention_cost
        revenue_protected = true_positives * customer_ltv
        net_savings = revenue_protected - campaign_cost

        if net_savings > best_savings:
            best_savings = net_savings
            best = {
                "model_version": model_version,
                "threshold": float(t),
                "true_positives": int(true_positives),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                "flagged_customers": int(flagged_customers),
                "campaign_cost": int(campaign_cost),
                "revenue_protected": int(revenue_protected),
                "net_savings": int(net_savings),
                "precision": float(p),
                "recall": float(r)
            }

    if best is None:
        return None
    import pandas as pd
    return pd.DataFrame(best, index=[0])
