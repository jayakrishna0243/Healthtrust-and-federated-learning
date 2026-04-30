import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler


TARGET_CANDIDATES = ["class", "classification", "ckd", "target", "label"]


def _infer_target_column(df: pd.DataFrame) -> str:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in TARGET_CANDIDATES:
        if cand in lower_cols:
            return lower_cols[cand]
    return df.columns[-1]


def _clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Replace common missing markers with NaN
    df = df.replace(["?", "NA", "N/A", "nan", "NaN", ""], np.nan)

    for col in df.columns:
        if df[col].dtype == "object":
            # Fill categorical with mode
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
            df[col] = df[col].fillna(fill_val)
        else:
            # Fill numeric with median
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def load_and_preprocess_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    return preprocess_df(df)


def preprocess_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    df = df.copy()

    target_col = _infer_target_column(df)
    df = _clean_missing_values(df)

    # Separate features and target
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Encode categorical features
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in X_raw.columns:
        if X_raw[col].dtype == "object":
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
            label_encoders[col] = le
        else:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce").fillna(0)

    # Encode target
    y = y_raw
    if y_raw.dtype == "object":
        y_le = LabelEncoder()
        y = y_le.fit_transform(y_raw.astype(str))
        label_encoders["_target"] = y_le
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw.values)

    metadata = {
        "target_col": target_col,
        "feature_names": list(X_raw.columns),
        "label_encoders": label_encoders,
        "scaler": scaler,
    }
    return X, y, metadata


def split_clients(X: np.ndarray, y: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # Simple even split for two clients
    mid = len(X) // 2
    return (X[:mid], y[:mid]), (X[mid:], y[mid:])
