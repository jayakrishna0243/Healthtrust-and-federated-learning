import base64
import time
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from backend.blockchain import BlockchainClient


def train_client_2(X: np.ndarray, y: np.ndarray, bc: BlockchainClient) -> Dict[str, Any]:
    start = time.perf_counter()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=7,
    )
    model.fit(X_train, y_train)

    preds = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)

    model_bytes = model.get_booster().save_raw()
    model_b64 = base64.b64encode(model_bytes).decode("utf-8")
    tx_hash = bc.store_weights(model_b64, model_id="client2")

    end = time.perf_counter()

    return {
        "client": "client2",
        "accuracy": acc,
        "time": end - start,
        "tx_hash": tx_hash,
    }
