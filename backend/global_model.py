import base64
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from backend.blockchain import BlockchainClient


def _load_booster_from_b64(b64_str: str) -> xgb.Booster:
    model_bytes = base64.b64decode(b64_str.encode("utf-8"))
    booster = xgb.Booster()
    booster.load_model(bytearray(model_bytes))
    return booster


def _parse_chunks(entries: List[str]) -> List[str]:
    """
    Reconstruct model strings from chunked entries stored on-chain.
    Each entry format: model_id:part/total:chunk
    """
    completed: List[str] = []
    buffers: Dict[str, Dict[int, str]] = {}
    totals: Dict[str, int] = {}

    for entry in entries:
        try:
            model_id, part_info, chunk = entry.split(":", 2)
            part_str, total_str = part_info.split("/", 1)
            part = int(part_str)
            total = int(total_str)
        except ValueError:
            # Skip malformed entries
            continue

        if model_id not in buffers:
            buffers[model_id] = {}
        buffers[model_id][part] = chunk
        totals[model_id] = total

        if len(buffers[model_id]) == total:
            ordered = "".join(buffers[model_id][i] for i in range(1, total + 1))
            completed.append(ordered)
            # reset for potential future models with same id
            buffers.pop(model_id, None)
            totals.pop(model_id, None)

    return completed


def evaluate_global_model(X: np.ndarray, y: np.ndarray, bc: BlockchainClient) -> Dict[str, Any]:
    start = time.perf_counter()

    weight_list = bc.get_weights()
    if not weight_list:
        raise RuntimeError("No weights found on blockchain. Train clients first.")

    model_strings = _parse_chunks(weight_list)
    if not model_strings:
        raise RuntimeError("No complete model weights reconstructed from blockchain chunks.")

    boosters = [_load_booster_from_b64(w) for w in model_strings]
    dmat = xgb.DMatrix(X)

    # Aggregate by averaging predicted probabilities from each client model
    probs = np.zeros(len(X))
    for booster in boosters:
        probs += booster.predict(dmat)
    probs /= len(boosters)

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)

    end = time.perf_counter()

    return {
        "global_accuracy": acc,
        "global_time": end - start,
        "models_aggregated": len(boosters),
    }
