import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on sys.path so "backend" package can be imported.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.preprocessing import load_and_preprocess_csv, split_clients
from backend.encryption import create_ckks_context, encrypt_matrix, decrypt_matrix
from backend.blockchain import BlockchainClient
from backend.client1 import train_client_1
from backend.client2 import train_client_2
from backend.global_model import evaluate_global_model


def run_pipeline(csv_path: str, provider_url: str, contract_json: str, contract_address: str) -> Dict[str, Any]:
    X, y, _ = load_and_preprocess_csv(csv_path)

    # Encrypt then decrypt (homomorphic storage simulation)
    ctx = create_ckks_context()
    enc_rows = encrypt_matrix(ctx, X)
    X_decrypted = decrypt_matrix(enc_rows)

    (X1, y1), (X2, y2) = split_clients(X_decrypted, y)

    bc = BlockchainClient(provider_url, contract_json, contract_address)

    c1 = train_client_1(X1, y1, bc)
    c2 = train_client_2(X2, y2, bc)

    global_metrics = evaluate_global_model(X_decrypted, y, bc)

    return {
        "client1": c1,
        "client2": c2,
        "global": global_metrics,
    }


if __name__ == "__main__":
    # Defaults assume standard Truffle build output and Ganache on localhost
    csv_path = os.path.join("..", "dataset", "ehr.csv")
    provider_url = "http://127.0.0.1:7545"
    contract_json = os.path.join("..", "blockchain", "build", "contracts", "FLContract.json")
    contract_address = os.environ.get("FL_CONTRACT_ADDRESS", "").strip()
    if not contract_address:
        contract_address = "0x209c325d5e79Bb0508cbc03570601B743EBC786D"

    if not contract_address:
        raise ValueError("Set FL_CONTRACT_ADDRESS environment variable with deployed contract address.")

    results = run_pipeline(csv_path, provider_url, contract_json, contract_address)
    print("Client1:", results["client1"])
    print("Client2:", results["client2"])
    print("Global:", results["global"])
