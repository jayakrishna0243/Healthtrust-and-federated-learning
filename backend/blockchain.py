import json
import os
from typing import List
from web3 import Web3


class BlockchainClient:
    def __init__(self, provider_url: str, contract_json_path: str, contract_address: str, timeout_seconds: int = 120):
        # Increase HTTP timeout to avoid read timeouts on large calls
        self.w3 = Web3(Web3.HTTPProvider(provider_url, request_kwargs={"timeout": timeout_seconds}))
        if not self.w3.is_connected():
            raise ConnectionError("Web3 provider not connected. Is Ganache running?")

        with open(contract_json_path, "r", encoding="utf-8") as f:
            contract_json = json.load(f)

        abi = contract_json["abi"]
        self.contract = self.w3.eth.contract(address=self.w3.to_checksum_address(contract_address), abi=abi)
        self.account = self.w3.eth.accounts[0]

    def store_weights(self, weights_str: str, model_id: str, chunk_size: int = 512) -> str:
        """
        Store large strings in chunks to avoid block gas limit issues.
        Each chunk is labeled as: model_id:part/total:chunk
        """
        total = (len(weights_str) + chunk_size - 1) // chunk_size
        last_tx = None
        for i in range(total):
            chunk = weights_str[i * chunk_size : (i + 1) * chunk_size]
            payload = f"{model_id}:{i+1}/{total}:{chunk}"
            tx_hash = self.contract.functions.storeWeights(payload).transact(
                {"from": self.account, "gas": 1_000_000}
            )
            last_tx = tx_hash
        receipt = self.w3.eth.wait_for_transaction_receipt(last_tx, timeout=180)
        return receipt.transactionHash.hex()

    def get_weights(self) -> List[str]:
        count = self.contract.functions.getWeightCount().call()
        results: List[str] = []
        for i in range(count):
            results.append(self.contract.functions.getWeight(i).call())
        return results
