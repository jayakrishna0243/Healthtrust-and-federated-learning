import tenseal as ts
import numpy as np
from typing import Tuple, List


def create_ckks_context() -> ts.Context:
    # CKKS parameters for moderate security and performance
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2**40
    return ctx


def encrypt_matrix(ctx: ts.Context, X: np.ndarray) -> List[ts.CKKSVector]:
    # Encrypt each row as a vector
    return [ts.ckks_vector(ctx, row.tolist()) for row in X]


def decrypt_matrix(enc_rows: List[ts.CKKSVector]) -> np.ndarray:
    return np.array([row.decrypt() for row in enc_rows])
