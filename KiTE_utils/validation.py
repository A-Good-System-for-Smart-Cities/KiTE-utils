import pandas as pd
from KiTE_utils import no_none_arg

@no_none_arg
def check_attributes(X, e, iterations=1000, n_jobs=1):
    """
    Check whether the input attributes are in proper format. If not exit with an Error message.
    """
    if X.shape[0] != e.shape[0]:
        raise ValueError(
            f"Incompatible dimension for X and e matrices. X and e should have the same feature dimension: X.shape[0] = {X.shape[0]} while e.shape[0] = {e.shape[0]}."
        )
    if not (isinstance(iterations, int) and iterations >= 2):
        raise ValueError(
            f"iterations has incorrect type or less than 2. iterations: {iterations}"
        )
    if not (isinstance(n_jobs, int) and n_jobs >= 1):
        raise ValueError(f"n_jobs is incorrect type or less than 1. n_jobs: {n_jobs}")
