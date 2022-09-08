from KiTE import no_none_arg


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


@no_none_arg
def check_credible_vector_args(Y, Yr_up, Yr_down, alpha):
    """
    Check arguments of construct_credible_error_vector()
    """
    if Y.flatten().shape[0] != Yr_up.flatten().shape[0]:
        raise ValueError(
            f"Incompatible dimension for Y and Yr_up matrices. Y and Yr_up should have the same feature dimension: Y.shape[0] == {Y.shape[0]} while Yr.shape[0] == {Yr_up.shape[0]}."
        )
    if Y.flatten().shape[0] != Yr_down.flatten().shape[0]:
        raise ValueError(
            f"Incompatible dimension for Y and Yr matrices. Y and Yr should have the same feature dimension. Y.shape[0] == {Y.shape[0]} while Yr_down.shape[0] == {Yr_down.shape[0]}."
        )
    if alpha < 0 or alpha > 1:
        raise ValueError(
            f"Incompatible value for alpha. alpha should be a real value between 0 and 1: alpha == {alpha}"
        )
