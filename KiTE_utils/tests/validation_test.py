from KiTE_utils import none_arg_msg
from KiTE_utils.validation import check_attributes
import numpy as np
import pytest
import re


@pytest.mark.validators
def test_check_attributes():
    X = np.array([[1, 2, 3], [11, 21, 31]])
    str_input = "cake"
    neg_input = -1
    # None Test
    with pytest.raises(ValueError, match=none_arg_msg):
        check_attributes(None, None)
    with pytest.raises(ValueError, match=none_arg_msg):
        check_attributes([], None)
    # Incompatible Dims Test
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Incompatible dimension for X and e matrices. X and e should have the same feature dimension: X.shape[0] = {X.shape[0]} while e.shape[0] = {X.T.shape[0]}."
        ),
    ):
        check_attributes(X, X.T)
    # Invalid iterations
    with pytest.raises(
        ValueError,
        match=f"iterations has incorrect type or less than 2. iterations: {str_input}",
    ):
        check_attributes(X, X, str_input)
    with pytest.raises(
        ValueError,
        match=f"iterations has incorrect type or less than 2. iterations: {neg_input}",
    ):
        check_attributes(X, X, neg_input)
    # Invalid n_jobs
    with pytest.raises(
        ValueError,
        match=f"n_jobs is incorrect type or less than 1. n_jobs: {str_input}",
    ):
        check_attributes(X, X, n_jobs=str_input)
    with pytest.raises(
        ValueError,
        match=f"n_jobs is incorrect type or less than 1. n_jobs: {neg_input}",
    ):
        check_attributes(X, X, n_jobs=neg_input)
