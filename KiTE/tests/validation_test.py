from KiTE import none_arg_msg
from KiTE.validation import check_attributes
import numpy as np
import pytest
import re


class Test_check_attributes():
    def test_None_inputs(self):
        with pytest.raises(ValueError, match=none_arg_msg):
            check_attributes(None, None)
        with pytest.raises(ValueError, match=none_arg_msg):
            check_attributes([], None)

    def test_incompatible_dims(self):
        X = np.array([[1, 2, 3], [11, 21, 31]])
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Incompatible dimension for X and e matrices. X and e should have the same feature dimension: X.shape[0] = {X.shape[0]} while e.shape[0] = {X.T.shape[0]}."
            ),
        ):
            check_attributes(X, X.T)

    def test_invalid_iterations(self):
        X = np.array([[1, 2, 3], [11, 21, 31]])
        str_input = "cake"
        neg_input = -1
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

    def test_invalid_n_jobs(self):
        X = np.array([[1, 2, 3], [11, 21, 31]])
        str_input = "cake"
        neg_input = -1
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
