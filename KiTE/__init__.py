__docformat__ = "restructuredtext"
__doc__ = """
KiTE contains utilities to validate and calidrate supervised machine learning models.

Main Features
-------------
Here are the major utilities provided by the package:
- Metrics to test if local bias is statistically significant within the given model
- Calibration utilities to reduce local bias
- Diffusion Map utilities to transform euclidean distance metrics into a diffusion space


Example Notebooks
-----------------
We created [Example Notebooks](https://github.com/A-Good-System-for-Smart-Cities/KiTE-utils/tree/main/notebooks) to showcase basic examples and applications of this library.

"""

import decorator
import numpy as np

none_arg_msg = "A given argument was None."


@decorator.decorator
def no_none_arg(f, *args, **kwargs):
    is_none = [_ is None for _ in args if not isinstance(_, np.ndarray)]
    if len(is_none) == 0 or True in is_none:
        raise ValueError(none_arg_msg)
    return f(*args, **kwargs)
