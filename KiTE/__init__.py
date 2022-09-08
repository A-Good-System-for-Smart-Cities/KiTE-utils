import decorator
import numpy as np

none_arg_msg = "A given argument was None."

@decorator.decorator
def no_none_arg(f, *args, **kwargs):
    is_none = [_ is None for _ in args if not isinstance(_, np.ndarray)]
    if len(is_none) == 0 or True in is_none:
        raise ValueError(none_arg_msg)
    return f(*args, **kwargs)


__docformat__ = "restructuredtext"
__doc__ = """
KiTE contains utilities to validate and calidrate supervised machine learning models.

Main Features
-------------
Here are the major utilities provided by the package:
- metrics
- calibrate
- plots

"""
