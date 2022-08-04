import decorator
import pandas as pd
import numpy as np

none_arg_msg = "A given argument was None."

@decorator.decorator
def no_none_arg(f,*args,**kwargs):
    is_none = [_ is None for _ in args if not isinstance(_, np.ndarray)]
    if len(is_none) == 0 or True in is_none:
        raise ValueError(none_arg_msg)
    return f(*args,**kwargs)
