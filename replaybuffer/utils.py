from typing import Iterable, Tuple


def remove_nones(*arrays: Iterable) -> Tuple[Iterable]:
    """
    Take inputted arrays that may contain None values, and
    return copies without Nones.

    Returns:
        tuple[Iterable]: New arrays with only non-None values
    """
    return tuple([[i for i in array if i is not None] for array in arrays])
