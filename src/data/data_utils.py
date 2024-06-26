import os
from typing import Callable, Dict


def filter_data_dict(
    data: Dict[str, Dict], filter_fn: Callable[[Dict], bool]
) -> Dict[str, Dict]:
    return {key: value for key, value in data.items() if filter_fn(value)}


def slice_data_dict(data: Dict[str, Dict], start: int, end: int) -> Dict[str, Dict]:
    return {key: value for key, value in list(data.items())[start:end]}
