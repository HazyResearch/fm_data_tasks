"""Prompt utils."""
import logging

import pandas as pd

from fm_data_tasks.utils import constants
from fm_data_tasks.utils.data_utils import sample_train_data

logger = logging.getLogger(__name__)


def get_manual_prompt(data_dir: str, example: pd.Series) -> str:
    """Get manual prompt for data name."""
    if data_dir not in constants.DATA2TASK.keys():
        raise ValueError(f"{data_dir} not recognized for prompts")
    subkey_attr = constants.DATA2EXAMPLE_SUBKEY_ATTR[data_dir]
    if subkey_attr is None:
        if not isinstance(constants.PREFIXES[data_dir], str):
            print(data_dir)
            raise ValueError(f"Prefix was not a string for {data_dir}")
        return constants.PREFIXES[data_dir]
    else:
        if not isinstance(constants.PREFIXES[data_dir], dict):
            raise ValueError(
                f"Prefix was not a dict with {subkey_attr} subkeys for {data_dir}"
            )
        return constants.PREFIXES[data_dir][str(example[subkey_attr])]


def get_random_prompt(train_data: pd.DataFrame, num_examples: int = 10) -> str:
    """Get random examples for prompt from trian data."""
    prefix_exs_rows = sample_train_data(train_data, num_examples)
    serialized_prefixes = [
        (txt.strip() + " " + label.strip())
        for txt, label in zip(prefix_exs_rows["text"], prefix_exs_rows["label_str"])
    ]
    prefix_exs = "\n\n".join(serialized_prefixes) + "\n"
    return prefix_exs
