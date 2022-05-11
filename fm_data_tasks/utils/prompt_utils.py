"""Prompt utils."""
# ADD PROMPTS HERE ... IDEALLY IN A DICTIONARY OR
# A SMALL CODE SCRIPT HTAT LOADS THEM FROM data/prompts...?
from pathlib import Path

from fm_data_tasks.utils.data_utils import DATA2TASK


def get_manual_prompt(data_name: str):
    """Get manual prompt for data name."""
    if data_name not in set([Path(pth).name for pth in DATA2TASK.keys()]):
        raise ValueError(f"{data_name} not recognized for prompts")
    # TODO FILL IN!!
    pass
