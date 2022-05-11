"""Data utils."""
import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_PATH = os.environ.get("DATASET_PATH", Path("data/datasets").resolve())

DATA2TASK = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Beer": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": "entity_matching",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": "entity_matching",
    f"{DATASET_PATH}/data_imputation/Buy": "data_imputation",
    f"{DATASET_PATH}/data_imputation/Restaurant": "data_imputation",
    f"{DATASET_PATH}/error_detection/Hospital": "error_detection",
}

COLS_TO_DROP = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": [],
    f"{DATASET_PATH}/entity_matching/structured/Beer": ["Style", "ABV"],
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": [],
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": [],
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": [],
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": ["CopyRight"],
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": [
        "category",
        "price",
        "brand",
    ],
    f"{DATASET_PATH}/data_imputation/Buy": [],
    f"{DATASET_PATH}/data_imputation/Restaurant": [],
    f"{DATASET_PATH}/error_detection/Hospital": [],
}

IMPUTE_COLS = {
    f"{DATASET_PATH}/data_imputation/Buy": "manufacturer",
    f"{DATASET_PATH}/data_imputation/Restaurant": "city",
}

MATCH_PROD_NAME = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "Product",
    f"{DATASET_PATH}/entity_matching/structured/Beer": "Product",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "Product",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "Product",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": "Product",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": "Song",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": "Product",
}

DATA2INSTRUCT = {
    f"{DATASET_PATH}/entity_matching/structured/Amazon-Google": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/Beer": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-ACM": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/DBLP-GoogleScholar": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/Fodors-Zagats": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/iTunes-Amazon": "Are Song A and Song B the same? Yes or No?",
    f"{DATASET_PATH}/entity_matching/structured/Walmart-Amazon": "Are Product A and Product B the same? Yes or No?",
    f"{DATASET_PATH}/data_imputation/Buy": "Who is the manufacturer?",
    f"{DATASET_PATH}/data_imputation/Restaurant": "What is the city?",
    f"{DATASET_PATH}/error_detection/Hospital": "error_detection",
}

INSTRUCTION_DICT = {
    "entity_matching": "Are Product A and Product B the same? Yes or No?",
    "data_imputation": "What is the missing value?",
    "error_detection": "Is there an error? Yes or No?",
}


def sample_train_data(train: pd.DataFrame, n_rows: int):
    """
    Sample train data.

    Used when random sampling points for prompt.
    """
    res = train.sample(n_rows)
    return res


def serialize_row(
    row: pd.core.series.Series, column_map: Dict[str, str], sep_tok: str
) -> str:
    """Turn structured row into string."""
    res = []
    for c_og, c_map in column_map.items():
        res.append(f"{c_map}: {row[c_og]}".strip())
    if len(sep_tok) > 0 and sep_tok != ".":
        sep_tok = f" {sep_tok}"
    return f"{sep_tok} ".join(res)


def serialize_match_pair(
    row: pd.core.series.Series,
    column_mapA: Dict[str, str],
    column_mapB: Dict[str, str],
    add_prefix: bool,
    task: str,
    sep_tok: str,
    prod_name: str,
) -> str:
    """Turn structured pair of entities into string for matching."""
    res = (
        f"{prod_name} A is {serialize_row(row, column_mapA, sep_tok)}."
        f" {prod_name} B is {serialize_row(row, column_mapB, sep_tok)}. "
        f"Are A and B the same? "
    )
    if add_prefix:
        res = f"{INSTRUCTION_DICT[task]} {res}"
    return res


def serialize_imputation(
    row: pd.core.series.Series,
    column_map: Dict[str, str],
    impute_col: str,
    add_prefix: bool,
    task: str,
    sep_tok: str,
) -> str:
    """Turn single entity into string for imputation."""
    assert impute_col not in column_map, f"{impute_col} cannot be in column map"
    # Rename to avoid passing white spaced sep token to serialize_row
    sep_tok_ws = sep_tok
    if len(sep_tok) > 0 and sep_tok != ".":
        sep_tok_ws = f" {sep_tok}"
    res = f"{serialize_row(row, column_map, sep_tok)}{sep_tok_ws} {impute_col}? "
    if add_prefix:
        res = f"{INSTRUCTION_DICT[task]} {res}"
    return res


def serialize_error_detection(
    row: pd.core.series.Series,
    add_prefix: bool,
    task: str,
    sep_tok: str,
) -> str:
    """Turn single cell into string for error detection."""
    column_map = {row["col_name"]: row["col_name"]}
    res = f"Is there a spelling error in {serialize_row(row, column_map, sep_tok)}? "
    if add_prefix:
        res = f"{INSTRUCTION_DICT[task]} {res}"
    return res


def read_blocked_pairs(
    split_path: str,
    tableA: pd.DataFrame,
    tableB: pd.DataFrame,
    add_prefix: bool,
    task: str,
    sep_tok: str,
    prod_name: str,
) -> pd.DataFrame:
    """Read in pre-blocked pairs with T/F match labels."""
    column_mapA = {f"{c}_A": c for c in tableA.columns if c != "id"}
    column_mapB = {f"{c}_B": c for c in tableB.columns if c != "id"}

    labels = pd.read_csv(split_path)

    mergedA = pd.merge(labels, tableA, right_on="id", left_on="ltable_id")
    merged = pd.merge(
        mergedA, tableB, right_on="id", left_on="rtable_id", suffixes=("_A", "_B")
    )

    merged["text"] = merged.apply(
        lambda row: serialize_match_pair(
            row, column_mapA, column_mapB, add_prefix, task, sep_tok, prod_name
        ),
        axis=1,
    )
    merged["label_str"] = merged.apply(
        lambda row: "Yes\n" if row["label"] == 1 else "No\n", axis=1
    )
    return merged


def read_imputation_single(
    split_path: str, impute_col: str, add_prefix: bool, task: str, sep_tok: str
) -> pd.DataFrame:
    """Read in table and create label impute col."""
    table = pd.read_csv(split_path)
    column_map = {c: c for c in table.columns if c != "id" and c != impute_col}

    table["text"] = table.apply(
        lambda row: serialize_imputation(
            row, column_map, impute_col, add_prefix, task, sep_tok
        ),
        axis=1,
    )
    table["label_str"] = table[impute_col].apply(lambda x: f"{x}\n")
    return table


def read_error_detection_single(
    split_path: str, table: pd.DataFrame, add_prefix: bool, task: str, sep_tok: str
) -> pd.DataFrame:
    """Read in table and create label impute col."""
    # row_id, col_name, is_clean
    labels = pd.read_csv(split_path)
    merged = pd.merge(labels, table, left_on="row_id", right_index=True)

    merged["text"] = merged.apply(
        lambda row: serialize_error_detection(row, add_prefix, task, sep_tok), axis=1
    )
    merged["label_str"] = merged.apply(
        lambda row: "No\n" if row["is_clean"] == 1 else "Yes\n", axis=1
    )
    return merged


def read_data(
    data_dir: str,
    class_balanced: bool = False,
    add_prefix: bool = False,
    max_train_samples: float = -1,
    sep_tok: str = "",
):
    """Read in data where each directory is unique for a task."""
    data_files_sep = {"test": {}, "train": {}, "validation": {}}
    logger.info(f"Processing {data_dir}")
    if data_dir not in DATA2TASK:
        raise ValueError(
            f"data_dir not one of {DATA2TASK.keys()}. Make sure to set DATASET_PATH."
        )
    task = DATA2TASK[data_dir]
    data_dir_p = Path(data_dir)
    if task == "entity_matching":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        tableA_file = data_dir_p / "tableA.csv"
        tableB_file = data_dir_p / "tableB.csv"

        tableA = pd.read_csv(tableA_file)
        tableB = pd.read_csv(tableB_file)
        for c in COLS_TO_DROP[data_dir]:
            tableA.drop(c, axis=1, inplace=True)
            tableB.drop(c, axis=1, inplace=True)

        label_col = "label"
        read_data_func = partial(
            read_blocked_pairs,
            tableA=tableA,
            tableB=tableB,
            add_prefix=add_prefix,
            task=task,
            sep_tok=sep_tok,
            prod_name=MATCH_PROD_NAME[data_dir],
        )
    elif task == "data_imputation":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        label_col = IMPUTE_COLS[data_dir]
        read_data_func = partial(
            read_imputation_single,
            impute_col=label_col,
            add_prefix=add_prefix,
            task=task,
            sep_tok=sep_tok,
        )
    elif task == "error_detection":
        train_file = data_dir_p / "train.csv"
        valid_file = data_dir_p / "valid.csv"
        test_file = data_dir_p / "test.csv"
        table_file = data_dir_p / "table.csv"
        table = pd.read_csv(table_file)
        for c in COLS_TO_DROP[data_dir]:
            table.drop(c, axis=1, inplace=True)
        label_col = "is_clean"
        read_data_func = partial(
            read_error_detection_single,
            table=table,
            add_prefix=add_prefix,
            task=task,
            sep_tok=sep_tok,
        )
    else:
        raise ValueError(f"Task {task} not recognized.")

    data_files_sep["train"] = read_data_func(train_file)
    # Don't class balance on open ended classificiation tasks
    if class_balanced and task != "data_imputation":
        # Class balance sample the train data
        label_cnts = data_files_sep["train"].groupby(label_col).count()
        sample_per_class = label_cnts.min()["text"]
        logger.info(f"Class balanced: train sample per class: {sample_per_class}")
        data_files_sep["train"] = (
            data_files_sep["train"]
            .groupby(label_col, group_keys=False)
            .apply(lambda x: x.sample(sample_per_class, random_state=42))
        )
    # Shuffle train data
    data_files_sep["train"] = (
        data_files_sep["train"].sample(frac=1, random_state=42).reset_index(drop=True)
    )
    if max_train_samples > 0:
        orig_train_len = len(data_files_sep["train"])
        if max_train_samples > 1.0:
            raise ValueError("max_train_samples must be between 0 and 1")
        max_examples = int(max_train_samples * orig_train_len)
        data_files_sep["train"] = data_files_sep["train"].iloc[:max_examples]
        logger.info(
            f"Length of {data_dir} train is "
            f"{data_files_sep['train'].shape[0]} from {orig_train_len}"
        )

    # Read validation
    data_files_sep["validation"] = read_data_func(valid_file)
    # Read test
    data_files_sep["test"] = read_data_func(test_file)
    return data_files_sep
