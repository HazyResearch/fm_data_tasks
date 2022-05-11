"""Run imputation."""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import openai
from tqdm.auto import tqdm

import fm_data_tasks.utils.data_utils as data_utils
import fm_data_tasks.utils.prompt_utils as prompt_utils
from fm_data_tasks.utils.utils import compute_metrics, setup_logger

logger = logging.getLogger(__name__)

openai.api_key = os.environ.get("OPENAI_API_KEY")


def parse_args():
    """Generate args."""
    parser = argparse.ArgumentParser(description="Simple calculator")

    parser.add_argument(
        "--data_dir", type=str, help="Which data directory to run", required=True
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default="output"
    )

    parser.add_argument("--k", type=int, help="Number examples in prompt", default=1)
    parser.add_argument(
        "--sample_method",
        type=str,
        help="Example generation method",
        default="random",
        choices=["random", "manual"],
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--class_balanced",
        help="Class balance training data. Good for classification tasks.",
        action="store_true",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="text-davinci-002",
        choices=[
            "text-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
        ],
    )
    parser.add_argument(
        "--num_run", type=int, help="Number examples to run through model", default=-1
    )
    parser.add_argument(
        "--num_print", type=int, help="Number example prompts to print", default=5
    )
    parser.add_argument(
        "--add_task_instruction", help="Add task instruction", action="store_true"
    )
    parser.add_argument("--do_test", help="Run on test file", action="store_true")
    parser.add_argument("--dry_run", help="Dry run", action="store_true")

    # Open AI args
    parser.add_argument("--temperature", type=float, help="Temperature", default=0.0)
    parser.add_argument(
        "--max_tokens", type=float, help="Max tokens to generate", default=3
    )

    args = parser.parse_args()
    return args


def main():
    """Run main method."""
    args = parse_args()
    # Get absolute path
    args.data_dir = str(Path(args.data_dir).resolve())
    setup_logger(args.output_dir)
    logger.info(json.dumps(vars(args), indent=4))
    # Will set seed for pandas
    np.random.seed(args.seed)

    test_file = "test" if args.do_test else "validation"

    pd_data_files = data_utils.read_data(
        data_dir=args.data_dir,
        class_balanced=args.class_balanced,
        add_prefix=False,
        max_train_samples=-1,
    )
    if test_file not in pd_data_files:
        raise ValueError(f"Need {test_file} data")

    train_data = pd_data_files["train"]
    test_data = pd_data_files[test_file]
    task = data_utils.DATA2TASK[args.data_dir]
    task_prefix = data_utils.INSTRUCTION_DICT[task]

    if args.num_run == -1:
        args.num_run = test_data.shape[0]
    args.num_run = min(args.num_run, test_data.shape[0])
    logger.info(f"Running {args.num_run} examples")

    logger.info(f"Train shape is {train_data.shape[0]}")
    logger.info(f"Test shape is {test_data.shape[0]}")

    entries = []
    prefixes = []
    preds = []
    model_inputs = []
    for _, row in test_data.iterrows():
        serialized_r = row["text"]
        entries.append(serialized_r)

        if args.sample_method == "manual":
            prefix_exs = prompt_utils.get_manual_prompt[Path(args.train_data).stem]
        else:
            prefix_exs_rows = data_utils.sample_train_data(train_data, args.k)
            serialized_prefixes = [
                txt + label
                for txt, label in zip(
                    prefix_exs_rows["text"], prefix_exs_rows["label_str"]
                )
            ]
            prefix_exs = "\n\n".join(serialized_prefixes)
        if args.add_task_instruction:
            prefixes.append(task_prefix + " " + prefix_exs + "\n\n")
        else:
            prefixes.append(prefix_exs + "\n\n")

    # Send to model for predictions
    gt = test_data["label_str"]
    num_print = args.num_print
    for prefix, query in tqdm(
        zip(prefixes, entries), desc="Querying", total=len(entries)
    ):
        if len(model_inputs) >= args.num_run:
            break
        string = data_utils.strip_value(prefix + query).lstrip()

        if num_print > 0:
            logger.info(string)
            logger.info("**********************")
        model_inputs.append(string)
        if not args.dry_run:
            response = openai.Completion.create(
                engine=args.model_name,
                prompt=string,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
            )
            vals = [
                response["choices"][i]["text"] for i in range(len(response["choices"]))
            ]
            vals = " ".join([val.strip().split("\n")[0] for val in vals]).strip()
            if num_print > 0:
                logger.info("====>", vals, "<====")
            preds.append(vals)
        else:
            preds.append("")
        num_print -= 1

    # Save predictions
    save_data = test_data.iloc[: args.num_run].copy(deep=True)
    gt = gt[: args.num_run]
    save_data["preds"] = preds
    save_data["model_inputs"] = model_inputs

    prec, rec, acc, f1 = compute_metrics(preds, gt)

    logger.info(
        f"Final Metrics\nPrec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
    )

    output_file = (
        Path(args.output_dir) / f"{Path(args.data_dir).stem}"
        f"{test_file}"
        f"_{args.model_name}"
        f"_{args.k}k"
        f"_{int(args.class_balanced)}cb"
        f"_{args.sample_method}"
        f"_{args.num_run}run"
        f"_{int(args.dry_run)}dry.feather"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved to {output_file}")

    save_data.to_feather(output_file)


if __name__ == "__main__":
    main()
