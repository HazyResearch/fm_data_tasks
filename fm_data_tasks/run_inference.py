"""Run inference."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import fm_data_tasks.utils.data_utils as data_utils
import fm_data_tasks.utils.prompt_utils as prompt_utils
from fm_data_tasks.utils import constants
from fm_data_tasks.utils.client import Client
from fm_data_tasks.utils.utils import compute_metrics, setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Generate args."""
    parser = argparse.ArgumentParser(description="Simple calculator")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Which data directory to run.",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory.", default="outputs"
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        help="File to cache input/output results.",
        default="openai_cache.sqlite",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite sqlite cache of input/output results.",
    )
    parser.add_argument("--k", type=int, help="Number examples in prompt", default=1)
    parser.add_argument(
        "--sample_method",
        type=str,
        help="Example generation method",
        default="random",
        choices=["random", "manual", "validation_clusters"],
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        default=None,
        help="Path to validation data error feather file. I.e., the result of \
            running `run_inference` of validation data. Used with \
            validation_clusters method.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--class_balanced",
        help="Class balance training data. Good for classification tasks \
             with random prompts.",
        action="store_true",
    )
    parser.add_argument(
        "--sep_tok",
        type=str,
        help="Separate for attr: val pairs in row. Default is '.'.",
        default=".",
    )
    parser.add_argument(
        "--nan_tok",
        type=str,
        help="Token to represent nan entries. Default is 'nan'.",
        default="nan",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Which OpenAI model to use.",
        default="text-davinci-002",
        choices=[
            "text-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
        ],
    )
    parser.add_argument(
        "--num_run",
        type=int,
        help="Number examples to run through model.",
        default=-1,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number trials to run. Results will be averaged with variance reported.",
        default=1,
    )
    parser.add_argument(
        "--num_print",
        type=int,
        help="Number example prompts to print.",
        default=10,
    )
    parser.add_argument(
        "--add_task_instruction",
        help="Add task instruction to the prompt before examples.",
        action="store_true",
    )
    parser.add_argument("--do_test", help="Run on test file.", action="store_true")
    parser.add_argument(
        "--dry_run", help="Dry run. Do not actually ping model.", action="store_true"
    )

    # Open AI args
    parser.add_argument("--temperature", type=float, help="Temperature.", default=0.0)
    parser.add_argument(
        "--max_tokens", type=int, help="Max tokens to generate.", default=3
    )

    args = parser.parse_args()
    return args


def main():
    """Run main method."""
    args = parse_args()
    if args.num_trials < 1:
        raise ValueError("num_trials must be greater than 0.")
    if args.sample_method == "validation_clusters" and args.validation_path is None:
        raise ValueError(
            "validation_path must be provided when \
             sample_method is validation_clusters."
        )
    # Get absolute path
    args.data_dir = str(Path(args.data_dir).resolve())
    setup_logger(args.output_dir)
    logger.info(json.dumps(vars(args), indent=4))

    # Will set seed for pandas
    np.random.seed(args.seed)

    test_file = "test" if args.do_test else "validation"

    # Read pandas DF datasets
    pd_data_files = data_utils.read_data(
        data_dir=args.data_dir,
        class_balanced=args.class_balanced,
        add_prefix=False,
        max_train_samples=-1,
        sep_tok=args.sep_tok,
        nan_tok=args.nan_tok,
    )
    if test_file not in pd_data_files:
        raise ValueError(f"Need {test_file} data")

    train_data = pd_data_files["train"]
    test_data = pd_data_files[test_file]
    task = constants.DATA2TASK[args.data_dir]
    task_instruction = constants.DATA2INSTRUCT[args.data_dir]

    if args.num_run == -1:
        args.num_run = test_data.shape[0]
    args.num_run = min(args.num_run, test_data.shape[0])

    logger.info(f"Train shape is {train_data.shape[0]}")
    logger.info(f"Test shape is {test_data.shape[0]}")
    logger.info(f"Running {args.num_run} examples for {args.num_trials} trials.")

    # Setup client
    client = Client(args.cache_file)
    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}
    # Cache prefix for validation_clusters
    saved_prefix = None
    for trial_num in range(args.num_trials):
        np.random.seed(args.seed + trial_num)
        entries = []
        prefixes = []
        preds = []
        model_inputs = []
        for _, row in test_data.iterrows():
            serialized_r = row["text"]
            entries.append(serialized_r)

            if args.sample_method == "manual":
                prefix_exs = prompt_utils.get_manual_prompt(args.data_dir, row)
            elif args.sample_method == "validation_clusters":
                if saved_prefix is None:
                    logger.info("Generating validation cluster prompt.")
                    saved_prefix = prompt_utils.get_validation_prompt(
                        args.validation_path,
                        num_examples=args.k,
                        task=task,
                    )
                prefix_exs = saved_prefix
            else:
                prefix_exs = prompt_utils.get_random_prompt(
                    pd_data_files["train"], num_examples=args.k
                )
            if args.add_task_instruction:
                prefixes.append(task_instruction + " " + prefix_exs + "\n")
            else:
                prefixes.append(prefix_exs + "\n")

        # Send to model for predictions
        gt = test_data["label_str"]
        num_print = args.num_print
        for prefix, query in tqdm(
            zip(prefixes, entries),
            desc=f"Querying trial {trial_num}",
            total=args.num_run,
        ):
            if len(model_inputs) >= args.num_run:
                break
            string = (prefix + query).strip()
            model_inputs.append(string)

            if num_print > 0:
                logger.info(string)
                logger.info("**********************")
            if not args.dry_run:
                response = client.query(
                    engine=args.model_name,
                    prompt=string,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                    overwrite_cache=args.overwrite_cache,
                )
                vals = [
                    response["choices"][i]["text"]
                    for i in range(len(response["choices"]))
                ]
                # Take tokens before first \n
                vals = " ".join([val.strip().split("\n")[0] for val in vals]).strip()
                if num_print > 0:
                    logger.info(f"====> {vals} <====")
                preds.append(vals)
            else:
                preds.append("")
            num_print -= 1

        # Save trial predictions
        save_data = test_data.iloc[: args.num_run].copy(deep=True).reset_index()
        gt = gt[: args.num_run]
        save_data["preds"] = preds
        save_data["model_inputs"] = model_inputs

        prec, rec, acc, f1 = compute_metrics(preds, gt, task)

        logger.info(
            f"Metrics Trial {trial_num}\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)

        output_file = (
            Path(args.output_dir)
            / f"{Path(args.data_dir).stem}"
            / f"{test_file}"
            / f"{args.model_name}"
            f"_{args.k}k"
            f"_{int(args.class_balanced)}cb"
            f"_{args.sample_method}"
            f"_{args.num_run}run"
            f"_{int(args.dry_run)}dry" / f"trial_{trial_num}.feather"
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saved to {output_file}")

        save_data.to_feather(output_file)

    for k, values in list(trial_metrics.items()):
        trial_metrics[f"{k}_avg"] = np.average(values)
        trial_metrics[f"{k}_std"] = np.std(values)

    output_metrics = output_file.parent / "metrics.json"
    json.dump(trial_metrics, open(output_metrics, "w"))

    logger.info(f"Final Metrics {json.dumps(trial_metrics, indent=4)}")
    logger.info(f"Metrics dumped to {output_metrics}")


if __name__ == "__main__":
    main()
