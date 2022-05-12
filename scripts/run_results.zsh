#!/usr/bin/zsh

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --do_test
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --do_test


# Data Imputation
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test
# Needed to add task instruction for this one
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test --add_task_instruction
