#!/usr/bin/zsh

# Commands to run 200 random and manual examples for each dataset with metrics

# Entity Matching Fodors-Zagats
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --nan_tok "" --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --do_test --nan_tok "" --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --nan_tok "" --do_test


# Entity Matching Beer
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/entity_matching/structured/Beer --nan_tok "" --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/entity_matching/structured/Beer --nan_tok "" --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/entity_matching/structured/Beer --nan_tok "" --do_test


# Entity Matching iTunes-Amazon
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/entity_matching/structured/iTunes-Amazon --nan_tok "" --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/entity_matching/structured/iTunes-Amazon --nan_tok "" --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/entity_matching/structured/iTunes-Amazon --nan_tok "" --do_test


# Entity Matching Walmart-Amazon
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/entity_matching/structured/Walmart-Amazon --nan_tok "" --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/entity_matching/structured/Walmart-Amazon --nan_tok "" --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/entity_matching/structured/Walmart-Amazon --nan_tok "" --do_test


# Entity Matching Amazon-Google
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/entity_matching/structured/Amazon-Google --nan_tok "" --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/entity_matching/structured/Amazon-Google --nan_tok "" --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/entity_matching/structured/Amazon-Google --nan_tok "" --do_test


# Entity Matching DBLP-ACM
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/entity_matching/structured/DBLP-ACM --nan_tok "" --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/entity_matching/structured/DBLP-ACM --nan_tok "" --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/entity_matching/structured/DBLP-ACM --nan_tok "" --do_test


# Entity Matching DBLP-GoogleScholar
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar --nan_tok "" --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar --nan_tok "" --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar --nan_tok "" --do_test


# Data Imputation Restaurant
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test


# Data Imputation Buy
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --num_trials 3 --data_dir data/datasets/data_imputation/Buy --max_tokens 10 --do_test

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/data_imputation/Buy --max_tokens 10 --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/data_imputation/Buy --max_tokens 10 --do_test


# Error Detection Hopsital
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random  --num_trials 3 --data_dir data/datasets/error_detection/Hospital --do_test --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 0 --sample_method random --data_dir data/datasets/error_detection/Hospital --do_test --add_task_instruction

poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --sample_method manual --data_dir data/datasets/error_detection/Hospital --do_test
