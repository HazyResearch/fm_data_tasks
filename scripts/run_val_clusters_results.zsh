#!/usr/bin/zsh

# Commands to run 200 validation clusters results

# Entity Matching Fodors-Zagats
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --nan_tok "" --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/Fodors-Zagats/validation/text-davinci-002_10k_1cb_random_190run_0dry/trial_0.feather --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --do_test --nan_tok ""


# Entity Matching Beer
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Beer --nan_tok "" --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/Beer/validation/text-davinci-002_10k_1cb_random_91run_0dry/trial_0.feather --data_dir data/datasets/entity_matching/structured/Beer --nan_tok "" --do_test


# Entity Matching iTunes-Amazon
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/iTunes-Amazon --nan_tok "" --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/iTunes-Amazon/validation/text-davinci-002_10k_1cb_random_109run_0dry/trial_0.feather --data_dir data/datasets/entity_matching/structured/iTunes-Amazon --nan_tok "" --do_test


# Entity Matching Walmart-Amazon
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Walmart-Amazon --nan_tok "" --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/Walmart-Amazon/validation/text-davinci-002_10k_1cb_random_200run_0dry/trial_0.feather --data_dir data/datasets/entity_matching/structured/Walmart-Amazon --nan_tok "" --do_test


# Entity Matching Amazon-Google
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Amazon-Google --nan_tok "" --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/Amazon-Google/validation/text-davinci-002_10k_1cb_random_200run_0dry/trial_0.feather --data_dir data/datasets/entity_matching/structured/Amazon-Google --nan_tok "" --do_test


# Entity Matching DBLP-ACM
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/DBLP-ACM --nan_tok "" --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/DBLP-ACM/validation/text-davinci-002_10k_1cb_random_200run_0dry/trial_0.feather --data_dir data/datasets/entity_matching/structured/DBLP-ACM --nan_tok "" --do_test


# Entity Matching DBLP-GoogleScholar
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar --nan_tok "" --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/DBLP-GoogleScholar/validation/text-davinci-002_10k_1cb_random_200run_0dry/trial_0.feather --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar --nan_tok "" --do_test


# Data Imputation Restaurant
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/Restaurant/validation/text-davinci-002_10k_0cb_random_156run_0dry/trial_0.feather --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test


# Data Imputation Buy
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random --data_dir data/datasets/data_imputation/Buy --max_tokens 10

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/Buy/validation/text-davinci-002_10k_0cb_random_117run_0dry/trial_0.feather --data_dir data/datasets/data_imputation/Buy --max_tokens 10 --do_test


# Error Detection Hopsital
poetry run python3 -m fm_data_tasks.run_inference --num_run 200 --k 10 --sample_method random  --data_dir data/datasets/error_detection/Hospital --class_balanced

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method validation_clusters --validation_path outputs/Hopsital/validation/text-davinci-002_10k_1cb_random_189run_0dry/trial_0.feather --data_dir data/datasets/error_detection/Hospital --do_test
