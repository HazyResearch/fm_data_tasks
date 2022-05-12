#!/usr/bin/zsh

# Commands to run 200 examples for each dataset with metrics

# Entity Matching Fodors-Zagats
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Fodors-Zagats  --do_test --num_run 200 --add_task_instruction --class_balanced
# >> 0.9302325581 F1

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/Fodors-Zagats --do_test --num_run 200 --add_task_instruction
# >>>

# Entity Matching Beer
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Beer  --do_test --num_run 200 --add_task_instruction --class_balanced
# >> 0.9333333333333333 F1

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/Beer --do_test --num_run 200 --add_task_instruction
# >>>

# Entity Matching iTunes-Amazon
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/iTunes-Amazon  --do_test --num_run 200 --add_task_instruction --class_balanced
# >> 0.8518518518518519 F1

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/iTunes-Amazon --do_test --num_run 200 --add_task_instruction
# >>>

# Entity Matching Walmart-Amazon
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Walmart-Amazon  --do_test --num_run 200 --add_task_instruction --class_balanced
# >> 0.625 F1

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/Walmart-Amazon --do_test --num_run 200 --add_task_instruction
# >>>

# Entity Matching Amazon-Google
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/Amazon-Google  --do_test --num_run 200 --add_task_instruction --class_balanced
# >> 0.72 F1

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/Amazon-Google --do_test --num_run 200 --add_task_instruction
# >>>

# Entity Matching DBLP-ACM
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/DBLP-ACM  --do_test --num_run 200 --add_task_instruction --class_balanced
# >> 1.0 F1

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/DBLP-ACM --do_test --num_run 200 --add_task_instruction
# >>>

# Entity Matching DBLP-GoogleScholar
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar  --do_test --num_run 200 --add_task_instruction --class_balanced
# >> 0.7096774193548387 F1

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/entity_matching/structured/DBLP-GoogleScholar --do_test --num_run 200 --add_task_instruction
# >>>

# Data Imputation Restaurant
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test --num_run 200 --add_task_instruction
# >>> 79 Accuracy
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/data_imputation/Restaurant --max_tokens 5 --do_test --num_run 200 --add_task_instruction
# >>> 90

# Data Imputation Buy
poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method random --data_dir data/datasets/data_imputation/Buy --max_tokens 5 --do_test --num_run 200 --add_task_instruction
# >>> 0.8923076923076924 Accuracy

poetry run python3 -m fm_data_tasks.run_inference --k 10 --sample_method manual --data_dir data/datasets/data_imputation/Buy --max_tokens 5 --do_test --num_run 200 --add_task_instruction
# >>>
