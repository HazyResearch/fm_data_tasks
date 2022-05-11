
install:
	poetry install
	poetry run pre-commit install

check:
	poetry run isort -c fm_data_tasks/
	poetry run black fm_data_tasks/ --check
	poetry run flake8 fm_data_tasks/

format:
	poetry run isort fm_data_tasks/
	poetry run black fm_data_tasks/


.PHONY: install check format
