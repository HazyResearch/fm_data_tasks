This is the repo for Foundation Models for Data Tasks.

# Install
Download the code:
```
git clone git@github.com:HazyResearch/fm_data_tasks.git
cd fm_data_tasks
```

Install:
```
pip install poetry
poetry install
poetry run pre-commit install
```
or
```
make install
```

<<<<<<< HEAD
Unpack the data:
```
tar xvf data/datasets.tar.gz
=======
Download and unpack the data:
```
mkdir data
wget https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz -P data
tar xvf data/datasets.tar.gz -C data/
>>>>>>> b1aca98022bd4498a3e92d96c41e304424584a6b
```

# Setup
You need to set your OpenAI key to run GPT inference. We also let you change where the datasets are downloaded in case you want to run the code on other data. We use the environment variables
```
export OPENAI_API_KEY="<YOU API KEY>"
export DATASET_PATH="$PWD/data/datasets"
```

# Run
<<<<<<< HEAD
=======
To run inference, use
```
python3 -m fm_data_tasks.run_infernece --help
```
To see options.

To see a full set of scripts with output results for 200 examples samples of each dataset, see `scripts/run_results.zsh`.

Some examples are a follows.

To run 10 dry run examples for Fodors Zagats entity matching with random selection,
```
python3 -m fm_data_tasks.run_inference \
    --dry_run \
    --num_run 10 \
    --sample_method random \
    --data_dir data/datasets/entity_matching/structured/Fodors-Zagats
```

To run 100 examples for 3 trials for Restaurant data imputation on the test data with manual selection,
```
python3 -m fm_data_tasks.run_inference \
    --num_run 100 \
    --num_trials 3 \
    --do_test \
    --sample_method manual \
    --data_dir data/datasets/data_imputation/Restaurant
```
>>>>>>> b1aca98022bd4498a3e92d96c41e304424584a6b
