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

Unpack the data:
```
tar xvf data/datasets.tar.gz
```

# Setup
You need to set your OpenAI key to run GPT inference. We also let you change where the datasets are downloaded in case you want to run the code on other data. We use the environment variables
```
export OPENAI_API_KEY="<YOU API KEY>"
export DATASET_PATH="$PWD/data/datasets"
```

# Run
