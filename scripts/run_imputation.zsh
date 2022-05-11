#!/usr/bin/zsh

# Run Imputation
# fld="restaurant"
# file="restaurant_imputation"
# col_to_dis="idx"
# label_col="city"

fld="buy"
file="buy"
col_to_dis="id"
label_col="manufacturer"

for model_name in "text-ada-001" "text-babbage-001" "text-curie-001" "text-davinci-002"; do
    # Run manual
    poetry run python3 gpt3_imputation.py \
        --test_data /home/users/laurel/data/munching/imputation/$fld/test1.csv \
        --train_data /home/users/laurel/data/munching/imputation/$fld/$file.csv \
        --label_column $label_col \
        --output_dir /home/users/laurel/logs/munching/imputation/ \
        --columns_to_discard $col_to_dis \
        --sample_method manual \
        --k 2 \
        --model_name $model_name \
        --num_run 110
    for k in 2 10; do
        poetry run python3 gpt3_imputation.py \
            --test_data /home/users/laurel/data/munching/imputation/$fld/test1.csv \
            --train_data /home/users/laurel/data/munching/imputation/$fld/$file.csv \
            --label_column $label_col \
            --output_dir /home/users/laurel/logs/munching/imputation/ \
            --columns_to_discard $col_to_dis \
            --sample_method random \
            --k $k \
            --model_name $model_name \
            --num_run 110
    done
done
