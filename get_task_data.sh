#!/bin/bash
if [[ (! $# -eq 1 ) || (! $1 -eq 1  && ! $1 -eq 2 ) ]]
then
    echo 'wrong param: 1 param in total (task_num, must be 1 for classification, or 2 for ranking)';
    exit
fi

# specify query file and product file directories
QUERY_FILE=shopping_queries_dataset_examples.parquet
PRODUCT_FILE=shopping_queries_dataset_products.parquet
TASK_FILE_FOLDER_DIR=task_data

if [ ! -f $QUERY_FILE_DIR ] || [ ! -f $PRODUCT_FILE_DIR ]
then
    echo 'Query file or Product file does not exist';
    exit
fi

if [ ! -r $TASK_FILE_FOLDER_DIR ]
then
    mkdir $TASK_FILE_FOLDER_DIR
fi

echo 'Start generating data for task' $1
python get_task_data.py \
       --task_num $1 \
       --query_file $QUERY_FILE \
       --product_file $PRODUCT_FILE \
       --task_file_folder_dir $TASK_FILE_FOLDER_DIR
echo 'Data successfully generated for task' $1
