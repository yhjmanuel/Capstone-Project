import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(
    description="Specify task number and obtain data for either the classification task (task_num: 1) \
                or the ranking task (task num: 2)"
)
parser.add_argument(
    "--query_file",
    type=str,
    help="File containing query information",
)
parser.add_argument(
    "--product_file",
    type=str,
    help="File containing product information",
)
parser.add_argument(
    "--task_file_folder_dir",
    type=str,
    help="The folder where to save the processed task-specific files",
)
parser.add_argument(
    "--task_num",
    type=int,
    help="The number for the classification task is 1, and the number for ranking task is 2",
)
args = parser.parse_args()


df_examples = pd.read_parquet(args.query_file)
df_products = pd.read_parquet(args.product_file)

# merge the query dataset and the product dataset
df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)

# generate task-specific dataset
if args.task_num == 1:
    df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
    df_task_1_train = df_task_1[df_task_1["split"] == "train"]
    df_task_1_test = df_task_1[df_task_1["split"] == "test"]
    df_task_1 = df_task_1.reset_index(drop=True)
    df_task_1.to_csv(os.path.join(args.task_file_folder_dir, 'df_task_1.csv'), index=False)
elif args.task_num == 2:
    df_task_2 = df_examples_products[df_examples_products["large_version"] == 1]
    df_task_2_train = df_task_2[df_task_2["split"] == "train"]
    df_task_2_test = df_task_2[df_task_2["split"] == "test"]
    df_task_2 = df_task_2.reset_index(drop=True)
    df_task_2.to_csv(os.path.join(args.task_file_folder_dir, 'df_task_2.csv'), index=False)
print('Task-Specific Data Generated')
