import torch
from data_utils import *
from train_utils import *
from classification_model import *
import pandas as pd
import random

parser = argparse.ArgumentParser(
    description="Specify experiment config file (for example, task1.config)"
)
parser.add_argument(
    "--config_file",
    type=str,
    help="Config file",
)

def cache_task2_encodings(config, data_dir='task_data/df_task_2.csv', seed=42, train_ratio=0.9):
    # store cache in this path
    if not os.path.exists(data_dir):
        raise FileNotFoundError('Data dir does not exist. Generate task-specific data first before proceeding')
    if not os.path.exists(os.sep.join(config['cache']['train_dir'].split(os.sep)[:-1])):
        os.mkdir(os.sep.join(config['cache']['train_dir'].split(os.sep)[:-1]))
    # make data
    print('Reading data, please wait...')
    cols = config['data_cols'].split(',')
    label_col = config['label_col']
    label_mapper = {'E': 0, 'S': 1, 'C': 2, 'I': 3}
    data = pd.read_csv(data_dir)[cols + [label_col] + ['split']]
    data[label_col] = data[label_col].apply(lambda x: label_mapper[x])
    train = data[data['split'] == 'train']
    test = data[data['split'] == 'test']

    tokenizer = BertTokenizer.from_pretrained(config['tokenizer'])

    def encode(sequence):
        encoding = tokenizer(sequence, padding='max_length',
                             max_length=int(config['max_length']),
                             return_token_type_ids=False,
                             truncation=True, return_tensors='pt')
        return encoding


    # cache encodings
    print('Encoding data, please wait...')
    train_data = [{'encoding': encode(' '.join([str(train.iloc[i][col]) for col in cols])),
                   'label': train.iloc[i][label_col]} for i in range(len(train))]
    test_data = [{'encoding': encode(' '.join([str(test.iloc[i][col]) for col in cols])),
                  'label': test.iloc[i][label_col]} for i in range(len(test))]
    del train, test

    # split data (make train, dev, test sets)
    random.Random(seed).shuffle(train_data)
    split_point = int(len(train_data) * train_ratio)
    dev_data = train_data[split_point:]
    train_data = train_data[:split_point]

    cache_encodings(config, train_data, dev_data, test_data)

def main():
    args = parser.parse_args()
    config = Config(args.config_file)
    if config.process_data_config['cache_available'] != 'true':
        cache_task2_encodings(config.process_data_config)
    train_data, dev_data, test_data = load_cached_encodings(config.process_data_config)
    loaders = get_dataloaders(train_data, dev_data, test_data,
                              int(config.process_data_config['batch_size']), TensorDataset)
    model = BaseBERTClassifier(config.train_config['model'])
    trainer = BaseTrainer(model, loaders, config.train_config)
    trainer.train_dev_test()


if __name__ == '__main__':
    main()
