# we use cross encoder as a baseline: https://www.sbert.net/examples/applications/cross-encoder/README.html
# encoder used: https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
# the baseline model optimizes on MSE

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_utils import *
from train_utils import *
import pandas as pd

parser = argparse.ArgumentParser(
    description="Specify experiment config file (for example, task1.config)"
)
parser.add_argument(
    "--config_file",
    type=str,
    help="Config file",
)

class BaseRankTrainer(BaseTrainer):

    # overrides
    def set_loss_func(self):
        # here we use mse loss instead of cross entropy loss
        self.train_config['loss_func'] = nn.MSELoss()

    # overrides
    def get_logits_and_loss(self, batch):
        # by not using **batch, we are indicating that we do not use token_type_ids for this project
        logits = self.model(**batch['encoding']).logits
        batch_loss = self.train_config['loss_func'](logits.view(-1), batch['label'])
        return logits, batch_loss

    # overrides
    def train(self):
        self.model.train()
        self.model.to(self.train_config['device'])
        n_batch, temp_loss = 0, 0.
        for idx, batch in self.data_loaders['train'].generate_batch_per_qid():
            for item in batch['encoding']:
                batch['encoding'][item] = batch['encoding'][item].to(self.train_config['device'])
            batch['label'] = batch['label'].to(self.train_config['device'])
            self.train_config['optimizer'].zero_grad()
            _, batch_loss = self.get_logits_and_loss(batch)
            temp_loss += batch_loss
            n_batch += 1
            batch_loss.backward()
            self.train_config['optimizer'].step()
            self.train_config['lr_scheduler'].step(batch_loss)
            if n_batch % int(self.train_config['print_freq']) == 0:
                print('Avg MSE Loss for batch {} - {}: {:.2f}'.format(n_batch - int(self.train_config['print_freq']) + 1,
                                                                  n_batch,
                                                                  temp_loss / int(self.train_config['print_freq'])))
                temp_loss = .0

    # overrides
    def eval(self, loader='dev'):
        self.model.eval()
        # when doing evaluations, the only metric we want to print is the loss
        n_batch, temp_loss = 0, 0.
        with torch.no_grad():
            for idx, batch in self.data_loaders[loader].generate_batch_per_qid():
                for item in batch['encoding']:
                    batch['encoding'][item] = batch['encoding'][item].to(self.train_config['device'])
                batch['label'] = batch['label'].to(self.train_config['device'])
                _, batch_loss = self.get_logits_and_loss(batch)
                temp_loss += batch_loss
                n_batch += 1
        print('Avg MSE Loss on loader {}: {:.2f}'.format(loader, temp_loss / n_batch))

        # save the best model during evaluation on dev set
        if loader == 'dev':
            if self.min_eval_loss is None or temp_loss / n_batch < self.min_eval_loss:
                # save model
                torch.save(self.model.state_dict(), self.train_config['save_model_dir'])
                # update loss
                self.min_eval_loss = temp_loss / n_batch


class RankingDataLoader:
    def __init__(self, config, query_data, product_data):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer'])
        self.query_data = query_data
        self.product_data = product_data

    def generate_batch_per_qid(self):
        for qid in self.query_data:
            products = self.product_data[qid]
            query_product_pairs = [[self.query_data[qid], products[i]['title']] for i in range(len(products))]
            labels = torch.tensor([products[i]['label'] for i in range(len(products))]).type(torch.float32)
            encoding = self.tokenizer(query_product_pairs, padding='max_length',
                                      max_length=int(self.config['max_length']),
                                      truncation=True, return_tensors='pt')
            yield qid, {'encoding': encoding, 'label': labels}


# different data split logic (qid_based)
# one qid can map to multiple product titles, so caching qid + qid encoding as one file,
# and product title encoding + corresponding qid + label is more space-saving
# we also want a different TensorDataset
def get_task1_dataloaders(config, data_dir='task_data/df_task_1.csv', train_ratio=0.9):
    query_col, product_col, id_col = config['query_col'], config['product_col'], config['id_col']
    cols = [query_col] + [product_col] + [id_col]
    label_col = config['label_col']
    label_mapper = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    print('Reading data, please wait...')
    data = pd.read_csv(data_dir)[cols + [label_col] + ['split']]
    data[label_col] = data[label_col].apply(lambda x: label_mapper[x])
    train = data[data['split'] == 'train']
    test = data[data['split'] == 'test']

    # id-based train/dev split
    train_query_data, train_product_data = {}, {}
    for i in range(len(train)):
        if train.iloc[i][id_col] not in train_query_data:
            train_query_data[train.iloc[i][id_col]] = train.iloc[i][query_col]
            train_product_data[train.iloc[i][id_col]] = [{'title': train.iloc[i][product_col],
                                                          'label': train.iloc[i][label_col]}]
        else:
            train_product_data[train.iloc[i][id_col]].append({'title': train.iloc[i][product_col],
                                                              'label': train.iloc[i][label_col]})
    del train
    # make test data
    test_query_data, test_product_data = {}, {}
    for i in range(len(test)):
        if test.iloc[i][id_col] not in test_query_data:
            test_query_data[test.iloc[i][id_col]] = test.iloc[i][query_col]
            test_product_data[test.iloc[i][id_col]] = [{'title': test.iloc[i][product_col],
                                                        'label':test.iloc[i][label_col]}]
        else:
            test_product_data[test.iloc[i][id_col]].append({'title': test.iloc[i][product_col],
                                                            'label': test.iloc[i][label_col]})
    del test

    qids = list(train_query_data.keys())
    split_point = int(len(qids) * train_ratio)
    train_qids, dev_qids = set(qids[:split_point]), set(qids[split_point:])
    dev_query_data = {k: train_query_data[k] for k in train_query_data if k in dev_qids}
    dev_product_data = {k: train_product_data[k] for k in train_product_data if k in dev_qids}
    train_query_data = {k: train_query_data[k] for k in train_query_data if k in train_qids}
    train_product_data = {k: train_product_data[k] for k in train_product_data if k in train_qids}

    train_loader = RankingDataLoader(config, train_query_data, train_product_data)
    dev_loader = RankingDataLoader(config, dev_query_data, dev_product_data)
    test_loader = RankingDataLoader(config, test_query_data, test_product_data)
    print('Data successfully loaded')

    return {'train': train_loader, 'dev': dev_loader, 'test': test_loader}


def main():
    args = parser.parse_args()
    config = Config(args.config_file)
    loaders = get_task1_dataloaders(config.process_data_config)
    model = AutoModelForSequenceClassification.from_pretrained(config.train_config['model']['pretrained_model'])
    trainer = BaseRankTrainer(model, loaders, config.train_config)
    trainer.train_dev_test()

if __name__ == '__main__':
    main()
