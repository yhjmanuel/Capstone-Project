import pickle
import os
from collections import defaultdict
import pyhdfs
from torch.utils.data import Dataset, DataLoader
import argparse


class Config:
    def __init__(self, config_file):
        self.process_data_config, self.train_config = defaultdict(defaultdict), defaultdict(defaultdict)
        self.split = '.'
        self.set = '='
        self.read_config(config_file)

    def read_config(self, config_file):
        with open(config_file) as config:
            for line in config:
                line = line[:-1]
                sub_configs = line.split(self.set)
                if len(sub_configs) == 2:
                    attrs = sub_configs[0].split(self.split)
                    if attrs[0] == "train":
                        if len(attrs) == 2:
                            self.train_config[attrs[1]] = sub_configs[1]
                        elif len(attrs) == 3:
                            self.train_config[attrs[1]][attrs[2]] = sub_configs[1]
                        else:
                            raise ValueError('{} is not a correct train config for this experiment'.format(line))
                    elif attrs[0] == "process":
                        if len(attrs) == 2:
                            self.process_data_config[attrs[1]] = sub_configs[1]
                        elif len(attrs) == 3:
                            self.process_data_config[attrs[1]][attrs[2]] = sub_configs[1]
                        else:
                            raise ValueError('{} is not a correct data process config for this experiment'.format(line))
        # used for making tokenizers
        self.process_data_config['tokenizer'] = self.train_config['model']['pretrained_model']


class TensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    # overwrite
    def __len__(self):
        return len(self.data)

    # overwrite
    def __getitem__(self, index):
        return self.data[index]


def load_cached_encodings(config):
    if config['hdfs']['use'] == 'true':
        fs = pyhdfs.HdfsClient(hosts=config['hdfs']['hosts'], user_name=config['hdfs']['user'])
        train_path = os.sep.join(config['hdfs']['save_dir'], config['cache']['train_dir'])
        dev_path = os.sep.join(config['hdfs']['save_dir'], config['cache']['train_dir'])
        test_path = os.sep.join(config['hdfs']['save_dir'], config['cache']['train_dir'])
        if not fs.exists(train_path) or not fs.exists(dev_path) or not fs.exists(test_path):
            raise FileNotFoundError('Cache does not exist on hdfs. Check config or Set process.cache_available=false')
        print('Loading cached encodings, please wait...')
        train_pkl = fs.open(train_path, 'rb')
        train_data = pickle.load(train_pkl)
        train_pkl.close()

        dev_pkl = fs.open(dev_path, 'rb')
        dev_data = pickle.load(dev_pkl)
        dev_pkl.close()

        test_pkl = fs.open(test_path, 'rb')
        test_data = pickle.load(test_pkl)
        test_pkl.close()
    else:
        if not os.path.exists(config['cache']['train_dir']) or not os.path.exists(config['cache']['dev_dir']) or not os.path.exists(config['cache']['test_dir']):
            raise FileNotFoundError('Cache does not exist on local. Check config or set process.cache_available=false')
        print('Loading cached encodings, please wait...')
        train_pkl = open(config['cache']['train_dir'], 'rb')
        train_data = pickle.load(train_pkl)
        train_pkl.close()

        dev_pkl = open(config['cache']['dev_dir'], 'rb')
        dev_data = pickle.load(dev_pkl)
        dev_pkl.close()

        test_pkl = open(config['cache']['test_dir'], 'rb')
        test_data = pickle.load(test_pkl)
        test_pkl.close()

    return train_data, dev_data, test_data


def get_dataloaders(train_data, dev_data, test_data, bs, dataset_class):
    # make datasets and dataloaders
    train_dataset = dataset_class(train_data)
    dev_dataset = dataset_class(dev_data)
    test_dataset = dataset_class(test_data)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, pin_memory=True)

    return {'train': train_loader, 'dev': dev_loader, 'test': test_loader}


def cache_encodings(config, train_data, dev_data, test_data):
    print('Caching encodings, please wait...')

    # store the files on hdfs
    if config['hdfs']['use'] == 'true':
        fs = pyhdfs.HdfsClient(hosts=config['hdfs']['hosts'], user_name=config['hdfs']['user'])
        path = config['hdfs']['dir']
        subpath = os.sep.join(config['cache']['train_dir'].split(os.sep)[:-1])
        # mkdir on hdfs
        if not fs.exists(os.sep.join(config['hdfs']['save_dir'], subpath)):
            fs.mkdirs(os.sep.join(config['hdfs']['save_dir'], subpath))

        train_pkl_hdfs = fs.open(os.sep.join(config['hdfs']['save_dir'], config['cache']['train_dir']), "wb")
        pickle.dump(train_data, train_pkl_hdfs)
        train_pkl_hdfs.close()

        dev_pkl_hdfs = fs.open(os.sep.join(config['hdfs']['save_dir'], config['cache']['dev_dir']), "wb")
        pickle.dump(dev_data, dev_pkl_hdfs)
        dev_pkl_hdfs.close()

        test_pkl_hdfs = fs.open(os.sep.join(config['hdfs']['save_dir'], config['cache']['test_dir']), "wb")
        pickle.dump(test_data, test_pkl_hdfs)
        test_pkl_hdfs.close()

    # save to local
    else:
        train_pkl = open(config['cache']['train_dir'], "wb")
        pickle.dump(train_data, train_pkl)
        train_pkl.close()

        dev_pkl = open(config['cache']['dev_dir'], "wb")
        pickle.dump(dev_data, dev_pkl)
        dev_pkl.close()

        test_pkl = open(config['cache']['test_dir'], "wb")
        pickle.dump(test_data, test_pkl)
        test_pkl.close()

    print('Encodings cached')