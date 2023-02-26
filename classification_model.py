import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BaseBERTClassifier(nn.Module):
    def __init__(self, model_config):
        super(BaseBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_config['pretrained_model']).to('cuda:0')
        self.act = None
        self.hidden_size = self.bert.config.hidden_size
        self.dense_layers = nn.ModuleList()
        self.parse_config(model_config)

    def parse_config(self, model_config):
        if model_config['act'] == 'relu':
            self.act = nn.ReLU()
        elif model_config['act'] == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('Unsupported activation function type, please check config')
        if 'dense_layers' in model_config:
            dense_layers = [int(model_config['dense_layers'][i]) for i in range(len(model_config['dense_layers'].split(',')))]
            self.set_dense_layers(dense_layers)

    def set_dense_layers(self, dense_layers):
        dense_layers = nn.Sequential()
        for i in range(0, len(dense_layers)):
            if i == 0:
                dense_layers.add_module("reproject_0", nn.Linear(self.hidden_size, dense_layers[0]))
            else:
                dense_layers.add_module("reproject_" + str(i), nn.Linear(dense_layers[i-1], dense_layers[i]))
            # do not set activation function for the last dense layer
            if i != len(dense_layers) - 1:
                dense_layers.add_module("act_" + str(i), self.act)
        self.dense_layers.append(dense_layers)

    def forward(self, x):
        x = self.bert(input_ids=x['input_ids'].squeeze(),
                      attention_mask=x['attention_mask'].squeeze()).last_hidden_state[:,0,:].squeeze()

        return self.dense_layers[0](x)
