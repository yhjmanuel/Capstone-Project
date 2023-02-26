import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class BaseTrainer:
    def __init__(self, model, data_loaders, train_config):
        self.optimizer, self.lr_scheduler = None, None
        self.model = model
        self.data_loaders = data_loaders
        self.train_config = train_config
        self.min_eval_loss = None
        self.normalize_config()

    def normalize_config(self):
        self.train_config['optimizer'] = torch.optim.Adam(self.model.parameters(), lr=float(self.train_config['lr']))
        ls_info = self.train_config['lr_scheduler']
        self.train_config['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.train_config['optimizer'],
                                            factor=float(ls_info['factor']), mode="min", patience=int(ls_info['patience']),
                                            cooldown=int(ls_info['cooldown']), min_lr=float(ls_info['min_lr']), verbose=True)
        save_model_dir = os.sep.join(self.train_config['save_model_dir'].split(os.sep)[:-1])
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        self.set_loss_func()

    def set_loss_func(self):
        # add loss function
        self.train_config['loss_func'] = nn.CrossEntropyLoss()

    # Simplifies inherited classes
    # This only applies to the classification task. For the ranking task,
    # we also need to re-write the eval method
    def get_logits_and_loss(self, batch):
        # by not using **batch, we are indicating that we do not use token_type_ids for this project
        logits = self.model(batch['encoding'])
        batch_loss = self.train_config['loss_func'](logits, batch['label'])
        return logits, batch_loss

    def train_dev_test(self):
        for i in range(int(self.train_config['n_epoch'])):
            print('******************Epoch {}******************'.format(i+1))
            self.train()
            self.eval(loader='train')
            self.eval(loader='dev')
        print('******************Training Finished******************')
        if 'test' in self.data_loaders:
            self.eval(loader='test')

    def train(self):
        self.model = self.model.to(self.train_config['device'])
        self.model.train()
	n_batch, temp_loss = 0, 0.
        for idx, batch in enumerate(self.data_loaders['train']):
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
                print('Avg Loss for batch {} - {}: {:.2f}'.format(n_batch - int(self.train_config['print_freq']) + 1,
                                                                  n_batch,
                                                                  temp_loss / int(self.train_config['print_freq'])))
                temp_loss = .0

    def eval(self, loader='dev'):
        n_batch, temp_loss, pred, label = 0, 0., [], []
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(self.data_loaders[loader]):
                for item in batch['encoding']:
                    batch['encoding'][item] = batch['encoding'][item].to(self.train_config['device'])
                batch['label'] = batch['label'].to(self.train_config['device'])
                logits, batch_loss = self.get_logits_and_loss(batch)
                temp_loss += batch_loss
                n_batch += 1
                pred.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                label.extend(batch['label'].cpu().numpy())
        print('Avg Loss on loader {}: {:.2f}'.format(loader, temp_loss / n_batch))
        print('Micro F1 on loader {}: {:.2f}'.format(loader, f1_score(label, pred, average='micro')))
        print('Macro F1 on loader {}: {:.2f}'.format(loader, f1_score(label, pred, average='macro')))

        # save the best model during evaluation on dev set
        if loader == 'dev':
            if self.min_eval_loss is None or temp_loss / n_batch < self.min_eval_loss:
                # save model
                torch.save(self.model.state_dict(), self.train_config['save_model_dir'])
                # update loss
                self.min_eval_loss = temp_loss / n_batch

