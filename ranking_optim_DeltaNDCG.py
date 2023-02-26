import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from train_utils import *
from data_utils import *
from ranking_baseline import RankingDataLoader, get_task1_dataloaders

parser = argparse.ArgumentParser(
    description="Specify experiment config file (for example, task1.config)"
)
parser.add_argument(
    "--config_file",
    type=str,
    help="Config file",
)


class LambdaRankTrainer(BaseTrainer):

    def get_ndcg(self, labels):
        labels = labels.to(self.train_config['device'])
        return torch.tensor(self.get_dcg(labels) / self.get_ideal_dcg(labels)) if self.get_ideal_dcg(labels) > 0 else torch.tensor(.0)

    def get_dcg(self, labels):
        if 'k' in self.train_config['ndcg']:
            labels = labels[: int(self.train_config['ndcg']['k'])]
        gain_type = self.train_config['ndcg']['gain_type']
        x = torch.arange(1, len(labels) + 1, 1).to(self.train_config['device'])
        discounts = torch.log2(x + 1)
        gains = 2 ** labels - 1 if gain_type == 'exp2' else labels
        return torch.sum(gains / discounts)

    def get_ideal_dcg(self, labels):
        if 'k' in self.train_config['ndcg']:
            labels = labels[: int(self.train_config['ndcg']['k'])]
        gain_type = self.train_config['ndcg']['gain_type']
        ideal = torch.sort(torch.tensor(labels), descending=True)[0]
        ideal_x = torch.arange(1, len(ideal) + 1, 1)
        ideal_discounts = torch.log2(ideal_x + 1)
        ideal_gains = 2 ** ideal - 1 if gain_type == 'exp2' else ideal
        ideal_gains = ideal_gains.to(self.train_config['device'])
        ideal_discounts = ideal_discounts.to(self.train_config['device'])
        return torch.sum(ideal_gains / ideal_discounts)

    # overrides
    def get_logits_and_loss(self, batch):
        gain_type = self.train_config['ndcg']['gain_type']
        if gain_type != 'exp2' and gain_type != 'identity':
            raise ValueError('Wrong NDCG gain type, check param')
        preds = self.model(**batch['encoding']).logits
        rank = 1 + preds.view(-1).argsort(descending=True).argsort()
        labels = batch['label'].view(-1, 1)
        sigma = torch.tensor(float(self.train_config['ndcg']['sigma'])).to(self.train_config['device'])
        # we want to get lambda[ij] as gradient
        # calculate denominator (pairwise score diff) first
        pairwise_score_diff = 1.0 + torch.exp(sigma * (preds - preds.t()))
        # then we calculate delta ndcg
        pairwise_label_diff = labels - labels.t()
        # for a pair, if i should be ranked higher than j, Sij will be 1; if i and j have the
        # same ranking, Sij will be 0; if j should be ranked higher than i, Sij will be -1
        Sij = (pairwise_label_diff > 0).type(torch.float64) - (pairwise_label_diff < 0).type(torch.float64)

        # compute ndcg gain diff if each pair of labels are swapped
        ndcg_gain_diff = 2 ** labels - 2 ** labels.t() if gain_type == 'exp2' else labels - labels.t()
        # compute ndcg discount diff if each pair of labels are swapped
        rank = rank.view(-1, 1)
        ndcg_discount_diff = 1.0 / torch.log2(rank + 1.0) - 1.0 / torch.log2(rank.t() + 1.0)

        # compute ideal dcg
        ideal = torch.sort(labels, descending=True)[0]
        ideal_x = torch.arange(1, len(ideal) + 1, 1)
        ideal_discounts = torch.log2(ideal_x + 1)
        ideal_gains = 2 ** ideal - 1 if gain_type == 'exp2' else ideal
        ideal_dcg = torch.sum(ideal_gains / ideal_discounts)
        
        ndcg_gain_diff = ndcg_gain_diff.to(self.train_config['device'])
        ndcg_discount_diff = ndcg_discount_diff.to(self.train_config['device'])
        ideal_dcg.to(self.train_config['device'])
        # compute delta ndcg
        delta_ndcg = torch.abs(ndcg_gain_diff * ndcg_discount_diff / ideal_dcg)

        # compute lambda gradient
        Sij = Sij.to(self.train_config['device'])
        pairwise_score_diff = pairwise_score_diff.to(self.train_config['device'])
        gradient = sigma * (0.5 * (1 - Sij) - 1 / pairwise_score_diff) * delta_ndcg
        gradient = torch.sum(gradient, 1, keepdim=True)

        return preds, gradient

    # overrides
    def train(self):
        self.model.train()
        self.model.to(self.train_config['device'])
        n_batch = 0
        for qid, batch in self.data_loaders['train'].generate_batch_per_qid():
            # the rankings for all products are the same, ignore this batch
            # because the model can learn nothing from it
            if len(set(batch['label'])) == 1:
                continue
            for item in batch:
                batch[item].to(self.train_config['device'])
            self.train_config['optimizer'].zero_grad()
            preds, gradient = self.get_logits_and_loss(batch)
            n_batch += 1
            preds.backward(gradient)
            self.train_config['optimizer'].step()
            if n_batch % int(self.train_config['print_freq']) == 0:
                print('{} batches processed'.format(n_batch))
                ndcg = self.eval(loader='dev')
                self.train_config['lr_scheduler'].step(-ndcg)

    # overrides
    def eval(self, loader='dev'):
        # ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
        # truncate the labels if k is given
        self.model.eval()
        results = {}
        ndcgs = []
        with torch.no_grad():
            for qid, batch in self.data_loaders[loader].generate_batch_per_qid():
                if len(set(batch['label'])) == 1:
                # evaluate on a batch with only one label is meaningless
                    continue
                for item in batch:
                    batch[item].to(self.train_config['device'])
                preds, gradient = self.get_logits_and_loss(batch)
                results[qid] = {'preds': preds.view(-1), 'label': batch['label']}
        for qid in results:
            sorted_preds, sorted_labels = zip(*sorted(zip(results[qid]['preds'], results[qid]['label']), reverse=True))
            ndcgs.append(float(self.get_ndcg(torch.tensor(sorted_labels).cpu())))
        at_k_info = '@ {}'.format(self.train_config['ndcg']['k']) if 'k' in self.train_config['ndcg'] else ''
        print('NDCG on {} set {}: {}'.format(loader, at_k_info, np.mean(ndcgs)))
        print('--------------------------------------')
        torch.save(self.model.state_dict(), self.train_config['save_model_dir'])
        return np.mean(ndcgs)


def main():
    args = parser.parse_args()
    config = Config(args.config_file)
    model = AutoModelForSequenceClassification.from_pretrained(config.train_config['model']['pretrained_model'])
    loaders = get_task1_dataloaders(config.process_data_config)
    trainer = LambdaRankTrainer(model, loaders, config.train_config)
    trainer.train_dev_test()


if __name__ == '__main__':
    main()
