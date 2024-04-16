import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
from recourse_modules.dense import MLPModule
from fairtorch import DemographicParityLoss, EqualiedOddsLoss
from recourse_utils.clf_constraints import EqualOpportunityLoss
import glob


class NNClassifier(torch.nn.Module):
    def __init__(self, dim_list, cfg):
        super(NNClassifier, self).__init__()
        self.model = MLPModule(h_dim_list=dim_list,
                               activ_name=cfg.act,
                               bn=cfg.bn,
                               drop_rate=cfg.drop_rate)

    def forward(self, x):
        outputs = torch.sigmoid(self.model(x))
        return outputs


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class Classifier:
    def __init__(self, input_dim, cfg):
        self.clf_model = None
        self.input_dim = input_dim
        self.output_dim = 1
        if len(cfg.dim_list) == 0:
            self.clf_model = LogisticRegression(input_dim, output_dim=self.output_dim)
            self.model_type = 'linear'
        else:
            dim_list = [input_dim] + cfg.dim_list + [self.output_dim]
            self.clf_model = NNClassifier(dim_list, cfg)
            self.model_type = 'nn'
        self.fairness_type = cfg.fairness
        # For now we always do binary prediction (1-0)
        self.criterion = torch.nn.BCELoss()
        self.sens_idx = cfg.sens_idx
        self.lmbd = cfg.lmbd
        if self.fairness_type == 'dp':
            print("Lambda fairness trade off", self.lmbd)
            self.discrimination = DemographicParityLoss(sensitive_classes=[0, 1], alpha=self.lmbd)
        # this is equal odds NOT equal opportunity
        elif self.fairness_type == 'eod':
            self.discrimination = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=self.lmbd)
            # self.discrimination.c = self.discrimination.c[:4]
            # self.discrimination.M = self.discrimination.M[:4, :]
        elif self.fairness_type == 'eop':
            self.discrimination = EqualOpportunityLoss(sensitive_classes=[0, 1], alpha=self.lmbd)
        else:
            self.discrimination = None
        self.optimizer = torch.optim.SGD(self.clf_model.parameters(), lr=cfg.learning_rate)
        self.batch_size = cfg.batch_size
        self.epochs = cfg.epochs

    def train_model(self, train_data, train_labels, train_sens_unscaled, cfg):
        verbose = cfg["verbose"]

        if verbose:
            range_object = tqdm(range(int(self.epochs)), desc='Training Epochs')
        else:
            range_object = range(int(self.epochs))

        train_set = TensorDataset(train_data, train_labels, train_sens_unscaled)
        train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)

        for _ in range_object:
            for batch in train_loader:
                x, labels, s = batch
                self.optimizer.zero_grad()  # Setting our stored gradients equal to zero
                outputs = self.clf_model(x)
                loss = self.criterion(torch.squeeze(outputs), labels)
                # AM: FairTorch uses logits! We were passing output probabilities!
                out_logits = torch.special.logit(outputs)
                if type(self.discrimination) == DemographicParityLoss:
                    loss += self.discrimination(x, torch.squeeze(out_logits), s)
                elif type(self.discrimination) == EqualiedOddsLoss:
                    loss += self.discrimination(x, torch.squeeze(out_logits), s, labels)
                elif type(self.discrimination) == EqualOpportunityLoss:
                    loss += self.discrimination(x, torch.squeeze(out_logits), s, labels)

                loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
                self.optimizer.step()  # Updates weights and biases with the optimizer (SGD)
        save_to = None
        self.validate(train_data, train_labels, train_sens_unscaled, save_to, verbose, 'Train')

    @staticmethod
    def valid_accuracy_fairness(outputs, s, y):
        total = 0
        correct = 0
        total += y.size(0)
        correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == torch.squeeze(y).detach().numpy())
        accuracy = correct / total

        pred = torch.squeeze(outputs).round().detach()
        pred_s_0 = torch.squeeze(1 - s) * pred
        pred_s_1 = torch.squeeze(s) * pred
        discrimination_dp = abs((torch.sum(pred_s_0).item() / len(pred))
                                - (torch.sum(pred_s_1).item() / len(pred)))
        pred = torch.squeeze(outputs).round().detach()
        pred_s_0_y_1 = torch.squeeze(1 - s) * pred * y
        pred_s_1_y_1 = torch.squeeze(s) * pred * y
        total_s_0_y_1 = torch.squeeze(1 - s) * y
        total_s_1_y_1 = torch.squeeze(s) * y
        pred_s_0_y_0 = torch.squeeze(1 - s) * pred * (1 - y)
        pred_s_1_y_0 = torch.squeeze(s) * pred * (1 - y)
        total_s_0_y_0 = torch.squeeze(1 - s) * (1 - y)
        total_s_1_y_0 = torch.squeeze(s) * (1 - y)
        discrimination_y_1 = abs(
            (torch.sum(pred_s_0_y_1).item() / torch.sum(total_s_0_y_1).item())
            - (torch.sum(pred_s_1_y_1).item() / torch.sum(total_s_1_y_1).item()))
        discrimination_y_0 = abs(
            (torch.sum(pred_s_0_y_0).item() / torch.sum(total_s_0_y_0).item())
            - (torch.sum(pred_s_1_y_0).item() / torch.sum(total_s_1_y_0).item()))
        discrimination_eod = (discrimination_y_1 + discrimination_y_0) / 2.0
        return accuracy, discrimination_dp, discrimination_y_0, discrimination_y_1, discrimination_eod

    def validate(self, valid_data, valid_labels, valid_sens, save_to=None, verbose=False, typ_data='Valid'):
        losses = []
        with torch.no_grad():
            # Calculating the loss and accuracy for the test dataset
            correct = 0
            total = 0
            x = valid_data
            y = valid_labels
            s = valid_sens
            outputs = self.clf_model(x)
            loss = self.criterion(torch.squeeze(outputs).to(torch.float32), y.to(torch.float32))

            predicted = outputs.round().detach().numpy()
            total += y.size(0)
            correct += np.sum(predicted == y.detach().numpy())
            accuracy = 100 * correct / total
            losses.append(loss.item())

            # Calculating the loss and accuracy for the train dataset
            accuracy, discrimination_dp, discrimination_y_0, discrimination_y_1, discrimination_eod = \
                self.valid_accuracy_fairness(outputs, s, y)
            if verbose:
                print(f"Lambda {self.lmbd}, Fairness {type(self.discrimination)}")
                print(f"{typ_data} -  Loss: {loss.item()}. Accuracy: {accuracy}. "
                      f"DP Discrimination: {discrimination_dp}. "
                      f"EOD Discrimination: {[discrimination_y_0, discrimination_y_1]}")
                print(f"{typ_data} data predicted label distribution (%) pred=1, pred=0: "
                      f"{sum(predicted).item() / len(predicted)}"
                      f"{sum(1 - predicted).item() / len(predicted)}")

            if save_to is not None:
                import csv
                my_dict = {"loss": loss.item(), "discrimination_dp": discrimination_dp,
                           "discrimination_y_0": discrimination_y_0, "discrimination_y_1": discrimination_y_1,
                           "discrimination_eod": discrimination_eod, "accuracy": accuracy,
                           "pred1": sum(predicted).item() / len(predicted)}
                if verbose:
                    print(f"Saving clf results now to {save_to}/metrics.csv")
                with open(f'{save_to}/metrics.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
                    w = csv.DictWriter(f, my_dict.keys())
                    w.writeheader()
                    w.writerow(my_dict)

    def get_labels(self, data):
        if type(data) is not torch.Tensor:
            data = torch.Tensor(data)
        return self.clf_model(data).round()

    def get_prediction_probs(self, data):
        return self.clf_model(data)

    def __call__(self, x, **kwargs):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        labels = self.get_labels(x)
        probs = self.get_prediction_probs(x)
        return labels, probs


def create_clf(preparator, cfg):
    cfg.input_dim = preparator.datasets[0].X.shape[1]
    clf_obj = Classifier(cfg.input_dim, cfg.params)
    return clf_obj


def load_clf(ckpt_file, cfg):
    clf_obj = Classifier(cfg.input_dim, cfg.params)
    clf_ckpts = glob.glob(ckpt_file)

    assert len(clf_ckpts) == 1, "more ore less than one ckpt found!"
    clf_obj.clf_model = torch.load(clf_ckpts[0])
    clf_obj.model_type = cfg.name
    return clf_obj


def train_clf(preparator, cfg, verbose, clf_obj=None):
    if clf_obj is None:
        clf_obj = create_clf(preparator, cfg.classifier)
    train_dataset = preparator.datasets[0]
    train_data = torch.tensor(train_dataset.X).type(torch.float)
    if verbose:
        print(train_data.size())
    train_labels = torch.tensor(train_dataset.Y).type(torch.float).reshape(-1)
    sens_idx = clf_obj.sens_idx
    train_sens_unscaled = train_data[:, sens_idx].view(-1, 1)

    scaler = preparator.get_scaler(fit=True)
    data_train = scaler.transform(train_data)
    clf_obj.train_model(data_train, train_labels, train_sens_unscaled, cfg)
    labs = clf_obj.get_labels(torch.tensor(scaler.transform(train_data)))
    if verbose:
        print("Train data predicted label distribution (%) pred=1, pred=0:", torch.sum(labs).item() / len(labs),
              torch.sum(1 - labs).item() / len(labs), "\n")
    return clf_obj


def validate_clf(clf, preparator, verbose, save_to=None):
    valid_data, valid_labels = torch.tensor(preparator.datasets[1].X).type(torch.float), \
                               torch.tensor(preparator.datasets[1].Y).reshape(-1)
    valid_sens = valid_data[:, clf.sens_idx]

    data_valid = preparator.scaler.transform(valid_data)
    clf.validate(data_valid, valid_labels, valid_sens, save_to, verbose)
