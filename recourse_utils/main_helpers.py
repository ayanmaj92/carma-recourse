import torch.nn
import numpy as np
import random
import pytorch_lightning as pl

from causal_flows.zuko.transforms import ComposedTransform


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def filter_datasets_by_label(data_module, clf_model, frac=1.0, filter_label=0.0):
    # TRAINING DATA.
    assert filter_label in [0.0, 1.0], "binary 0 or 1 for label!"
    list_datasets = data_module.datasets
    if frac > 1:
        frac = 1.0
    if frac <= 0.001:
        frac = 0.001
    if data_module.scaler is None:
        data_module.get_scaler(fit=True)
    for idx, dataset in enumerate(list_datasets):
        scaled_data = data_module.scaler.transform(dataset.X)
        pred_labels = clf_model.get_labels(scaled_data).detach().numpy().flatten()
        pred_probs = clf_model.get_prediction_probs(scaled_data).detach().numpy().flatten()
        dataset.pred_label = torch.tensor(pred_labels)
        dataset.pred_prob = torch.tensor(pred_probs)

        mask_prediction_negative = pred_labels == filter_label
        if dataset.X is not None:
            dataset.X = dataset.X[mask_prediction_negative, :]
        if dataset.U is not None:
            dataset.U = dataset.U[mask_prediction_negative, :]
        if dataset.Y is not None:
            dataset.Y = dataset.Y[mask_prediction_negative]
        dataset.pred_label = dataset.pred_label[mask_prediction_negative]
        dataset.pred_prob = dataset.pred_prob[mask_prediction_negative]

        if frac < 1.0 and idx in [0, 1]:  # only filter the training and validation data, fix the test data size.
            len_d = int(frac * len(dataset.X))
            idx = np.random.choice(np.arange(len(dataset.X)), len_d, replace=False)
            dataset.X = dataset.X[idx, :]
            dataset.U = dataset.U[idx, :]
            dataset.Y = dataset.Y[idx]
            dataset.pred_label = dataset.pred_label[idx]
            dataset.pred_prob = dataset.pred_prob[idx]


def set_z_data(data_module, causal_model, causal_model_type):
    list_datasets = data_module.datasets
    if data_module.scaler is None:
        data_module.get_scaler(fit=True)
    for dataset in list_datasets:
        loader = data_module._data_loader(dataset, batch_size=dataset.X.shape[0], shuffle=False, num_workers=1)
        if causal_model_type == 'causal_nf':
            n_flow = causal_model.flow()
            n_flow.transform = ComposedTransform(data_module.scaler_transform, n_flow.transform)
            for batch in loader:
                z_factual = n_flow.transform(batch[0])
                z_factual = z_factual.view(batch[0].shape[0], -1, 1)
                dataset.Z = z_factual
        elif causal_model_type == 'vaca':
            for batch in loader:
                num_graphs = batch.num_graphs
                num_nodes = batch.x.shape[0]
                batch.x = data_module.scaler.transform(batch.x.reshape(num_graphs, -1)).reshape(
                        num_nodes, -1
                    )
                logits = causal_model.encoder_gnn(batch)
                z_factual, qz_x = causal_model.posterior_distr(logits, return_mean=True)
                z_factual = z_factual.view(num_graphs, -1, causal_model.z_dim)
                dataset.Z = z_factual
