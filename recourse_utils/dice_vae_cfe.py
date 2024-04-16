import os
import pickle
import time

import numpy as np
import torch
import yaml

from recourse_utils.dice_vae_model import DiceVAE
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from recourse_utils.constants import Cte


class DiceVAECFExplainer:
    def __init__(self, cfg, data_module, clf_model, seed):
        super().__init__()
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.vae_trainer = None
        self.seed = seed
        self.cfg = cfg
        self.num_nodes = data_module.num_nodes
        self.data_module = data_module
        self.batch_size = cfg.dice_vae.params.batch_size
        self.clf_model = clf_model
        self.immut_ids = cfg.dice_vae.params.immut_feats
        self.mut_ids = [x for x in range(self.num_nodes) if x not in self.immut_ids]
        self.dice_vae_model = DiceVAE(cfg, self.num_nodes, clf_model)
        if data_module.scaler is None:
            self.scaler = data_module.get_scaler(fit=True)
        else:
            self.scaler = data_module.scaler

    def load_model_from_ckpt(self, ckpt):
        self.dice_vae_model = DiceVAE.load_from_checkpoint(ckpt,
                                                           cfg=self.cfg,
                                                           num_nodes=self.num_nodes,
                                                           clf_model=self.clf_model)

    def setup_data(self):
        self.train_data = self.scaler.transform(self.data_module.datasets[0].X)
        self.val_data = self.scaler.transform(self.data_module.datasets[1].X)
        self.test_data = self.scaler.transform(self.data_module.datasets[2].X)

    def get_logging_name(self):
        logging_name = '_'.join(map(str, self.cfg.dice_vae.params.enc_hdim_list))
        logging_name += f"_{self.cfg.dice_vae.params.z_dim}_{self.cfg.dice_vae.params.act}_" \
                        f"{self.cfg.dice_vae.params.bn}_{self.cfg.dice_vae.params.drop_rate}_" \
                        f"{self.cfg.dice_vae.params.optim_lr}_" \
                        f"{self.cfg.dice_vae.params.loss_type}_" \
                        f"{self.cfg.dice_vae.params.beta_kld}_" \
                        f"{self.cfg.dice_vae.params.hinge_margin}_" \
                        f"{self.cfg.dice_vae.params.valid_reg}_{self.cfg.dice_vae.params.sens_reg}"
        return logging_name

    def train_vae_model(self):
        ckpt_callback = pl.callbacks.ModelCheckpoint(save_last=True)
        self.setup_data()
        data_path = f"{self.cfg.dataset.name}"
        data_spec_path = f"{self.cfg.dataset.sem_name}_{self.cfg.dice_vae.params.batch_size}"
        save_path = os.path.join(self.cfg.root_dir, data_path, data_spec_path)
        clf_spec_path = f"{self.clf_model.model_type}_{self.clf_model.fairness_type}_{self.clf_model.lmbd}"
        save_path = os.path.join(save_path, clf_spec_path)
        logging_name = self.get_logging_name()
        logger = CSVLogger(save_path, name=logging_name, version=self.seed)
        kwargs = {}
        if not self.cfg.verbose:
            kwargs['progress_bar_refresh_rate'] = 0
            kwargs['weights_summary'] = None
        self.vae_trainer = pl.Trainer(default_root_dir=save_path,
                                      callbacks=[ckpt_callback],
                                      gpus=None,
                                      max_epochs=self.cfg.dice_vae.params.max_epochs,
                                      check_val_every_n_epoch=self.cfg.dice_vae.params.val_every_n_epochs,
                                      logger=logger,
                                      **kwargs
                                      )
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.vae_trainer.fit(self.dice_vae_model, train_loader, val_loader)
        if self.cfg.verbose:
            print("Training Done. Saved at:", ckpt_callback.best_model_path)
        with open(os.path.join(logger.log_dir, 'params_dice.pkl'), 'wb') as f:
            pickle.dump(self.cfg, f)
        self.get_cost_preds_from_data(self.train_data,
                                      save_yaml_loc=os.path.join(logger.log_dir, 'dice_vae_train_result.yaml'))
        self.get_cost_preds_from_data(self.val_data,
                                      save_yaml_loc=os.path.join(logger.log_dir, 'dice_vae_val_result.yaml'),
                                      per_datum=True)

    def generate_counterfactual_exp(self, data, get_preds_labs=False):
        self.dice_vae_model.eval()
        # Check data shape!
        if not torch.is_tensor(data):
            data = torch.tensor(data)
        batch_size = data.size(0)
        data = data.view(batch_size, -1)
        init_labels = self.clf_model.get_labels(data).flatten()
        rec_cf = self.dice_vae_model.cf_reconstruct(data)
        rec_cf[init_labels == 1] = data[init_labels == 1]
        if not get_preds_labs:
            return rec_cf
        labs = self.clf_model.get_labels(rec_cf)
        pred_probs = self.clf_model.get_prediction_probs(rec_cf)
        return rec_cf, labs, pred_probs

    def get_cost_preds_from_data(self, data, save_yaml_loc='', return_results=False, per_datum=False):
        batch_size = data.size(0)
        data = data.view(batch_size, -1)
        avg_time = -1
        acts_list = []
        if not per_datum:
            cf_data, cf_labels, cf_probs = self.generate_counterfactual_exp(data, get_preds_labs=True)
        else:
            cf_data, cf_labels, cf_probs = [], [], []
            time_list = []
            for datum in data:
                init_time = time.time()
                datum = datum.view(1, -1)
                cf_d, cf_l, cf_p = self.generate_counterfactual_exp(datum, get_preds_labs=True)
                final_time = time.time()
                cf_data.append(cf_d.flatten())
                cf_labels.append(cf_l.flatten())
                cf_probs.append(cf_p.flatten())
                time_list.append(final_time - init_time)
                cf_flat = cf_d.flatten().detach().tolist()
                acts = {i: cf_flat[i] for i in self.mut_ids}
                acts_list.append(acts)
            cf_data = torch.stack(cf_data)
            cf_labels = torch.stack(cf_labels)
            cf_probs = torch.stack(cf_probs)
        cost_x = self.dice_vae_model.recon_loss(data, cf_data).item()
        avg_validity = torch.mean(cf_labels).item()
        avg_prediction_probabilities = torch.mean(cf_probs).item()
        if per_datum:
            avg_time = np.sum(time_list).item()
        cost_s_0, cost_s_1 = 0, 0
        cost_mutable_ids_s_0, cost_mutable_ids_s_1 = 0, 0
        degeneracy_measure_s_1, degeneracy_measure_s_0 = np.NaN, np.NaN
        adv_grp = data[:, 0].max()
        if 'sens' in self.data_module.name.lower():
            cost_s_1 = self.dice_vae_model.recon_loss(data[data[:, 0] == adv_grp],
                                                      cf_data[data[:, 0] == adv_grp]).item()
            cost_s_0 = self.dice_vae_model.recon_loss(data[data[:, 0] != adv_grp],
                                                      cf_data[data[:, 0] != adv_grp]).item()
            cost_mutable_ids_s_1 = self.dice_vae_model.recon_loss(
                data[data[:, 0] == adv_grp][:, self.mut_ids], cf_data[data[:, 0] == adv_grp][:, self.mut_ids]).item()
            cost_mutable_ids_s_0 = self.dice_vae_model.recon_loss(
                data[data[:, 0] != adv_grp][:, self.mut_ids], cf_data[data[:, 0] != adv_grp][:, self.mut_ids]).item()
        if save_yaml_loc != '' and save_yaml_loc is not None:
            result_dict = {"Cost": cost_x, "Validity": avg_validity, "Clf_Pred_Prob": avg_prediction_probabilities,
                           'Num_data': batch_size}
            if per_datum:
                result_dict['Avg_time'] = avg_time
            if self.data_module.dataset_name in [Cte.DATASET_LIST]:
                result_dict["Cost_S_0"] = cost_s_0
                result_dict["Cost_S_1"] = cost_s_1
                result_dict["Mutable_Cost_S_0"] = cost_mutable_ids_s_0
                result_dict["Mutable_Cost_S_1"] = cost_mutable_ids_s_1
            with open(save_yaml_loc, 'w') as yaml_file:
                yaml.dump(result_dict, yaml_file, default_flow_style=False)
        if return_results:
            if 'sens' in self.data_module.name.lower():
                return cost_x, avg_validity, avg_prediction_probabilities, avg_time, acts_list, cost_s_1, cost_s_0, \
                       degeneracy_measure_s_1, degeneracy_measure_s_0
            else:
                return cost_x, avg_validity, avg_prediction_probabilities, avg_time, acts_list
        if self.cfg['verbose']:
            print(f"Cost: {cost_x} || Validity: {avg_validity} || "
                  f"Classifier_Probability: {avg_prediction_probabilities}")
            if 'sens' in self.data_module.dataset_name.lower():
                print(f"Cost Sensitive_0: {cost_s_0} || Sensitive_1: {cost_s_1}")
                print(f"Mutable Cost Sensitive_0: {cost_mutable_ids_s_0} || Sensitive_1: {cost_mutable_ids_s_1}")
                print(f"Degeneracy Sensitive_0: {degeneracy_measure_s_0} || Sensitive_1: {degeneracy_measure_s_1}")
                print(f"Orig Adv: {data[data[:, 0] == adv_grp][:5]}")
                print(f"Recons Adv: {cf_data[data[:, 0] == adv_grp][:5]}")
                print(f"Orig DisAdv: {data[data[:, 0] != adv_grp][:5]}")
                print(f"Recons DisAdv: {cf_data[data[:, 0] != adv_grp][:5]}")
