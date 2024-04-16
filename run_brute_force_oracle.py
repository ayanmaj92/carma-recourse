import argparse
import glob
import itertools
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from causal_flows.causal_nf.preparators.scm import SCMPreparator
from recourse_utils.clf_trainer import load_clf, validate_clf
from recourse_utils.main_helpers import *
from yacs.config import CfgNode as CN
from recourse_utils.recourse_reward_utils import feat_stats
from scipy.special import erf
import recourse_utils.args_parser as argtools
import torch.nn.functional as F


def gauss_cdf(x, m, s):
    return 0.5 * (1 + erf((x - m) / (np.sqrt(2) * s)))


def ReLU(x):
    return x * (x > 0)


def compute_counterfactual(scm, x_factual, indices, values):
    u = scm.transform.inv(x_factual)
    for index, value in zip(indices, values):
        scm.intervene(index, value)
    x_cf = scm.transform(u)
    for index in indices:
        scm.stop_intervening(index)
    return x_cf


def is_point_constraint_satisfied(data_mod, clf_model, data_f, action_set):
    indices, values = action_set.keys(), action_set.values()
    if len(data_f.shape) == 1:
        data_f = data_f.view(1, -1)
    counterfactual_instance = compute_counterfactual(data_mod.scm, data_f, indices, values)
    with torch.no_grad():
        scaled_data = data_mod.scaler.transform(counterfactual_instance)
        prob_pred = clf_model(scaled_data)[1].item()
        # if clf_model.model_type == 'linear':
        #     prob_pred = clf_model(torch.tensor(transform_data(counterfactual_instance)).float())[1].item()
        # else:
        #     prob_pred = clf_model(torch.tensor(transform_data(counterfactual_instance)).float().view(1, -1))[1].item()
    return counterfactual_instance, prob_pred


def get_valid_discretized_action_sets(training_data, data_sub, intervene_idxs, grid_search_bins, max_intervention_cardinality):
    possible_actions_per_node = []
    for idx in intervene_idxs:
        # For now considering real numeric.
        number_decimals = 10
        tmp_min = torch.min(training_data[:, idx])
        tmp_max = torch.max(training_data[:, idx])
        # tmp_mean = torch.mean(training_data[:, idx])
        tmp = list(
            # np.around(
            torch.round(
                torch.linspace(
                    data_sub[idx] - 2 * (data_sub[idx] - tmp_min),
                    data_sub[idx] + 2 * (tmp_max - data_sub[idx]),
                    grid_search_bins
                ), decimals=number_decimals
            )
        )
        tmp.append('n/a')  # This accounts for do not intervene.
        tmp = list(dict.fromkeys(tmp))  # This removes any potential duplicates.
        possible_actions_per_node.append(tmp)

    all_action_tuples = list(itertools.product(
        *possible_actions_per_node
    ))
    all_action_tuples = [
        elem1 for elem1 in all_action_tuples
        if len([
            elem2 for elem2 in elem1 if elem2 != 'n/a'
        ]) <= max_intervention_cardinality
    ]
    all_action_sets = [
        dict(zip(intervene_idxs, elem))
        for elem in all_action_tuples
    ]
    valid_action_sets = []
    for action_set in all_action_sets:
        valid_action_sets.append({k: v for k, v in action_set.items() if v != 'n/a'})
    if {} in valid_action_sets:
        valid_action_sets.remove({})
    return valid_action_sets


def measure_cost(data_f, cf_data, training_data, scaler, intervene_feats, action_set, weights, cost_type='l2'):
    # Whenever we are doing action measurement, we NEED TO DO WITH TRANSFORMED DATA.

    f_scaled = torch.flatten(scaler.transform(data_f))
    cf_scaled = torch.flatten(scaler.transform(cf_data))
    train_data_scaled = scaler.transform(training_data)
    action_tensor = torch.zeros_like(f_scaled)
    weights = np.array(weights)
    # if np.sum(weights) > 1.0:
    #     weights = weights / np.sum(weights)
    for i in action_set.keys():
        action_tensor[i] = 1
    if cost_type == 'l2':
        list_loss = []
        k = -1
        for i in range(f_scaled.shape[0]):
            if i in intervene_feats:
                k += 1
                list_loss.append(
                    weights[k] *
                    action_tensor[i] *
                    F.mse_loss(f_scaled[i], cf_scaled[i], reduction='none') / (
                                torch.max(train_data_scaled[:, i]) - torch.min(train_data_scaled[:, i]))
                )
        loss = torch.stack(list_loss)
        loss = torch.mean(loss)
        return loss
    elif cost_type == 'effort':
        raise NotImplementedError
    else:
        raise NotImplementedError


def compute_optimal_action_set(training_data, intervene_features, feature_weights, data_f, clf_mod, grid_search_bins=10,
                               max_interv_cardinality=1000, cost_type='l2'):
    valid_action_sets = get_valid_discretized_action_sets(training_data=training_data, data_sub=data_f,
                                                          intervene_idxs=intervene_features,
                                                          grid_search_bins=grid_search_bins,
                                                          max_intervention_cardinality=max_interv_cardinality)
    min_cost = np.infty
    min_cost_action_set = {}
    best_pred_prob = 0.0
    for action_set in valid_action_sets:
        cf_data, prob_pos = is_point_constraint_satisfied(data_mod=preparator, clf_model=clf_mod,
                                                          data_f=data_f, action_set=action_set)
        if prob_pos >= 0.5:
            cost_of_action_set = measure_cost(data_f=data_f, cf_data=cf_data, training_data=training_data,
                                              scaler=preparator.scaler, intervene_feats=intervene_features,
                                              action_set=action_set, cost_type=cost_type, weights=feature_weights)
            if cost_of_action_set < min_cost:
                min_cost = cost_of_action_set
                min_cost_action_set = action_set
                best_pred_prob = prob_pos
    return min_cost_action_set, min_cost, best_pred_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-cfg', type=str, help='Config file')
    parser.add_argument('--clf_path', '-clf', type=str, help='Path to trained clf')
    parser.add_argument('--features_intervene', '-f', type=str, default="1,2,3")
    parser.add_argument('--bins', '-b', type=int, default=15)
    parser.add_argument('--cost_fn', type=str, choices=['l2', 'effort'], default='l2')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--feat_weights', type=str, default="1,1,1")

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    sys.path.append('./causal_flows/')

    args = parser.parse_args()
    args.features_intervene = list(int(i) for i in str(args.features_intervene).split(','))
    args.feat_weights = list(float(i) for i in str(args.feat_weights).split(','))
    assert len(args.features_intervene) == len(args.feat_weights)
    cfg = argtools.parse_args(args.config_file)
    cfg = argtools.to_yacs(CN(), cfg)

    # cfg.dataset.sem_name = args.eq_type
    yaml_files = glob.glob(os.path.join(args.clf_path, 'params_clf.pkl'))
    with open(yaml_files[0], 'rb') as f:
        clf_cfg = pickle.load(f)
    cfg.classifier = clf_cfg.classifier
    if len(args.out_dir) > 0:
        cfg.root_dir = args.out_dir
    cfg.verbose = args.verbose
    verbose = args.verbose

    clf_ckpt = os.path.join(args.clf_path, 'classifier.pt')

    path = Path(cfg.root_dir)
    path.mkdir(parents=True, exist_ok=True)

    preparator = SCMPreparator.loader(cfg.dataset)
    preparator.prepare_data()

    if preparator.scaler is None:
        preparator.get_scaler(fit=True)

    num_nodes = preparator.num_nodes
    feat_stats_array = feat_stats(preparator)
    cfg.reward_util_array = feat_stats_array

    clf_params = cfg.classifier
    trained_clf_model = load_clf(ckpt_file=clf_ckpt, cfg=clf_params)

    validate_clf(trained_clf_model, preparator, verbose=args.verbose)
    filter_datasets_by_label(preparator, trained_clf_model)

    data_path = f"{cfg.dataset.name}_{cfg.dataset.sem_name}"
    save_path = os.path.join(cfg.root_dir, data_path)
    clf_spec_path = f"{trained_clf_model.model_type}_{trained_clf_model.fairness_type}"
    save_path = os.path.join(save_path, clf_spec_path)

    os.makedirs(save_path, exist_ok=True)

    save_yaml_loc = os.path.join(save_path, 'oracle_test_results.yaml')

    data, train_data = preparator.datasets[2].X, preparator.datasets[0].X
    optimal_action_set_list = []
    optimal_cost_list = []
    pred_prob_list = []
    time_list = []
    bins = args.bins
    cost = args.cost_fn

    pbar = tqdm(range(data.shape[0]))

    for idx in pbar:
        factual_data = data[idx, :]
        with torch.no_grad():
            init_prob_pred = trained_clf_model(preparator.scaler.transform(factual_data))[1].item()
        if init_prob_pred < 0.5:
            init_time = time.time()
            optimal_action_set, optimal_cost, prob_pos = compute_optimal_action_set(
                train_data, args.features_intervene, args.feat_weights, factual_data,
                trained_clf_model, cost_type=cost, grid_search_bins=bins)
            final_time = time.time()
            optimal_action_set_list.append(optimal_action_set)
            optimal_cost_list.append(optimal_cost)
            pred_prob_list.append(prob_pos)
            time_list.append(final_time - init_time)
            pbar.set_postfix({'cost': np.round(optimal_cost, 5).item()})
        else:
            continue

    optimal_cost_list = np.array(optimal_cost_list)
    optimal_action_set_list = np.array(optimal_action_set_list)
    results_dict = {
        'validity': np.mean(optimal_cost_list != np.infty).item(),
        'cost': np.mean(optimal_cost_list).item(),
        'num_actions': np.mean([len(x) for x in optimal_action_set_list]).item(),
        'pred_prob': np.mean(pred_prob_list).item(),
        'num_intervened': len(time_list),
        'time_total': np.sum(time_list).item(),
        'time_avg': np.mean(time_list).item()
    }
    if verbose:
        print(f"Results: {results_dict}")
    with open(save_yaml_loc, 'w') as yaml_file:
        yaml.dump(results_dict, yaml_file, default_flow_style=False)
    save_res_loc = os.path.join(save_path, 'oracle_bf_per_datum_acts.pkl')
    with open(save_res_loc, 'wb') as pkl_file:
        pickle.dump(optimal_action_set_list, pkl_file)
