import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode as CN
import recourse_utils.args_parser as argtools
from causal_flows.causal_nf.preparators.scm import SCMPreparator
from recourse_utils.clf_trainer import load_clf, validate_clf
from recourse_utils.main_helpers import filter_datasets_by_label


def compute_counterfactual(scm, x_factual, indices, values):
    # TODO: CHECK THROUGH
    u = scm.transform.inv(x_factual)
    for index, value in zip(indices, values):
        scm.intervene(index, value)
    x_cf = scm.transform(u)
    for index in indices:
        scm.stop_intervening(index)
    return x_cf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-cfg', type=str, help='Config file')
    parser.add_argument('--clf_path', '-clf', type=str, help='Path to trained clf')
    parser.add_argument('--res_path', '-res', type=str, help='Path to per-datum results dir (give entire tree).')
    sys.path.append('./causal_flows/')
    args = parser.parse_args()
    cfg = argtools.parse_args(args.config_file)
    cfg = argtools.to_yacs(CN(), cfg)
    yaml_files = glob.glob(os.path.join(args.clf_path, 'params_clf.pkl'))
    with open(yaml_files[0], 'rb') as f:
        clf_cfg = pickle.load(f)

    clf_ckpt = os.path.join(args.clf_path, 'classifier.pt')

    path = Path(cfg.root_dir)
    path.mkdir(parents=True, exist_ok=True)

    preparator = SCMPreparator.loader(cfg.dataset)
    preparator.prepare_data()

    if preparator.scaler is None:
        preparator.get_scaler(fit=True)

    trained_clf_model = load_clf(ckpt_file=clf_ckpt, cfg=clf_cfg.classifier)
    filter_datasets_by_label(preparator, trained_clf_model)

    data = preparator.datasets[2].X

    pbar = tqdm(range(data.shape[0]))

    with open(os.path.join(args.res_path, 'carma_per_datum_acts.pkl'), 'rb') as f:
        per_dat = pickle.load(f)

    lst = []
    for idx in pbar:
        factual_data = data[idx, :]
        for k in per_dat.keys():
            act = per_dat[k][idx]
            indices, values = list(act.keys()), list(act.values())
            data_f = factual_data.view(1, -1)
            scm_cf = compute_counterfactual(preparator.scm, data_f, indices, values)
            scaled_data = preparator.scaler.transform(scm_cf)
            prob_pred = trained_clf_model(scaled_data)[1].item()
            lst.append(prob_pred > 0.5)
    print(np.mean(lst), np.std(lst))
