import os
import pickle
import sys
import warnings
from recourse_utils.clf_trainer import *
from recourse_utils.main_helpers import *
from causal_flows.causal_nf.preparators.scm import SCMPreparator
from pathlib import Path
from yacs.config import CfgNode as CN
import recourse_utils.args_parser as argtools
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file", "--config_file", type=str, help="Configuration_file")
    parser.add_argument('-s', '--seed', default=42, type=int, help='set random seed, default: random')
    parser.add_argument('-dir', '--root_dir', default='', type=str, help='directory for storing results')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('-data_only', '--generate_data_only', default=False, action="store_true",
                        help='if true, exit after data generation')
    parser.add_argument('--no_save', action='store_true', default=False)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    sys.path.append('./causal_flows/')

    args = parser.parse_args()
    cfg = argtools.parse_args(args.config_file)
    cfg = argtools.to_yacs(CN(), cfg)
    print(cfg)
    cfg.seed = args.seed
    if len(args.root_dir) > 0:
        cfg.root_dir = args.root_dir
    cfg.verbose = args.verbose

    verbose = args.verbose
    seed = cfg.seed
    seed_everything(seed)

    preparator = SCMPreparator.loader(cfg.dataset)
    preparator.prepare_data()

    trained_clf_model = train_clf(preparator, cfg, verbose, clf_obj=None)

    path_save = os.path.join(cfg.root_dir, f"{preparator.name}_{preparator.sem_name}",
                             f"{trained_clf_model.model_type}")
    fair_type = cfg.classifier.params.fairness
    f_lmbd = cfg.classifier.params.lmbd
    if fair_type is None or fair_type.lower() == 'none' or f_lmbd is None:
        f_path = 'none'
    else:
        f_path = f"{fair_type}_{f_lmbd}"
    path_save = os.path.join(path_save, f_path)
    Path(path_save).mkdir(parents=True, exist_ok=True)
    if not args.no_save:
        torch.save(trained_clf_model.clf_model, os.path.join(path_save, 'classifier.pt'))
        # argtools.save_yaml(cfg, file_path=os.path.join(path_save, 'params_clf.pkl'))
        with open(os.path.join(path_save, 'params_clf.pkl'), 'wb') as f:
            pickle.dump(cfg, f)
        print("Saved ckpt to", os.path.join(path_save, 'classifier.pt'))

    cfg.classifier.input_dim = preparator.datasets[0].X.shape[1]
    if not args.no_save:
        save_to = path_save
    else:
        save_to = None
    validate_clf(trained_clf_model, preparator, verbose, save_to)
