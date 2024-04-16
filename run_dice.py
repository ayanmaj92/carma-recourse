import argparse
import concurrent.futures
import os
import pickle
import sys
import warnings
from pathlib import Path
import recourse_utils.args_parser as argtools
import optuna
import yaml

from causal_flows.causal_nf.preparators.scm import SCMPreparator
from recourse_utils.clf_trainer import *
from recourse_utils.constants import Cte
from recourse_utils.dice_vae_cfe import DiceVAECFExplainer
from recourse_utils.main_helpers import *
from yacs.config import CfgNode as CN
from recourse_utils.recourse_reward_utils import feat_stats


def build_model(params, cfg_, data_module, clf_mod, seed_):
    seed_everything(seed_)
    if params is not None:
        # Otherwise, there are default values in my yaml that will be used
        cfg_.dice_vae.params.optim_lr = params['learning_rate']
        h_dim_list = [params['nodes'] for _ in range(params['n_layers'])]
        cfg_.dice_vae.params.enc_hdim_list = h_dim_list
        cfg_.dice_vae.params.act = Cte.RELU
        cfg_.dice_vae.params.max_epochs = params['epochs']
        cfg_.dice_vae.params.batch_size = params['batch_size']
        cfg_.dice_vae.params.hinge_margin = params['hinge_margin']
    dice_explainer = DiceVAECFExplainer(cfg_, data_module, clf_mod, seed_)
    return dice_explainer


def run(params, cfg_, data_module, clf_mod, seed_):
    explainer = build_model(params, cfg_, data_module, clf_mod, seed_)
    # Train
    explainer.setup_data()
    if cfg_.verbose:
        pbar, mod_sum = True, True
    else:
        pbar, mod_sum = False, False
    vae_trainer = pl.Trainer(gpus=None,
                             max_epochs=cfg_.dice_vae.params.max_epochs,
                             check_val_every_n_epoch=cfg_.dice_vae.params.max_epochs,
                             enable_progress_bar=pbar,
                             enable_model_summary=mod_sum,
                             logger=False,
                             enable_checkpointing=False
                             )
    train_loader = DataLoader(explainer.train_data, batch_size=explainer.batch_size)
    val_loader = DataLoader(explainer.val_data, batch_size=explainer.batch_size, shuffle=False)
    vae_trainer.fit(explainer.dice_vae_model, train_loader, val_loader)
    if cfg_.hparam_tune:
        res_tup = \
            explainer.get_cost_preds_from_data(explainer.val_data, save_yaml_loc='', return_results=True)
        recourse_cost, recourse_success = res_tup[0], res_tup[1]
        return recourse_cost, recourse_success
    else:
        res_tup = \
            explainer.get_cost_preds_from_data(explainer.test_data, save_yaml_loc='', return_results=True,
                                               per_datum=True)
        recourse_cost, recourse_success = res_tup[0], res_tup[1]
        recourse_pred_prob, recourse_time = res_tup[2], res_tup[3]
        action_list = res_tup[4]
        return recourse_cost, recourse_success, recourse_pred_prob, recourse_time, action_list


def objective(trial):
    global cfg, params_history, args, seeds_list
    if cfg.hparam_tune:
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate',
                                                       [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'nodes': trial.suggest_categorical('nodes', [num_nodes, 8, 16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 100, 500, step=50),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256]),
            'hinge_margin': trial.suggest_float('hinge_margin', 0.001, 0.2, step=0.001)
        }
        am_k = tuple(params.values())
        if am_k in params_history:
            return params_history[am_k]
        mean_val_dict = {'cost': [], 'success': []}
        if args.num_procs == 1 or args.num_runs == 1:
            for seed_ in seeds_list:
                r = run(params, cfg, preparator, trained_clf_model, seed_)
                mean_val_dict['cost'].append(r[0])
                mean_val_dict['success'].append(r[1])
        else:
            with concurrent.futures.ProcessPoolExecutor(min(args.num_runs, args.num_procs)) as executor:
                f1 = executor.map(run, [params for _ in range(args.num_runs)],
                                  [cfg for _ in range(args.num_runs)],
                                  [preparator for _ in range(args.num_runs)],
                                  [trained_clf_model for _ in range(args.num_runs)], seeds_list)
                for r in f1:
                    mean_val_dict['cost'].append(r[0])
                    mean_val_dict['success'].append(r[1])
        params_history[am_k] = (np.mean(mean_val_dict['cost']), np.mean(mean_val_dict['success']))
        return np.mean(mean_val_dict['cost']), np.mean(mean_val_dict['success'])
    else:
        params = args.params
        print(params)
        mean_val_dict = {'cost': [], 'success': [], 'prob': [], 'times': []}
        actions_dict = {}
        for seed_ in seeds_list:
            r = run(params, cfg, preparator, trained_clf_model, seed_)
            mean_val_dict['cost'].append(r[0])
            mean_val_dict['success'].append(r[1])
            mean_val_dict['prob'].append(r[2])
            mean_val_dict['times'].append(r[3])
            actions_dict[seed_] = r[4]
        for k in mean_val_dict:
            mean_val_dict[k] = [np.mean(mean_val_dict[k]).item(), np.std(mean_val_dict[k]).item()]
        return mean_val_dict, actions_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-cfg', type=str, help='Config file')
    parser.add_argument('--clf_path', '-clf', type=str, help='Path to trained clf')
    parser.add_argument('--device', default='cpu', type=str, help='device to train on')
    parser.add_argument('--seed', '-s', default=100, type=int)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--run_mode', type=str, choices=['hparam', 'train', 'both'], default='both')
    # parser.add_argument('--hparam_tune', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('-par', '--params', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define model configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('--num_trials', type=int, default=200, help='Optuna number of trials')  # Only used for Optuna
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_procs', '-np', type=int, default=3)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    sys.path.append('./causal_flows/')

    args = parser.parse_args()
    cfg = argtools.parse_args(args.config_file)
    cfg = argtools.to_yacs(CN(), cfg)
    # Ensure any clf params are saved with this name.
    yaml_files = glob.glob(os.path.join(args.clf_path, 'params_clf.pkl'))
    with open(yaml_files[0], 'rb') as f:
        clf_cfg = pickle.load(f)
    cfg.classifier = clf_cfg.classifier
    if len(args.out_dir) > 0:
        cfg.root_dir = args.out_dir
    cfg.verbose = args.verbose
    verbose = args.verbose
    cfg.run_mode = args.run_mode
    seed = cfg.seed
    seed_everything(seed)

    # cfg.dataset.sem_name = args.eq_type
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

    params_history = {}

    clf_params = cfg.classifier
    trained_clf_model = load_clf(ckpt_file=clf_ckpt, cfg=clf_params)

    validate_clf(trained_clf_model, preparator, verbose=args.verbose)
    filter_datasets_by_label(preparator, trained_clf_model)

    pareto_best_trials = None
    seeds_list = list(np.arange(0, args.num_runs) * 1000 + 1000)

    data_path = f"{cfg.dataset.name}_{cfg.dataset.sem_name}"
    save_path = os.path.join(cfg.root_dir, data_path)
    clf_spec_path = f"{trained_clf_model.model_type}_{trained_clf_model.fairness_type}"
    save_path = os.path.join(save_path, clf_spec_path)
    os.makedirs(save_path, exist_ok=True)

    if cfg.run_mode == 'hparam' or cfg.run_mode == 'both':
        cfg.hparam_tune = True
        study = optuna.create_study(directions=["minimize", "maximize"],
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=args.num_trials)
        pareto_best_trials = study.best_trials
        pkl_path = os.path.join(save_path, 'dice_optuna.pkl')
        for trial in pareto_best_trials:
            print("Cost, Validity:", trial.values, "Params:", trial.params)
        with open(pkl_path, 'wb') as f:
            pickle.dump(pareto_best_trials, f)

    cfg.hparam_tune = False

    if cfg.run_mode == 'both':
        # Get the best params.
        assert pareto_best_trials is not None, "best trials is None, something went wrong!"
        all_res = [trial.values[0] + 1.0 * (1.0 - trial.values[1]) for trial in pareto_best_trials]
        idx = np.argsort(all_res)
        res_optuna = np.array(pareto_best_trials)
        best_params = res_optuna[idx][0].params
        # Set the best params to args.params
        args.params = best_params

    if cfg.run_mode == 'both' or cfg.run_mode == 'train':
        val_res_dict, actions_dict = objective(None)
        print(val_res_dict)
        save_yaml_loc = os.path.join(save_path, 'dice_test_results.yaml')
        with open(save_yaml_loc, 'w') as yaml_file:
            yaml.dump(val_res_dict, yaml_file, default_flow_style=False)
        save_cfg_loc = os.path.join(save_path, 'params_dice.pkl')
        with open(save_cfg_loc, 'wb') as pkl_file:
            pickle.dump(cfg, pkl_file)
        save_acts_loc = os.path.join(save_path, 'actions_list.pkl')
        with open(save_acts_loc, 'wb') as pkl_file:
            pickle.dump(actions_dict, pkl_file)
