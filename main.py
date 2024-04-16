import argparse
import concurrent.futures
import glob
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
import optuna
import yaml
from tqdm import tqdm

from causal_flows.causal_nf.preparators.scm import SCMPreparator
import causal_flows.causal_nf.utils.training as causal_nf_train
from recourse_model.loss import lagrangian
from recourse_utils.clf_trainer import load_clf, validate_clf
from recourse_utils.main_helpers import *
from yacs.config import CfgNode as CN

from recourse_utils.recourse_optimizer import create_optimizer
from recourse_utils.recourse_reward_utils import feat_stats
from recourse_model.model import *
import recourse_utils.args_parser as argtools


def build_model(cfg_, prepper, z_clip, params, clf, causal_mod, seed_val, action_model_type):
    seed_everything(seed_val)
    if cfg_.causal_model.model.name == 'vaca':
        z_dim = causal_mod.z_dim
    elif cfg_.causal_model.model.name == 'causal_nf':
        z_dim = 1
    else:
        raise NotImplementedError
    num_units = z_dim * prepper.num_nodes

    if params is not None:
        cfg_.carma.params.action_nw_hdim_list = [params['action_nw_nodes'] for _ in range(params['action_nw_n_layers'])]
        cfg_.carma.params.batch_size = params['batch_size']
        cfg_.carma.params.mask_nw_hdim_list = [params['mask_nw_nodes'] for _ in range(params['mask_nw_n_layers'])]
        cfg_.carma.params.mask_nw_hdim_list.insert(0, num_units)
        cfg_.carma.params.optim_lr = params['learning_rate']
        cfg_.carma.params.optim_max_epochs = params['epochs']
        cfg_.carma.params.hinge_margin = params['hinge_margin']
        cfg_.carma.params.tau = params['tau']
    else:
        cfg_.carma.params.action_nw_hdim_list = [cfg_.carma.params.action_nw_nodes
                                                 for _ in range(cfg_.carma.params.action_nw_n_layers)]
        cfg_.carma.params.mask_nw_hdim_list = [cfg_.carma.params.mask_nw_nodes
                                               for _ in range(cfg_.carma.params.mask_nw_n_layers)]
        cfg_.carma.params.mask_nw_hdim_list.insert(0, num_units)
    act = cfg_.carma.params.act

    # The action network that suggests the action intervention values

    action_model = create_action_model(h_dim_list=cfg_.carma.params.action_nw_hdim_list,
                                       z_dim=z_dim,
                                       act_name=act,
                                       bn=cfg_.carma.params.bn,
                                       drop_rate=cfg_.carma.params.drop_rate,
                                       num_nodes=prepper.num_nodes,
                                       clip_z=z_clip,
                                       model_type=action_model_type)
    action_model.to(torch.device(cfg_.device))

    # The mask network that suggests which features to act on
    mask_encoder_model = RecourseMaskEncoder(h_dim_list=cfg_.carma.params.mask_nw_hdim_list,
                                             num_nodes=prepper.num_nodes,
                                             act_name=act,
                                             device=cfg_.device)
    # The combined CARMA recourse model
    recourse_model = RecourseModel(predictor=action_model,
                                   encoder=mask_encoder_model,
                                   feat_to_intervene=cfg_.carma.params.features_to_intervene,
                                   causal_model=causal_mod,
                                   causal_model_name=cfg_.causal_model.model.name,
                                   clf_model=clf,
                                   down_caller='predictor',
                                   tau=cfg_.carma.params.tau)
    return recourse_model


def run(params, cfg_, prepper, z_clip, clf_mod, causal_mod, seed_val, action_model_type):
    recourse_model = build_model(cfg_, prepper, z_clip, params, clf_mod, causal_mod, seed_val, action_model_type)
    optimizer = create_optimizer(recourse_model.parameters(), cfg_)
    train_loader = prepper.get_dataloader_train(cfg_.carma.params.batch_size, shuffle=True)
    custom_prior = cfg_.custom_prior
    lmbd = 0.0
    using_tqdm = False
    if cfg_.hparam_tune:
        tbar = range(0, cfg_.carma.params.optim_max_epochs)
    else:
        if not cfg_.verbose:
            tbar = range(0, cfg_.carma.params.optim_max_epochs)
        else:
            using_tqdm = True
            tbar = tqdm(range(0, cfg_.carma.params.optim_max_epochs))
    feature_cdf_costs = cfg_.reward_util_array
    feat_to_intervene = cfg_.carma.params.features_to_intervene
    feat_weights = cfg_.feat_weights
    hinge_margin = cfg_.carma.params.hinge_margin
    # Perform training
    for cur_epoch in tbar:
        recourse_model.train()
        if using_tqdm:
            tbar.set_description(f'Epoch {cur_epoch}')
        for batch in train_loader:
            optimizer.zero_grad()
            if prepper.type == 'torch':
                # Restructure the batch.
                b_dict = {
                    'x': batch[0], 'u': batch[1], 'y': batch[2], 'z': batch[3]
                }
                batch = DotDict(b_dict)
            if prepper.type == 'torch':
                batch.x = prepper.scaler.transform(batch.x)
            else:
                old_shape = batch.x.shape[0]
                batch.x = prepper.scaler.transform(batch.x.reshape(batch.num_graphs, -1)).reshape(
                    old_shape, -1
                )
            policy_cf, recourse_prob, recourse_lab, action_tensor, phi_for_kl = recourse_model(batch)
            data = batch.x
            if prepper.type == 'pyg':
                # Need proper shaping
                data = data.view(batch.num_graphs, -1)
                policy_cf = policy_cf.view(batch.num_graphs, -1)
            loss, recourse_cost, loss_2 = lagrangian(data, policy_cf, recourse_prob, feature_cdf_costs,
                                                     action_tensor, phi_for_kl, feat_to_intervene, lmbd, hinge_margin,
                                                     feat_weights, custom_prior)
            loss.backward()
            lmbd = lmbd + 1 * loss_2.detach()
            lmbd = torch.clip(lmbd, min=0.0)
            optimizer.step()

    # Training completed. Now do validation or testing.
    sum_data = 0
    val_dict = {'cost': [], 'success': [], 'prob': [], 'acts': []}
    time_taken = 0.0
    if cfg_.hparam_tune:
        # In this case use validation set.
        loader = prepper.get_dataloaders(batch_size=cfg_.carma.params.batch_size)[1]
    else:
        # In this case use test set. Also do per datum to get proper time info.
        loader = prepper.get_dataloaders(batch_size=1)[2]
    # Storing per-datum things.
    per_datum_action_list = []
    with torch.no_grad():
        for batch in loader:
            recourse_model.eval()
            if prepper.type == 'torch':
                # Restructure the batch.
                b_dict = {
                    'x': batch[0], 'u': batch[1], 'y': batch[2], 'z': batch[3]
                }
                batch = DotDict(b_dict)
            if prepper.type == 'torch':
                batch.x = prepper.scaler.transform(batch.x)
            else:
                old_shape = batch.x.shape[0]
                batch.x = prepper.scaler.transform(batch.x.reshape(batch.num_graphs, -1)).reshape(
                    old_shape, -1
                )
            if recourse_model.causal_model_name == 'causal_nf':
                batch_size = batch.x.shape[0]
            else:
                batch_size = batch.num_graphs
            sum_data += batch_size
            init_time = time.time()
            policy_cf, recourse_prob, recourse_lab, action_tensor, phi_for_kl = recourse_model(batch, is_train=False)
            final_time = time.time()
            time_taken += (final_time - init_time)
            data = batch.x
            if prepper.type == 'pyg':
                # Need proper shaping
                data = data.view(batch.num_graphs, -1)
                policy_cf = policy_cf.view(batch.num_graphs, -1)
            loss, recourse_cost, _ = lagrangian(data, policy_cf, recourse_prob, feature_cdf_costs, action_tensor,
                                                phi_for_kl, feat_to_intervene, lmbd, hinge_margin, feat_weights,
                                                custom_prior)
            val_dict['cost'].append(recourse_cost.item() * batch_size)
            val_dict['success'].append(recourse_lab.sum())
            val_dict['acts'].append(action_tensor.sum(dim=1).sum().item())
            val_dict['prob'].append(recourse_prob.sum())
            # Store the action for this datum
            if not cfg_.hparam_tune:
                unscaled_cf = prepper.scaler.inverse_transform(policy_cf).flatten().detach().numpy()
                act_ = action_tensor.flatten().detach().tolist()
                action_dict = {ix: unscaled_cf[ix].item() for ix, m in enumerate(act_) if m == 1}
                per_datum_action_list.append(action_dict)
        val_dict = {k: np.sum(val_dict[k]) / sum_data for k in val_dict}
        val_dict['time_taken'] = time_taken
        if cfg_.hparam_tune:
            return val_dict
        else:
            return val_dict, per_datum_action_list, recourse_model


def objective(trial):
    global cfg, params_history, args, preparator, z_clipper, seeds_list

    if cfg.hparam_tune:
        if cfg.causal_model.model.name == 'vaca':
            action_nw_node_list = [16, 32, 64, 128]
            mask_nw_node_list = [16, 32, 64, 128]
        else:
            action_nw_node_list = [8, 16, 32]
            mask_nw_node_list = [8, 16, 32]
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate',
                                                       [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]),
            'action_nw_n_layers': trial.suggest_int('action_nw_n_layers', 2, 4),
            'action_nw_nodes': trial.suggest_categorical('action_nw_nodes', action_nw_node_list),
            'mask_nw_n_layers': trial.suggest_int('mask_nw_n_layers', 1, 3),
            'mask_nw_nodes': trial.suggest_categorical('mask_nw_nodes', mask_nw_node_list),
            'epochs': trial.suggest_int('epochs', 100, 500, step=50),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256]),
            'hinge_margin': trial.suggest_float('hinge_margin', 0.001, 0.02, step=0.001),
            'tau': trial.suggest_float('tau', 0.01, 1, step=0.01)
        }
        am_k = tuple(params.values())
        if am_k in params_history:
            return params_history[am_k]
        mean_val_dict = {'cost': [], 'success': []}
        if args.num_procs == 1 or args.num_runs == 1:
            for seed_ in seeds_list:
                r = run(params, cfg, preparator, z_clipper, trained_clf_model, causal_model,
                        seed_, args.action_model_type)
                mean_val_dict['cost'].append(r['cost'])
                mean_val_dict['success'].append(r['success'])
        else:
            with concurrent.futures.ProcessPoolExecutor(min(args.num_runs, args.num_procs)) as executor:
                f1 = executor.map(run, [params for _ in range(args.num_runs)], [cfg for _ in range(args.num_runs)],
                                  [preparator for _ in range(args.num_runs)],
                                  [z_clipper for _ in range(args.num_runs)],
                                  [trained_clf_model for _ in range(args.num_runs)],
                                  [causal_model for _ in range(args.num_runs)],
                                  seeds_list,
                                  [args.action_model_type for _ in range(args.num_runs)])
                for r in f1:
                    mean_val_dict['cost'].append(r['cost'])
                    mean_val_dict['success'].append(r['success'])
        params_history[am_k] = (np.mean(mean_val_dict['cost']), np.mean(mean_val_dict['success']))
        return np.mean(mean_val_dict['cost']), np.mean(mean_val_dict['success'])
    else:
        params = args.params
        print(params)
        mean_val_dict = {'cost': [], 'success': [], 'acts': [], 'prob': [], 'times': []}
        mod_list = []
        per_datum_actions_results = {}
        for seed_ in seeds_list:
            r, per_datum_actions, recourse_mod = \
                run(params, cfg, preparator, z_clipper, trained_clf_model, causal_model, seed_, args.action_model_type)
            mean_val_dict['cost'].append(r['cost'])
            mean_val_dict['success'].append(r['success'])
            mean_val_dict['acts'].append(r['acts'])
            mean_val_dict['prob'].append(r['prob'])
            mean_val_dict['times'].append(r['time_taken'])
            mod_list.append(recourse_mod)
            per_datum_actions_results[int(seed_)] = per_datum_actions
        for k in mean_val_dict:
            mean_val_dict[k] = [np.mean(mean_val_dict[k]).item(), np.std(mean_val_dict[k]).item()]
        return mean_val_dict, per_datum_actions_results, mod_list


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    sys.path.append('./causal_flows/')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-cfg', type=str, help='config yaml file for CARMA')
    parser.add_argument('--clf_path', '-clf', type=str, help='path to trained classifier')
    parser.add_argument('--causal_model_path', '-cau', type=str, help='path to trained causal generative model')
    parser.add_argument('--out_dir', '-o', type=str, default='')
    parser.add_argument('--seed', '-s', type=int, default=100)
    parser.add_argument('--run_mode', type=str, choices=['hparam', 'train', 'both'], default='both')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('-par', '--params', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define model configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('--num_trials', type=int, default=200, help='Optuna number of trials')  # Only used for Optuna
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_procs', '-np', type=int, default=3)
    parser.add_argument('--action_model_type', '-m', type=int, default=2)
    parser.add_argument('--train_data_fraction', '-frac', type=float, default=1.0)
    parser.add_argument('--feat_weights', type=str, default='')
    parser.add_argument('--custom_prior', action='store_true', default=False)

    args = parser.parse_args()
    if args.feat_weights == '':
        args.feat_weights = None
    else:
        args.feat_weights = list(float(i) for i in str(args.feat_weights).split(','))
    cfg = argtools.parse_args(args.config_file)
    cfg = argtools.to_yacs(CN(), cfg)
    cfg.custom_prior = args.custom_prior
    if args.feat_weights is None:
        args.feat_weights = [1.0 for _ in range(len(cfg.carma.params.features_to_intervene))]

    # Ensure any clf params are saved with this name.
    yaml_files = glob.glob(os.path.join(args.clf_path, 'params_clf.pkl'))
    with open(yaml_files[0], 'rb') as f:
        clf_cfg = pickle.load(f)
    cfg.classifier = clf_cfg.classifier
    # Ensure any causal_gen_model params are saved with this name.
    yaml_files = glob.glob(os.path.join(args.causal_model_path, 'params_causal_model.pkl'))
    with open(yaml_files[0], 'rb') as f:
        causal_model_cfg = pickle.load(f)
    cfg.causal_model = CN()
    cfg.causal_model.model = causal_model_cfg.model
    cfg.causal_model.train = causal_model_cfg.train
    cfg.causal_model.optim = causal_model_cfg.optim
    if cfg.causal_model.model.name == 'vaca':
        # These are extra params that VACA need.
        cfg.causal_model.gnn = causal_model_cfg.gnn
        cfg.causal_model.gnn2 = causal_model_cfg.gnn2

    # cfg.dataset.sem_name = args.eq_type
    if cfg.causal_model.model.name == 'vaca':
        if cfg.dataset.type != 'pyg':
            print('Dataset type for VACA should be pyg. Changing it.')
            cfg.dataset.type = 'pyg'

    if len(args.out_dir) > 0:
        cfg.root_dir = args.out_dir
    cfg.verbose = args.verbose
    verbose = args.verbose
    seed = cfg.seed
    seed_everything(seed)

    cfg.run_mode = args.run_mode

    path = Path(cfg.root_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Make the data preparator and compute some stats
    preparator = SCMPreparator.loader(cfg.dataset)
    preparator.prepare_data()
    if preparator.scaler is None:
        preparator.get_scaler(fit=True)

    num_nodes = preparator.num_nodes
    feat_stats_array = feat_stats(preparator)
    cfg.reward_util_array = feat_stats_array

    # Load classifier and filter data.
    clf_params = cfg.classifier
    clf_ckpt = os.path.join(args.clf_path, 'classifier.pt')
    trained_clf_model = load_clf(ckpt_file=clf_ckpt, cfg=clf_params)
    validate_clf(trained_clf_model, preparator, verbose=args.verbose)
    filter_datasets_by_label(preparator, trained_clf_model, args.train_data_fraction)

    # Load the causal generative model
    causal_model_ckpt = os.path.join(args.causal_model_path, 'last.ckpt')
    cfg.causal_model.device = cfg.device
    causal_trainer = causal_nf_train.load_model(
        cfg=cfg.causal_model, preparator=preparator, ckpt_file=causal_model_ckpt
    )
    causal_model = causal_trainer.model
    causal_model_param_count = causal_trainer.param_count()
    if verbose:
        print(f"Param count of causal generative model: {causal_model_param_count}")

    # Set all params of pre-trained models to no-grad
    for param in trained_clf_model.clf_model.parameters():
        param.requires_grad = False
    trained_clf_model.clf_model.eval()
    for param in causal_model.parameters():
        param.requires_grad = False
    causal_model.eval()

    # Set the z of data. Shape (batch_size x num_feats x z_dim) where z_dim=1 for causal_nf
    set_z_data(preparator, causal_model, cfg.causal_model.model.name)
    z_clipper = (preparator.datasets[0].Z.min().floor(),
                 preparator.datasets[0].Z.max().ceil())
    if verbose:
        print(f"Latent z-clipper: {z_clipper}")

    cfg.feat_weights = args.feat_weights

    data_path = f"{cfg.dataset.name}_{cfg.dataset.sem_name}"
    if args.train_data_fraction != 1.0:
        data_path += f"_{args.train_data_fraction}"
    save_path = os.path.join(cfg.root_dir, data_path)
    clf_spec_path = f"{trained_clf_model.model_type}_{trained_clf_model.fairness_type}"
    causal_model_spec = f"{cfg.causal_model.model.name}"
    save_path = os.path.join(save_path, clf_spec_path, causal_model_spec)
    os.makedirs(save_path, exist_ok=True)
    params_history = {}
    pareto_best_trials = None
    seeds_list = list(np.arange(0, args.num_runs) * 1000 + 1000)
    if cfg.run_mode == 'both' or cfg.run_mode == 'hparam':
        cfg.hparam_tune = True
        study = optuna.create_study(directions=["minimize", "maximize"],
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=args.num_trials)
        pareto_best_trials = study.best_trials
        pkl_path = os.path.join(save_path, 'carma_optuna.pkl')
        if verbose:
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

    if cfg.run_mode == 'train' or cfg.run_mode == 'both':
        val_res_dict, per_datum_results, recourse_mod_list = objective(None)
        if verbose:
            print(val_res_dict)
        save_yaml_loc = os.path.join(save_path, 'carma_test_results.yaml')
        with open(save_yaml_loc, 'w') as yaml_file:
            yaml.dump(val_res_dict, yaml_file, default_flow_style=False)
        save_cfg_loc = os.path.join(save_path, 'params_carma.pkl')
        with open(save_cfg_loc, 'wb') as pkl_file:
            pickle.dump(cfg, pkl_file)
        save_res_loc = os.path.join(save_path, 'carma_per_datum_acts.pkl')
        with open(save_res_loc, 'wb') as pkl_file:
            pickle.dump(per_datum_results, pkl_file)
        for i in range(0, len(recourse_mod_list)):
            save_model_loc = os.path.join(save_path, f'carma_model_{seeds_list[i]}.pt')
            torch.save(recourse_mod_list[i], save_model_loc)
            save_model_state_loc = os.path.join(save_path, f'carma_model_state_{seeds_list[i]}.pt')
            torch.save(recourse_mod_list[i].state_dict(), save_model_state_loc)
