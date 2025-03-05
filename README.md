# CARMA: Causal Algorithmic Recourse with (Neural Network) Model-based Amortization

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

<img src="logo.png" alt="logo" width="100"/>

This repository contains the code implementing the CARMA approach, a practical framework for generating recommendations for causal algorithmic recourse at scale. 
CARMA was introduced in the paper `CARMA: A practical framework to generate recommendations for causal algorithmic recourse at scale` and was accepted at the [ACM FAccT 2024](https://facctconference.org/2024/) conference. Check out the full paper at [this link](https://dl.acm.org/doi/10.1145/3630106.3659003).
## Overview
1. [Setup](#setup)
2. [Training downstream decision-making classifier](#clf)
3. [Hyperparameter tuning causal generative model](#hparam_cgm)
4. [Running causal generative model](#run_cgm)
5. [Using CARMA](#carma)
6. [Running baseline recourse methods](#run_base)
7. [Pointers for adding new datasets](#new_data)

<a name="setup"></a>
## Setup 
Create the conda environment.
```bash
conda create --name carma python=3.9.12 --no-default-packages
conda activate carma
```
Install torch related things.
```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

pip install torch_geometric==2.3.1
pip install torch-scatter==2.1.1

pip install -r requirements.txt
```

### Note
Required parameter files can be found in the `_params/` folder. Please follow the format of these files to expand to other datasets and methods.

<a name="clf"></a>
## Training downstream decision-making classifier 
The script to train downstream decision-making classifier models is `run_clf.py`. Note that our current method only supports differentiable models, so we use logistic regression and neural network classification models.
To train a model:
```bash
python run_clf.py --config_file _params/_clf/loan.yaml --seed 42 --verbose
```
The dataset is created/loaded from the location that is provided in the yaml. For instance:
```yaml
dataset:
  root: New_Data/clf
  name: loan
  sem_name: non-linear # non-linear #non-additive
  splits: [ 0.5,0.25,0.25 ]
  k_fold: 1
  shuffle_train: True
  single_split: False
  loss: default
  scale: default
  num_samples: 20000
  base_version: 1
  add_noise: False
  type: torch
  use_edge_attr: False
```
The model checkpoints and params are saved in the parameter set in the config file:
```yaml
root_dir: _models_trained/clf
```

<a name="hparam_cgm"></a>
## Hyperparameter tuning causal generative model
Hyperparameter tuning causal generative models like the causal flows or VACA follows these steps. Note that this is based on the [original causal flows code](https://github.com/psanch21/causal-flows).
### a. Create grid params for grid search
Under `_params/_grid/<causal_generative_model_name>/` we have provided a basic structure.
The `base.yaml` file provides the hyperparameter values to loop over. Then, `base_linear.yaml` and `base_non_linear.yaml` files list the names of the datasets and the structural equation formats to run tuning for.

For example, `_params/_grid/causal_nf/base_non_linear.yaml`:
```yaml
dataset:
  name: [trianglesens, collidersens, chainsens, loan]
  sem_name: [non-linear]
```

Keep the `base.yaml` files fixed, and simply add new data names into the `base_linear.yaml` or `base_non_linear.yaml` files.
### b. Running grid search
To run grid search on the specified grid space, use `causal_flows/generate_jobs.py` script. Give help to see options:
```bash
python causal_flows/generate_jobs.py --help
```
For instance, one can run it as:
```bash
python causal_flows/generate_jobs.py --grid_file _params/_grid/causal_nf/base.yaml --format shell --jobs_per_file 20000 --batch_size 500 --wandb_mode disabled
```
This will create the scripts to run in a folder like `_params/_grid/causal_nf/base/jobs_sh/`. We will need to run all files without `_test` in the name.
All results are saved into the directory defined in the `base.yaml` file, e.g., for causal flows `causal_nf/`:
```yaml
root_dir: [ hparam_grid_causal_mods/comparison_causal_nf ]
```
### c. Analyzing grid search outputs
To analyze the grid search outcome, run the following scripts:
```bash
# For causal flows
python causal_flows/scripts/create_comparison_flows.py hparam_grid_causal_mods/comparison_causal_nf/
# For VACA
python causal_flows/scripts/create_comparison_vaca.py hparam_grid_causal_mods/comparison_vaca/
```
This will print out the best hyperparameters on the console.

<a name="run_cgm"></a>
## Running causal generative model
### a. Make param file
First, we need to create a param file for the causal generative model we want to run based on the dataset we want to run it on.
This is also the file where we need to *manually* set the best hyperparameters we found.
One can find the param files we have used for our datasets in `_params/_causal_models/` path.
For instance, for causal flows on loan data, the file `loan_causal_nf.yaml` has the hyperparameter set as:
```yaml
model:
  dim_inner: [64]
```
The corresponding data is created/loaded from the path we provide in the YAML file. For instance:
```yaml
dataset:
  root: New_Data/causal_estimator
  name: loan
  sem_name: non-linear # non-linear #non-additive
  splits: [ 0.8,0.1,0.1 ]
  k_fold: 1
  shuffle_train: True
  loss: default
  scale: default
  num_samples: 25000
  base_version: 1
```
**IMPORTANT**: For VACA, the data format needs to be different since it uses Graph Neural Nets:
```yaml
dataset:
  type: pyg
  use_edge_attr: True
```
These two lines **must be added** to all parameter files for VACA.

Note that some default parameters are taken from the file `default_config.yaml`.
These structures are adopted from the original Flows [code](https://github.com/psanch21/causal-flows).
### b. Train causal generative model
Train the desired model by running the script `run_causal_estimator.py`. Use the help option for more details:
```bash
python run_causal_estimator.py --help
```
For instance, for the loan data, we can train the causal flows model using:
```bash
python run_causal_estimator.py --config_file _params/_causal_models/loan_causal_nf.yaml --wandb_mode disabled --project CAUSAL_NF
```
Again, we can set the checkpoint saving location in the param file, e.g., 
```yaml
root_dir: _models_trained/vaca/
```

<a name="carma"></a>
## Using CARMA

### a. Param files
Similar to the classifier and causal generative models, create a YAML parameter file for the CARMA model that needs to be trained for the particular dataset.
For an example, see `_params/_recourse/loan_nonlin_carma.yaml`.

The saving directory, where all checkpoints and results are saved, is provided as:
```yaml
root_dir: carma_results/causal_nf/
```

The dataset configurations are provided as:
```yaml
dataset:
  root: New_Data/recourse
  name: loan
  sem_name: non-linear
  splits: [ 0.5,0.25,0.25 ]
  k_fold: 1
  shuffle_train: True
  num_samples: 20000
  base_version: 1
  type: torch
  ....
```

As stated above, to use VACA in CARMA, the following lines need to be added under `dataset:`.
```yaml
dataset:
  ....
  type: pyg
  use_edge_attr: True
```

Finally, CARMA parameters are provided as:
```yaml
carma:
  name: 'carma'
  params:
    features_to_intervene: [2,3,5,6]
    mask_nw_nodes: 32
    mask_nw_n_layers: 2
    action_nw_nodes: 32
    action_nw_n_layers: 2
    act: relu
    ....
```

### b. Hyperparameter tune
To tune the hyperparameters of CARMA, we need to use the particular flag `--run_mode hparam` when running `main.py`.
```bash
python main.py --config_file _params/_recourse/loan_nonlin_carma.yaml \
--clf_path _models_trained/clf/loan_non-linear/nn/none/ \
--causal_model_path _models_trained/causal_nf/loan_non-linear/ \
--out_dir carma_hparam/causal_nf/ --run_mode hparam --num_trials 50 --verbose --num_runs 5 --num_procs 1
```
`num_trials` indicates how many trials Optuna should run to find the best hyperparameters, `num_runs` is the number of seeds, `num_procs` is the number of parallel processes to speed-up if necessary and available.

Once the hyperparameter tuning is done, run `_analyze_helpers/analyze_optuna_result.py` giving the path of the `.pkl` result file (gets saved in the path given in `--out_dir` above).
This gives the Pareto-optimal runs, giving `0` as input index will give the parameters of the run that had the best run in terms of *cost and validity*.
If the cost, or the validity is more important in selecting the parameters, select other index.

### c. Train models
Once the best parameters are obtained, these can be provided when calling `main.py`. It is important to note that we now use `--run_mode train`. One example is as shown here:
```bash
python main.py --config_file _params/_recourse/loan_nonlin_carma.yaml \
--clf_path _models_trained/clf/loan_non-linear/nn/none/ \
--causal_model_path _models_trained/causal_nf/loan_non-linear/ \
--out_dir carma_optimal_results/ --run_mode train --verbose --num_runs 10 --num_procs 1 -m 2 \
--params learning_rate=0.001+action_nw_n_layers=4+action_nw_nodes=32+mask_nw_n_layers=3+mask_nw_nodes=8+epochs=350+batch_size=64+hinge_margin=0.019+tau=0.21
```
The parameters are provided using the `--params` flag, using the format `--params param1=value1+param2=value2+...`.

Note: If desired, we can run both hyperparameter tuning and training using `--run_mode both`, this will select the parameters at position `0` in the list of Pareto-optimal hyperparameters.

### d. CARMA with recourse feature preferences

To input feature preferences, use the flags as for example, `--feat_weights 3,2,2,1 --custom_prior`.
The features weights can give the weight for each feature **as they appear in the dataset** (the order of the features and the structural equations defined).
We **always assume** that the easiest features have weight 1, and all other *more difficult* features have weights higher than 1. 
The `--custom_prior` changes the prior for the Bernoulli used in the mask network. The custom prior uses the inverse of the provided `feat_weights` values. 


```bash
python main.py --config_file _params/_recourse/loan_nonlin_carma.yaml \
--clf_path _models_trained/clf/loan_non-linear/nn/none/ \
--causal_model_path _models_trained/causal_nf/loan_non-linear/ \
--out_dir loan_priored_weighted_carma_optimal/ --run_mode train --verbose --num_runs 10 --num_procs 1 -m 2 --feat_weights 3,2,2,1 --custom_prior \
--params learning_rate=0.001+action_nw_n_layers=3+action_nw_nodes=32+mask_nw_n_layers=3+mask_nw_nodes=16+epochs=250+batch_size=64+hinge_margin=0.005+tau=0.1
```

<a name="run_base"></a>
## Running baseline recourse methods
In the paper, we utilize two baseline recourse methods: a. Unamortized causal oracle, b. Amortized non-causal DiCE-VAE (Mahajan et al. 2019).
### a. Causal oracle
```bash
python run_brute_force_oracle.py --config_file _params/_recourse/loan_nonlin_oracle.yaml --clf_path _models_trained/clf/loan_non-linear/nn/none/ --bins 15 --cost_fn l2 --features_intervene 2,3,5,6 --verbose
```
Note that like other methods, the oracle also needs its own YAML parameter file, a pre-trained classifier. We also define the number of bins for discretization for the grid search function, and the features to intervene on.

### b. Non-causal DiCE-VAE
```bash
python run_dice.py --config_file _params/_recourse/loan_nonlin_dice.yaml \
--clf_path _models_trained/clf/loan_non-linear/nn/none/ \
--out_dir dice_optimal_results/ --run_mode train --verbose --num_runs 10 --num_procs 1 \
--params learning_rate=5e-05+n_layers=1+nodes=64+epochs=300+batch_size=128+hinge_margin=0.072
```
Again, we need a proper parameter YAML file. We can also perform hyperparameter tuning and then training like CARMA. Please follow similar steps as CARMA. The example here shows training when we have some specific hyperparameters.

<a name="new_data"></a>


## Pointers for adding new datasets
To add new datasets, follow the structure of the core `causal_flows` repository code.
Specifically, one needs to add the structural equations in `causal_flows/causal_nf/sem_equations/`. See `collidersens.py` or `loan.py` for the form of the script.
Next, to define the *exogenous noise factors* for the data, add corresponding code defining the distributions in `causal_flows/causal_nf/preparators/scm/_base_distributions.py`. See the functions `base_distribution_4_nodes_sens` or `base_distribution_7_nodes_sens` for guidance.
Add these to the `pu_dict_sens` dictionary at the end of the `_base_distributions.py` file.
Next, in `causal_flows/causal_nf/preparators/scm/scm_preparator.py`, in the `__init__` of `SCMPreparator`, add the link to the base noise distribution in this code:
```python
if 'sens' in name.lower() or name.lower() == 'loan':
    # Override the base version for our datasets to ensure we are never wrong!
    if name == 'trianglesens' and sem_name == 'linear':
        base_version = 1
    elif name == 'collidersens' and sem_name == 'linear':
        base_version = 1
    elif name == 'chainsens' and sem_name == 'linear':
        base_version = 1
    elif name == 'chainsens' and sem_name == 'non-linear':
        base_version = 1
    elif name == 'trianglesens' and sem_name == 'non-linear':
        base_version = 2
    elif name == 'collidersens' and sem_name == 'non-linear':
        base_version = 2
    elif name == 'loan':
        base_version = 1
    base_distribution = pu_dict_sens[self.num_nodes](base_version)
```

### Analysis helpers
Analyzing some of the results, e.g., the feature interventions, the Optuna hyperparameter tuning, and the causal validity of the proposed interventions, can be found in `_analyze_helpers/`.

## Contact and citation
For any questions, please contact `ayanm{at}mpi-sws.org`. For citation, please cite our [original paper](https://dl.acm.org/doi/10.1145/3630106.3659003).

## Acknowledgements
The code on VACA and Causal Flows is based on [this repo](https://github.com/psanch21/causal-flows).
A huge thanks to [Adrián](https://github.com/adrianjav/) and [Pablo](https://github.com/psanch21) for providing guidance on their code!
