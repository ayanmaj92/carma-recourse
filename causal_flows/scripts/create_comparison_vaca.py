#
import sys
sys.path.append("../")
sys.path.append("./")

import os
import pandas as pd
import numpy as np
from causal_flows.notebooks import helpers as pb_help

# from tueplots import bundles

# plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update(figsizes.icml2022_full())
from causal_flows.scripts import helpers as script_help
from causal_flows.causal_nf.utils import dataframe as causal_nf_df
from causal_flows.causal_nf.utils import io as causal_nf_io
import warnings
import yaml


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

if len(sys.argv) == 1 or sys.argv[1] == '' or sys.argv[1] == ' ':
    root = "gridding_causal_mods"
else:
    root = sys.argv[1]

folder = os.path.join("results", "dataframes")
causal_nf_io.makedirs(folder, only_if_not_exists=True)
keep_cols = ["dataset__name", "dataset__sem_name", "dataset__num_samples", "dataset__base_version", "model__name",
             "model__layer_name", "model__dim_inner", "model__adjacency", "model__base_to_data", "model__base_distr",
             "train__regularize"]

# %% Load dataframes
exp_folders = [
    "comparison_vaca",
]
df_all_last_list = []
df_all_training_list = []
for exp_folder in exp_folders:
    df = pb_help.load_df(root, [exp_folder], keep_cols, freq=10)
    df_all_last_list.append(df.last)
    df_all_training_list.append(df.training)


df_last_all = pd.concat(df_all_last_list, axis=0)
df_training_all = pd.concat(df_all_training_list, axis=0)

# %%

column_mapping = {"dataset__name": "Dataset", "dataset__sem_name": "SEM", "model__name": "Model"}

# %% Get training

filter_ = {}
# filter_["model__base_to_data"] = [False]


df_ = causal_nf_df.filter_df(df_training_all.copy(), filter_)
df_ = df_[df_.split == "train"]
df_ = df_[~df_.epoch.isin(["last", "best"])]
df_ = script_help.update_names(df_, column_mapping=column_mapping, model="vaca")

row_id_cols = [
    "SEM",
    "Dataset",
    "Model",
    "gnn__num_layers_pre",
    "model__dropout",
    "model__layer_name",
]
row_id_cols_all = [*row_id_cols, "time_forward"]


df_training = df_[row_id_cols_all].groupby(row_id_cols).agg(["mean", "std", "count"])


# %% Get last

filter_ = {}


df_ = causal_nf_df.filter_df(df_last_all.copy(), filter_)
df_ = script_help.update_names(df_, column_mapping=column_mapping, model="vaca")

df_["rmse_cf"] = df_.filter(regex="rmse_cf").mean(1)
df_["mmd_int"] = df_.filter(regex="mmd_int").mean(1)
df_["rmse_ate"] = df_.filter(regex="rmse_ate").mean(1)
# df_["kl_forward"] = df_["log_prob_true"] - df_["log_prob"]


row_id_cols = ["SEM", "Dataset", "Model"]

row_id_cols_ = [
    *row_id_cols,
]

df_best_all, row_id_cols_all = pb_help.get_best_df(
    df=df_, row_id_cols=row_id_cols, fn=np.argmax, metric="log_prob", show=False
)

df_tmp = df_best_all.groupby(row_id_cols_all).agg(["mean", "std", "count"])


# %%

df_table = df_tmp.copy()
cond = df_table.index.get_level_values("split") == "test"
df_table = df_table.loc[cond, :]

print("Best Hyperparam")
hparam_dict = {}
for v in df_table.index.get_level_values('Dataset').values:
    hparam_dict[v] = {'hparams': {}, 'metric': {}}
    tmp_df = df_table[df_table.index.get_level_values('Dataset') == v]
    for c in df_table.index.names:
        if '__' in c:
            hparam_dict[v]['hparams'][c] = tmp_df.index.get_level_values(c).values.item()
            hparam_dict[v]['metric']['log_prob_mean'] = tmp_df['log_prob']['mean'].values.item()
            hparam_dict[v]['metric']['log_prob_std'] = tmp_df['log_prob']['std'].values.item()

print(yaml.dump(hparam_dict, default_flow_style=False))
exit()

# Remove the specified level names
# for level in ["split"]:
#     df_table.index = df_table.index.droplevel(level)
#
#
# # df_table = df_table.droplevel(level=[0,2])
# index_order_latex = []
# index_order_latex.append("SEM")
# index_order_latex.append("Dataset")
# index_order_latex.append("Model")
#
# index_order = [*index_order_latex]
# index_order.append("model__layer_name")
# index_order.append("model__dropout")
# index_order.append("gnn__num_layers_pre")
#
# index_order_table = [df_table.index.names.index(i) for i in index_order]
# df_table.index = df_table.index.reorder_levels(index_order_table)
#
# index_order_training = [df_training.index.names.index(i) for i in index_order]
#
# df_training.index = df_training.index.reorder_levels(index_order_training)
#
# df_training_good = df_training.loc[df_table.index]
# df_table = df_table.drop(["time_forward"], axis=1)
# df_table = df_table.join(df_training_good)
#
#
# # %%
#
#
# # Get the list of level names to remove
# level_names_to_remove = [
#     level for level in df_table.index.names if level not in index_order_latex
# ]
#
# # Remove the specified level names
# for level in level_names_to_remove:
#     df_table.index = df_table.index.droplevel(level)
#
# l0 = df_table.index.names.index("SEM")
# l1 = df_table.index.names.index("Dataset")
# l2 = df_table.index.names.index("Model")
#
# df_table = df_table.sort_index(
#     level=[df_table.index.names.index(i) for i in index_order_latex], ascending=True
# )
#
#
# metrics_cols = [
#     "time_sample_obs",
#     "time_log_prob",
#     "time_forward",
#     "kl_forward",
#     "rmse_ate",
#     "rmse_cf",
#     "param_count",
# ]
#
# df_table = df_table.loc[:, df_table.columns.get_level_values(0).isin(metrics_cols)]
#
# # %%
#
# filename = os.path.join(folder, "comparison_vaca.pickle")
# print("Saving to {}".format(filename))
# df_table.to_pickle(filename)
