import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj


def get_adj_matrix(loader):
    return loader.dataset.adj_object.adj_matrix.copy()


def set_interventions_for_vaca(batch, action_list):
    x = torch.tensor(1)
    adj_matrix = to_dense_adj(batch.edge_index)[0]
    intervention_ids = (action_list == x).nonzero().flatten()
    adj_matrix[:, intervention_ids] = 0.0
    adj_matrix[intervention_ids, intervention_ids] = 1.0
    edge_index_i, _ = dense_to_sparse(torch.clone(adj_matrix))
    hash_table = {tuple(edge_index_i[:, i].tolist()): i for i in range(edge_index_i.shape[1])}
    take_idxs = []
    for j in range(batch.edge_index.shape[1]):
        idx = hash_table.get(tuple(batch.edge_index[:, j].tolist()), None)
        if idx is not None:
            take_idxs.append(j)
    batch.edge_index = edge_index_i
    return batch


def downstream_for_recourse(batch, action_list, causal_mod, predictor_nn, clf_model, causal_mod_name='causal_nf'):
    if causal_mod_name == 'vaca':
        # 1. Set intervened edge_index, 2. Set the z values by intervention, 3. Compute CF.
        batch_size = batch.num_graphs
        batch_i = batch.clone()
        batch_i = set_interventions_for_vaca(batch_i, action_list)
        z_factual = batch.z
        if causal_mod_name == 'causal_nf':
            intervention_z = predictor_nn(z_factual.view(batch_size, -1), action_list.t())
        else:
            intervention_z = predictor_nn(z_factual.view(batch_size, -1), action_list)
        z_cf_I = intervention_z.view(z_factual.shape)
        batch_i.z_i = z_cf_I
        acts = action_list.view(-1, 1)
        z_counterfactual = acts * torch.clone(z_cf_I) + z_factual
        batch.z_cf = z_counterfactual
        batch_i.x = z_counterfactual
        logits = causal_mod.decoder_gnn(batch_i)
        x_cf, _ = causal_mod.likelihood_distr(logits, return_mean=True)
        batch.x_cf = x_cf
        batch.edge_index_i = batch_i.edge_index
        data_policy_cf_ = batch.x_cf.view(batch_size, -1)
    elif causal_mod_name == 'causal_nf':
        batch_size = batch.z.shape[0]
        intervention_z = predictor_nn(batch.z.view(batch_size, -1), action_list)
        z_cf_I = intervention_z.view(batch_size, -1)
        z_factual = batch.z.view(batch_size, -1)
        batch.z_i = intervention_z
        z_counterfactual = action_list * torch.clone(z_cf_I) + z_factual
        batch.z_cf = z_counterfactual
        x_cf = causal_mod.flow().transform.inv(batch.z_cf)
        batch.x_cf = x_cf
        data_policy_cf_ = batch.x_cf
    else:
        raise NotImplementedError

    recourse_pol_label, recourse_pol_prob = clf_model(data_policy_cf_)
    return data_policy_cf_, recourse_pol_prob, recourse_pol_label
