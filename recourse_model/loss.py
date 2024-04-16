import torch
import torch.distributions as dist
import torch.nn.functional as F


def categorical_kl_divergence(phi: torch.Tensor, is_custom: bool, weights) -> torch.Tensor:
    """
    Source: https://github.com/jxmorris12/categorical-vae/blob/master/train.py
    """
    # phi is logits of shape [B, N, K] where B is batch, N is number of categorical distributions,
    # K is number of classes
    B, N, K = phi.shape
    phi = phi.view(B*N, K)
    q = dist.Categorical(logits=phi)
    if not is_custom:
        p = dist.Categorical(probs=torch.full((B * N, K), 1.0/K))  # uniform bunch of K-class categorical distributions
    else:
        prob_tensor = torch.full((N, K), 1.0/K)
        # Only for loan data
        # prob_tensor[0] = torch.tensor([0.75, 0.25])
        # prob_tensor[3] = torch.tensor([0.25, 0.75])
        for k in range(len(weights)):
            x = (1.0 / (weights[k] * K))
            prob_tensor[k] = torch.tensor([1 - x, x])
        # x = (1.0 / (3 * K))
        # prob_tensor[0] = torch.tensor([1 - x, x])
        # x = (1.0 / (2 * K))
        # prob_tensor[1] = torch.tensor([1 - x, x])
        # prob_tensor[2] = torch.tensor([1 - x, x])
        p_t = prob_tensor.repeat(B, 1)
        p = dist.Categorical(probs=p_t)
    kl = dist.kl.kl_divergence(q, p)  # kl is of shape [B*N]
    return kl.view(B, N)


def compute_l2_cost(data, cf_data, feature_cdf_costs, action_list, reduce=True, feat_to_intervene=None, weights=None):
    list_loss = []
    k = -1
    for i in range(data.shape[1]):
        if i in feat_to_intervene:
            k += 1
            list_loss.append(
                weights[k] *
                action_list[:, i] *
                F.mse_loss(data[:, i], cf_data[:, i], reduction='none') /
                (feature_cdf_costs[i][2] - feature_cdf_costs[i][3])
            )
    loss = torch.stack(list_loss, dim=1)
    if reduce:
        loss = torch.mean(loss, dim=1).mean(dim=0)
    return loss


def lagrangian(data, cf_data, pred_proba, feature_cdf_costs, action_mask, phi, feat_to_intervene, lmbd, eps, weights,
               custom_prior):
    assert data.shape == cf_data.shape, f'Mismatch shapes data {data.shape} and cf {cf_data.shape}'
    if feat_to_intervene is not None and len(feat_to_intervene) > 0:
        phi = phi[:, feat_to_intervene, :]
    kl_loss = torch.mean(torch.sum(categorical_kl_divergence(phi, custom_prior, weights), dim=1))
    cost = compute_l2_cost(data, cf_data, feature_cdf_costs, action_mask, reduce=True,
                           feat_to_intervene=feat_to_intervene, weights=weights)
    loss_1 = cost + kl_loss
    loss_2 = F.hinge_embedding_loss(pred_proba - (1 - pred_proba), torch.tensor(-1.0).type_as(pred_proba), eps)
    damping = 10
    damp = damping * loss_2.detach()
    return loss_1 + (lmbd - damp) * loss_2, cost, loss_2
