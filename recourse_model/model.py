from copy import deepcopy

import torch
from torch import nn
from recourse_utils.meta_helpers import downstream_for_recourse
import torch.nn.functional as F
from recourse_modules.dense import MLPModule
from recourse_utils.activations import get_activation
from typing import List
from recourse_utils.constants import Cte
from functools import partial


# def init_weights(m):
#     """Performs weight initialization."""
#     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#         m.weight.data.fill_(1.0)
#         m.bias.data.zero_()
#     elif isinstance(m, nn.Linear):
#         m.weight.data = nn.init.xavier_uniform_(m.weight.data,
#                                                 gain=nn.init.calculate_gain('relu'))
#         if m.bias is not None:
#             m.bias.data.zero_()


def _init_weights(m, gain=1.):
    """
    @param m: layer
    @param gain: gain value for initialization
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.01)


class RecourseMaskEncoder(nn.Module):
    def __init__(self, h_dim_list, num_nodes, act_name, device='cpu'):
        super(RecourseMaskEncoder, self).__init__()
        self.backbone = MLPModule(h_dim_list=h_dim_list, activ_name=act_name, bn=False, drop_rate=0.0, apply_last=True)
        self.mask_head = nn.Linear(h_dim_list[-1], num_nodes * 2)
        for m in [self.backbone, self.mask_head]:
            # m.apply(init_weights)
            m.apply(partial(_init_weights, gain=nn.init.calculate_gain('relu')))
            m.to(torch.device(device))

    def forward(self, data):
        internal_feat = self.backbone(data)
        mask_phi = self.mask_head(internal_feat)
        return mask_phi, internal_feat


def create_action_model(h_dim_list, z_dim, drop_rate, act_name, bn, num_nodes, clip_z, model_type=1):
    if model_type == 1:
        model = RecourseActionPredictor(h_dim_list=h_dim_list,
                                        z_dim=z_dim,
                                        drop_rate=drop_rate,
                                        act_name=act_name,
                                        bn=bn,
                                        num_nodes=num_nodes,
                                        clip_z=clip_z)
    else:
        model = RecourseActionDecoder(z_dim=z_dim, num_nodes=num_nodes, h_dim_list=h_dim_list,
                                      act_name=act_name, z_clipper=clip_z)

    model.apply(partial(_init_weights, gain=nn.init.calculate_gain('relu')))
    return model


class RecourseActionPredictor(nn.Module):
    """
    Heterogenouse Predictor Module
    """

    def __init__(self,
                 h_dim_list: List[int],
                 z_dim: int,
                 num_nodes: int,
                 drop_rate: float = 0.0,
                 act_name: str = Cte.RELU,
                 bn: bool = False,
                 clip_z: tuple = (-5, 5),
                 verbose=False):

        super(RecourseActionPredictor, self).__init__()

        self.num_nodes = num_nodes

        # Instantiate Input Embedding
        dim_input = h_dim_list[0]
        self.dim_input = dim_input

        self.clipping_z = clip_z

        self._input_embeddings = nn.ModuleList()

        for i in range(num_nodes):
            x_dim_i = z_dim + 1  # dimension of each input = length of z + 1 for mask
            # x_dim_i = 1 + 1  # 1 for x and 1 for mask

            if x_dim_i > 2 * dim_input:
                embed_i = nn.Sequential(nn.Linear(x_dim_i, 2 * dim_input, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(2 * dim_input, dim_input, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate))
            else:
                embed_i = nn.Sequential(nn.Linear(x_dim_i, dim_input, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate))
            self._input_embeddings.append(embed_i)
        if verbose:
            print("Predictor Input Embedding")
            print(self._input_embeddings)
        # Instantiate shared module
        self.predictor_module = MLPModule(h_dim_list=h_dim_list,
                                          activ_name=act_name,
                                          bn=bn,
                                          drop_rate=drop_rate,
                                          apply_last=True)
        if verbose:
            print("Core predictor module")
            print(self.predictor_module)

        # Instantiate the output embeddings
        self.dim_output = h_dim_list[-1]
        self.z_dim = z_dim

        self._output_embeddings = nn.ModuleList()

        for i in range(num_nodes):
            if self.dim_output > 2 * self.z_dim:
                embed_i = nn.Sequential(nn.Linear(self.dim_output, 2 * self.z_dim, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(2 * self.z_dim, self.z_dim, bias=True),
                                        )
            else:
                embed_i = nn.Sequential(nn.Linear(self.dim_output, self.z_dim, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(self.z_dim, self.z_dim, bias=True))
            self._output_embeddings.append(embed_i)
        if verbose:
            print("Predictor output embedding")
            print(self._output_embeddings)

    def predictor_params(self):
        params = list(self.predictor_module.parameters()) \
                 + list(self._input_embeddings.parameters()) \
                 + list(self._output_embeddings.parameters())
        return params

    def input_embeddings(self, Z, xI):
        if Z.size()[1] != self.z_dim * self.num_nodes:
            Z = Z.reshape(-1, self.z_dim * self.num_nodes)
            # Z = Z.view(-1, self.z_dim * self.num_nodes)

        assert Z.size()[1] == self.z_dim * self.num_nodes, "something wrong with size of input"

        embeddings = []

        for i, embed_i in enumerate(self._input_embeddings):
            Z_i = Z[:, (i * self.z_dim):((i + 1) * self.z_dim)]

            xI_i = xI[i] * torch.ones(Z_i.shape[0])
            xI_i = torch.reshape(xI_i, (-1, 1))

            ZxI_i = torch.cat([Z_i, xI_i], dim=1)

            H_i = embed_i(ZxI_i)

            embeddings.append(H_i)

        cat_emb = torch.cat(embeddings, dim=1)

        return cat_emb.view(-1, self.dim_input)  # [768, 32]

    def output_embeddings(self, Z):

        d = Z.shape[1]

        Z_0 = Z.view(-1, self.num_nodes * d)  # 3x16 = 48

        embeddings = []
        for i, embed_i in enumerate(self._output_embeddings):
            Z_i = Z_0[:, (i * d):((i + 1) * d)]
            H_i = embed_i(Z_i)

            embeddings.append(H_i)

        return_statement = torch.cat(embeddings, dim=1)

        assert return_statement.shape[1] == self.num_nodes * self.z_dim, "problem with output shapes"

        return return_statement

    def predictor(self, zf, xI):
        pred_input = self.input_embeddings(zf, xI)
        pred_output = self.predictor_module(pred_input)
        zI = self.output_embeddings(pred_output)
        return zI

    def forward(self, z, action):
        action = action.t()
        zI = self.predictor(z, action)
        zI = torch.clamp(zI, self.clipping_z[0], self.clipping_z[1])
        return zI

    def predict(self, z, action):
        zI = self.predictor(z, action)
        zI = torch.clamp(zI, self.clipping_z[0], self.clipping_z[1])
        return zI


class RecourseActionDecoder(nn.Module):
    def __init__(self, z_dim, num_nodes, h_dim_list, act_name, z_clipper=None):
        super(RecourseActionDecoder, self).__init__()
        in_dim = z_dim * num_nodes + num_nodes  # latent + mask
        # AM: Taking features in from the Encoder.
        h_dim_list.insert(0, in_dim)
        h_dim_list.append(num_nodes * z_dim)
        self.model = MLPModule(h_dim_list=h_dim_list, activ_name=act_name, bn=False,
                               drop_rate=0.0, apply_last=False)
        self.z_clipper = z_clipper
        self.model.apply(partial(_init_weights, gain=nn.init.calculate_gain('relu')))

    def forward(self, z, action):
        feats = torch.cat((z, action), dim=1)
        if self.z_clipper is None:
            return self.model(feats)
        else:
            return torch.clamp(self.model(feats), self.z_clipper[0], self.z_clipper[1])


class RecourseModel(nn.Module):
    def __init__(self, predictor, encoder, causal_model, causal_model_name, clf_model,
                 feat_to_intervene, down_caller, tau=1):
        super(RecourseModel, self).__init__()
        self.predictor = predictor
        self.encoder = encoder
        self.feat_to_intervene = feat_to_intervene
        self.down_caller = down_caller
        self.causal_model = causal_model
        self.causal_model_name = causal_model_name
        self.clf_model = clf_model
        self.tau = tau

    def forward(self, batch, is_train=True):
        # AM: 1. Concat the z of the features
        z = batch.z.clone()
        if self.causal_model_name == 'vaca':
            # AM: Since dataloader is PYG, need to reshape it properly here.
            num_graphs = batch.num_graphs
            num_nodes_total = batch.x.shape[0]
            num_nodes = num_nodes_total // num_graphs
            z = z.view(-1, num_nodes, self.causal_model.z_dim)
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)
        # AM: 2. Pass through encoder to get phi
        phi, feats = self.encoder(z_flat)
        phi_for_kl = phi.view(batch_size, -1, 2)  # batch_size x num_nodes x 2
        # AM: Reshape phi to (batch_size * num_nodes x 2)
        phi_reshaped = phi.view(-1, 2)
        # AM: 3. Call gumbel-softmax (with hard=True) to get the mask. Sample!
        if is_train:
            mask = F.gumbel_softmax(phi_reshaped, tau=self.tau, hard=True)  # batch_size * num_nodes x 2
            mask_1 = mask[:, 1]
        else:
            # TODO: Discuss with Isabel. Should we do this? Or we move to CATE-based interventional,
            #  and output "diverse" counterfactuals with Monte-Carlo samples
            mask_1 = (torch.sigmoid(phi_reshaped[:, 1]) > 0.5).float()
        # AM: Reshape the mask to (batch_size x num_nodes)
        action_tensor = mask_1.view(batch_size, -1)
        # AM: Masking out action tensor based on which features we can intervene on.
        intervene_allow_mask = torch.zeros_like(action_tensor, requires_grad=False)
        intervene_allow_mask[:, self.feat_to_intervene] = 1
        action_tensor = action_tensor * intervene_allow_mask
        # AM: 4. Call predictor with mask to get Delta!
        # TODO: NEEDS CHECKING AND NEEDS CHANGING.
        policy_cf, recourse_prob, recourse_lab = downstream_for_recourse(batch, action_tensor, self.causal_model,
                                                                         self.predictor, self.clf_model,
                                                                         causal_mod_name=self.causal_model_name)
        return policy_cf, recourse_prob, recourse_lab, action_tensor, phi_for_kl
