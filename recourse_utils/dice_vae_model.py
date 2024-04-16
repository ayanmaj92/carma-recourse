import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from recourse_utils.constants import Cte

from recourse_modules.dense import MLPModule


class DiceVAE(pl.LightningModule):
    def __init__(self, cfg, num_nodes, clf_model):
        super().__init__()
        enc_h_dim_list = cfg.dice_vae.params.enc_hdim_list
        enc_h_dim_list = [num_nodes] + enc_h_dim_list
        act_name = cfg.dice_vae.params.act
        bn = cfg.dice_vae.params.bn
        drop_rate = cfg.dice_vae.params.drop_rate
        z_dim = cfg.dice_vae.params.z_dim
        self.optim_lr = cfg.dice_vae.params.optim_lr
        self.loss_type = cfg.dice_vae.params.loss_type
        self.beta = 1.0
        self.util_array = cfg.reward_util_array
        if self.loss_type == Cte.L2:
            self.obj_x = nn.MSELoss(reduction='none')
        elif self.loss_type == Cte.L1:
            self.obj_x = nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError
        self.encoder = MLPModule(h_dim_list=enc_h_dim_list,
                                 activ_name=act_name,
                                 bn=bn,
                                 drop_rate=drop_rate,
                                 apply_last=True)
        self.enc_mu = nn.Linear(enc_h_dim_list[-1], z_dim)
        self.enc_log_var = nn.Linear(enc_h_dim_list[-1], z_dim)
        dec_h_dim_list = [z_dim] + enc_h_dim_list[::-1]
        self.decoder = MLPModule(h_dim_list=dec_h_dim_list,
                                 activ_name=act_name,
                                 bn=bn,
                                 drop_rate=drop_rate)
        self.clf_model = clf_model
        self.hinge_margin = cfg.dice_vae.params.hinge_margin
        self.valid_reg = 0.0
        self.immutable_ids = cfg.dice_vae.params.immut_feats

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.enc_mu(hidden)
        log_var = self.enc_log_var(hidden)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, batch):
        mu, log_var = self.encode(batch)
        z = self._reparameterize(mu, log_var)
        delta_recon = self.decode(z)
        mask = torch.ones_like(batch)
        mask[:, self.immutable_ids] = 0
        x_cf = batch + delta_recon * mask
        return x_cf

    @torch.no_grad()
    def cf_reconstruct(self, data, sampling=False):
        mu, log_var = self.encode(data)
        if sampling:
            z = self._reparameterize(mu, log_var)
        else:
            z = mu
        x_delta = self.decode(z)
        mask = torch.ones_like(data)
        mask[:, self.immutable_ids] = 0
        x_cf = data + x_delta * mask
        return x_cf

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.optim_lr)

    def recon_loss(self, x, x_cf):
        list_loss = []
        for i in range(x.shape[1]):
            if i not in self.immutable_ids:
                list_loss.append(self.obj_x(x[:, i], x_cf[:, i]) / (self.util_array[i][2] - self.util_array[i][3]))
        loss = torch.stack(list_loss, dim=1)
        loss = torch.mean(loss, dim=1).mean(dim=0)
        return loss

    def training_step(self, batch, batch_idx):
        mu, log_var = self.encode(batch)
        z = self._reparameterize(mu, log_var)
        x_delta = self.decode(z)
        mask = torch.ones_like(batch)
        mask[:, self.immutable_ids] = 0
        x_cf = batch + x_delta * mask
        _, pred_probs = self.clf_model(x_cf)
        kl_div = self.kl_divergence(mu, log_var)
        recon_loss = self.recon_loss(batch, x_cf)
        vae_loss = recon_loss + self.beta * kl_div
        clf_hinge_loss = F.hinge_embedding_loss(pred_probs - (1 - pred_probs),
                                                torch.tensor(-1.0).type_as(pred_probs), self.hinge_margin)
        damping = 10.0
        damp = damping * clf_hinge_loss.detach()
        loss = vae_loss + (self.valid_reg - damp) * clf_hinge_loss
        self.valid_reg = self.valid_reg + 1.0 * clf_hinge_loss.detach()
        self.valid_reg = torch.clip(self.valid_reg, min=0.0)
        self.log('train_vae_loss', vae_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_kl_loss', kl_div, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_valid_loss', clf_hinge_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, sampling=False):
        mu, log_var = self.encode(batch)
        if not sampling:
            z = mu
        else:
            z = self._reparameterize(mu, log_var)
        x_cf = self.decode(z)
        _, pred_probs = self.clf_model(x_cf)
        kl_div = self.kl_divergence(mu, log_var)
        recon_loss = self.recon_loss(batch, x_cf)
        vae_loss = recon_loss + self.beta * kl_div
        clf_hinge_loss = F.hinge_embedding_loss(pred_probs - (1 - pred_probs),
                                                torch.tensor(-1.0).type_as(pred_probs), self.hinge_margin)
        immutable_loss = F.l1_loss(batch[:, self.immutable_ids], x_cf[:, self.immutable_ids])
        loss = vae_loss + self.valid_reg * clf_hinge_loss
        self.log('valid_vae_loss', vae_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_immut_loss', immutable_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_valid_loss', clf_hinge_loss, on_step=False, on_epoch=True, prog_bar=True)
        return x_cf, loss

    @staticmethod
    def _reparameterize(mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z

    @staticmethod
    def kl_divergence(mu, log_var, reduction='mean'):
        kl_loss = (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1))
        if reduction == 'none' or reduction == '':
            return kl_loss
        elif reduction == 'mean':
            return kl_loss.mean(dim=0)
        elif reduction == 'sum':
            return kl_loss.sum(dim=0)
        else:
            raise NotImplementedError
