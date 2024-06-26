import torch.nn as nn

from recourse_utils import constants
from .blocks.dense import basic_dense_block


class MLPModule(nn.Module):
    def __init__(self, h_dim_list,
                 activ_name,
                 bn,
                 drop_rate,
                 apply_last=False):
        """
        Initializes the MLP NN
        @param h_dim_list: hidden layer dimension list
        @param activ_name: activation function
        @param bn: batch normalization
        @param drop_rate: dropout rate
        """
        super(MLPModule, self).__init__()
        assert isinstance(h_dim_list, list)
        assert len(h_dim_list) > 1
        n_layers = len(h_dim_list) - 1

        layers = []

        for i, (h_in, h_out) in enumerate(zip(h_dim_list[:-1], h_dim_list[1:])):
            if (i + 1) < n_layers:
                layers.append(basic_dense_block(h_in, h_out, activ_name, drop_rate=drop_rate, bn=bn))
            else:
                # last layer!
                if not apply_last:
                    layers.append(basic_dense_block(h_in, h_out, activation_name=constants.Cte.IDENTITY,
                                                    drop_rate=0.0, bn=False))
                else:
                    layers.append(basic_dense_block(h_in, h_out, activ_name, drop_rate=drop_rate, bn=bn))

        self.mlp = nn.Sequential(*layers)
        self.dims = None

    def set_output_dims(self, dims):
        self.dims = dims

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        logits = self.mlp(x)
        if self.dims is not None:
            logits = logits.view(logits.size(0), *self.dims)

        return logits
