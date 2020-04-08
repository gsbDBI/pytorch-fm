
import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class EmbeddingsModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, obsItem_dims, embed_dim, no_obs=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.obsItem_coeff = FeaturesEmbedding(field_dims[1:], obsItem_dims)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.no_obs = no_obs

    def forward(self, x, x_item_obs):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # print(x, x_item_obs, x_item_obs.shape)
        # print(self.obsItem_coeff(x[:, 1:])[:,0,:].shape)
        # print(torch.sum(torch.mul(self.obsItem_coeff(x[:, 1:])[:,0,:], x_item_obs), dim=1, keepdims=True).shape)
        # print(x_item_obs.to(torch.float32), x_item_obs.shape)
        # input()
        if self.no_obs:
            x = self.linear(x) + self.fm(self.embedding(x))
        else:
            x = self.linear(x) + self.fm(self.embedding(x)) + torch.sum(torch.mul(self.obsItem_coeff(x[:, 1:])[:,0,:], x_item_obs), dim=1, keepdims=True)
        # print(x.shape)
        # print(x)
        # input()
        return torch.sigmoid(x.squeeze(1))
