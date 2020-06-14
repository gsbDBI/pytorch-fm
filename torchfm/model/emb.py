
import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class EmbeddingsModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, obsItem_dims, obsUser_dims, embed_dim, obs, embed):
        super().__init__()
        # print(field_dims, embed_dim)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # print(field_dims[0:1], obsItem_dims)
        # input()
        # if obs:
        self.obsItem_coeff = FeaturesEmbedding(field_dims[0:1], obsItem_dims)
        self.obsUser_coeff = FeaturesEmbedding(field_dims[1:2], obsUser_dims)
        self.linear = FeaturesLinear(field_dims, bias=False)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.obs = obs
        self.embed = embed
        assert obs or embed, "One of obs or embed must be true\n" 

    def forward(self, x, x_item_obs=None, x_user_obs=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # print(x, x_item_obs, x_item_obs.shape)
        # print(self.obsItem_coeff(x[:, 1:])[:,0,:].shape)
        # print(torch.sum(torch.mul(self.obsItem_coeff(x[:, 1:])[:,0,:], x_item_obs), dim=1, keepdims=True).shape)
        # print(x_item_obs.to(torch.float32), x_item_obs.shape)
        # input()
        # temp = self.obsUser_coeff(x[:, 1:2])[:,0,:]
        # print(torch.mul(self.obsItem_coeff(x[:, 0:1])[:,0,:], x_item_obs))
        # print(self.obsItem_coeff(x[:, 0:1])[:,0,:].shape, x_item_obs.shape)
        # print(temp)
        # print(x_user_obs)
        # print(torch.mul(temp, x_user_obs))
        # input()
        if self.embed:
            # print(x)
            y = self.linear(x) + self.fm(self.embedding(x)) 
        if self.obs:
            z = torch.sum(torch.mul(self.obsItem_coeff(x[:, 0:1])[:,0,:], x_item_obs), dim=1, keepdims=True)  + torch.sum(torch.mul(self.obsUser_coeff(x[:, 1:2])[:,0,:], x_user_obs), dim=1, keepdims=True)

        if self.embed and self.obs:
            ans = y + z
        elif self.embed:
            ans = y
        elif self.obs:
            ans = z
        else:
            assert False, "Unreachable; one of obs and embed must be True"
            # x = torch.sum(torch.mul(self.obsItem_coeff(x[:, 0:1])[:,0,:], x_item_obs), dim=1, keepdims=True)
        # print(x.shape)
        # print(x)
        # input()
        return torch.sigmoid(ans.squeeze(1))

    def get_embeddings(self):
        return self.embedding.embedding.weight.data, self.linear.fc.weight.data, self.obsItem_coeff.embedding.weight.data, self.obsUser_coeff.embedding.weight.data
