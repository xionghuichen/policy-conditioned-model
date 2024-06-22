import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from models.components import Swish, ResBlock, soft_clamp


class PolicyEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=200,
        rnn_num_layers=3,
        weight_decay=0.0,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.rnn_layer = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True
        )

        self.weight_decay = weight_decay

        self.device = torch.device(device)
        self.to(self.device)
    
    def forward(self, input, h_state=None):
        input = torch.as_tensor(input, dtype=torch.float32, device=self.device)
        embedding, h_state = self.rnn_layer(input, h_state)
        embedding = embedding.reshape(-1, self.hidden_dim)
        return embedding, h_state

    def get_decay_loss(self):
        decay_loss = 0
        for param in self.parameters():
            decay_loss += torch.norm(param)
        return self.weight_decay * decay_loss


class PolicyDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim=200,
        hidden_dims=[200, 200, 200, 200],
        dropout_rate=0.1,
        dist=None,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = torch.device(device)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = Swish()

        self.input_layer = ResBlock(input_dim, hidden_dims[0], dropout=dropout_rate, with_residual=False)

        module_list = []
        dims = list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(ResBlock(in_dim, out_dim, dropout=dropout_rate))
        self.pi_backbones = nn.ModuleList(module_list)

        # pool_target = int(embedding_dim/emb_dim_after_pool)
        # self.avg_p = nn.AvgPool1d(pool_target, stride=pool_target)

        # emb_dim_after_pool = 200
        self.pi_merge_layer = nn.Linear(dims[0] + embedding_dim, hidden_dims[0])
        self.dist = dist
        self.to(self.device)

    def forward(self, obs, policy_embedding):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        output_state = self.input_layer(obs)

        # # handle policy embedding
        # policy_embedding = torch.squeeze(self.avg_p(torch.unsqueeze(policy_embedding, 1)))
        # if len(policy_embedding.shape) == 1:
        #     policy_embedding = torch.unsqueeze(policy_embedding, 0)
        # policy_embedding = policy_embedding * torch.tensor(0.1)

        output = torch.cat([output_state, self.dropout(policy_embedding)], dim=-1)
        output = self.activation(self.pi_merge_layer(output))
        for layer in self.pi_backbones:
            output = layer(output)
        pi_dist = self.dist(output)

        return pi_dist


class PolicyConditionedDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_input_layer=3,
        embedding_dim=200,
        hidden_dims=[200, 200, 200, 200],
        dropout_rate=0.1,
        activation=Swish(),
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
        input_layer = [nn.GRUCell(input_size=input_dim, hidden_size=hidden_dims[0])]
        for _ in range(num_input_layer - 1):
            input_layer.append(nn.GRUCell(input_size=hidden_dims[0], hidden_size=hidden_dims[0]))
        self.input_layer = nn.ModuleList(input_layer)

        module_list = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            module_list.append(ResBlock(in_dim, out_dim, dropout=dropout_rate))
        self.backbones = nn.ModuleList(module_list)

        # pool_target = int(embedding_dim/emb_dim_after_pool)
        # self.avg_p = nn.AvgPool1d(pool_target, stride=pool_target)

        # emb_dim_after_pool = 200

        self.merge_layer = nn.Linear(hidden_dims[0]+embedding_dim, hidden_dims[0])
        self.output_layer = nn.Linear(hidden_dims[-1], 2 * output_dim)

        self.max_logvar = nn.Parameter(torch.ones(output_dim) * 0.5, requires_grad=True)
        self.min_logvar = nn.Parameter(torch.ones(output_dim) * -10, requires_grad=True)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, obs_act, policy_embedding):
        obs_act = torch.as_tensor(obs_act, dtype=torch.float32).to(self.device)
        output = obs_act
        for layer in self.input_layer:
            output = layer(output, None)

        # policy_embedding = torch.squeeze(self.avg_p(torch.unsqueeze(policy_embedding, 1)))
        # if len(policy_embedding.shape) == 1:
        #     policy_embedding = torch.unsqueeze(policy_embedding, 0)
        # policy_embedding = policy_embedding * torch.tensor(0.1)
        # output = torch.cat([output, policy_embedding], dim=-1)

        output = torch.cat([output, self.dropout(policy_embedding)], dim=-1)
        output = self.activation(self.merge_layer(output))
        for layer in self.backbones:
            output = layer(output)
        
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)

        return mean, logvar