import numpy as np
import torch
import torch.nn as nn

from models.components import Swish, ResBlock, soft_clamp


class VanillaDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_input_layer=3,
        hidden_dims=[200, 200, 200, 200],
        dropout_rate=0.1,
        device="cpu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = torch.device(device)

        self.activation = Swish()

        input_layer = [nn.Linear(input_dim, hidden_dims[0]), Swish()]

        self.input_layer = nn.ModuleList(input_layer)

        module_list = []
        dims = list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(ResBlock(in_dim, out_dim, dropout=dropout_rate))
        self.backbones = nn.ModuleList(module_list)
        
        self.output_layer = nn.Linear(hidden_dims[-1], 2 * output_dim)

        self.max_logvar = nn.Parameter(torch.ones(output_dim) * 0.5, requires_grad=True)
        self.min_logvar = nn.Parameter(torch.ones(output_dim) * -10, requires_grad=True)

        self.to(self.device)

    def forward(self, obs_act):
        obs_act = torch.as_tensor(obs_act, dtype=torch.float32).to(self.device)
        output = obs_act
        for layer in self.input_layer:
            output = layer(output)
        
        for layer in self.backbones:
            output = layer(output)
        
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)

        return mean, logvar