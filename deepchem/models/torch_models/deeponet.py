import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss


class DeepONet(nn.Module):

    def __init__(self,
                 branch_input_dim: int,
                 trunk_input_dim: int,
                 branch_hidden: Tuple[int, ...] = (64, 64),
                 trunk_hidden: Tuple[int, ...] = (64, 64),
                 output_dim: int = 64,
                 activation_fn: str = 'relu') -> None:
        super().__init__()
        self.output_dim = output_dim

        self.branch_net = MultilayerPerceptron(d_input=branch_input_dim,
                                               d_hidden=branch_hidden,
                                               d_output=output_dim,
                                               activation_fn=activation_fn)

        self.trunk_net = MultilayerPerceptron(d_input=trunk_input_dim,
                                              d_hidden=trunk_hidden,
                                              d_output=output_dim,
                                              activation_fn=activation_fn)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        branch_input, trunk_input = inputs[0], inputs[1]
        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)
        output = torch.sum(branch_out * trunk_out, dim=-1,
                           keepdim=True) + self.bias
        return output


class DeepONetModel(TorchModel):

    def __init__(self,
                 branch_input_dim: int,
                 trunk_input_dim: int,
                 branch_hidden: Tuple[int, ...] = (64, 64),
                 trunk_hidden: Tuple[int, ...] = (64, 64),
                 output_dim: int = 64,
                 activation_fn: str = 'relu',
                 **kwargs) -> None:
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        model = DeepONet(branch_input_dim=branch_input_dim,
                         trunk_input_dim=trunk_input_dim,
                         branch_hidden=branch_hidden,
                         trunk_hidden=trunk_hidden,
                         output_dim=output_dim,
                         activation_fn=activation_fn)
        super().__init__(model, loss=L2Loss(), **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for X, y, w, ids in dataset.iterbatches(batch_size=self.batch_size,
                                                    deterministic=deterministic,
                                                    pad_batches=pad_batches):
                branch_input = X[:, :self.branch_input_dim]
                trunk_input = X[:, self.branch_input_dim:]
                inputs = [
                    torch.tensor(branch_input, dtype=torch.float32),
                    torch.tensor(trunk_input, dtype=torch.float32)
                ]
                labels = [torch.tensor(y, dtype=torch.float32)]
                weights = [torch.tensor(w, dtype=torch.float32)]
                yield (inputs, labels, weights)
