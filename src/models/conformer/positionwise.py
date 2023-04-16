import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer."""

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: nn.Module = nn.ReLU()):
        """Construct a PositionwiseFeedForward object.

        FeedForward are applied on each position of the sequence.
        The output dim is same with the input dim.

        Args:
            idim (int): Input dimension.
            hidden_units (int): The number of hidden units.
            dropout_rate (float): Dropout rate.
            activation (torch.nn.Layer): Activation function
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            xs: input tensor (B, Lmax, D)
        Returns:
            output tensor, (B, Lmax, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
