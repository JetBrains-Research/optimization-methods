import torch
from torch import nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.attn = nn.Linear(units, units, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Calculate attention weights
        :param hidden: [batch size; units]
        :param encoder_outputs: [batch size; seq len; units]
        :return: [batch size; seq len]
        """
        batch_size = hidden.size(0)
        # [batch size; units]
        attended_hidden = self.attn(hidden)
        # [batch size; seq len]
        score = torch.bmm(encoder_outputs, attended_hidden.view(batch_size, -1, 1)).squeeze(-1)

        # [batch size; seq len]
        weights = F.softmax(score, dim=1)
        return weights
