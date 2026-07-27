import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_mask = temperatures <= 0
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        float_logits = logits.float()
        probs = torch.softmax(float_logits / safe_temperatures.unsqueeze(dim=1), dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_tokens = float_logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
