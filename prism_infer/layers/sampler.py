import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Fast path: all requests are greedy — argmax directly on raw logits,
        # identical to HF do_sample=False.
        if temperatures.eq(0).all():
            return logits.argmax(dim=-1)

        # Mixed batch: some greedy, some stochastic.
        greedy_mask = temperatures.eq(0)
        if greedy_mask.any():
            temperatures = temperatures.clone()
            temperatures[greedy_mask] = 1.0  # placeholder to avoid div-by-zero

        sample_tokens = self._gumbel_sample(logits, temperatures)

        if greedy_mask.any():
            sample_tokens = torch.where(greedy_mask, logits.argmax(dim=-1), sample_tokens)

        return sample_tokens

    @torch.compile  # fuse softmax + noise + argmax into one kernel
    def _gumbel_sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
