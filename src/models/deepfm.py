from __future__ import annotations

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, field_dims: list[int], embed_dim: int = 16, mlp_dims: tuple[int, ...] = (64, 32), dropout: float = 0.1, n_heads: int = 1):
        super().__init__()
        self.n_fields = len(field_dims)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.mlp_dims = tuple(mlp_dims)
        self.first_order = nn.ModuleList(
            [nn.Embedding(d, 1, padding_idx=0) for d in field_dims]
        )
        self.second_order = nn.ModuleList(
            [nn.Embedding(d, embed_dim, padding_idx=0) for d in field_dims]
        )
        for emb in list(self.first_order) + list(self.second_order):
            nn.init.xavier_uniform_(emb.weight)
            with torch.no_grad():
                emb.weight[0].zero_()
        mlp_in = self.n_fields * embed_dim
        layers: list[nn.Module] = []
        last = mlp_in
        for dim in mlp_dims:
            layers += [nn.Linear(last, dim), nn.ReLU(), nn.Dropout(dropout)]
            last = dim
        self.mlp = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(last, 1) for _ in range(n_heads)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, fm_term = self._shared(x)
        if self.n_heads == 1:
            return self.heads[0](hidden).squeeze(-1) + fm_term
        return torch.stack([head(hidden).squeeze(-1) + fm_term for head in self.heads], dim=-1)

    def _shared(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_first = torch.stack([emb(x[:, i]) for i, emb in enumerate(self.first_order)], dim=1).sum(dim=1).squeeze(-1)
        v = torch.stack([emb(x[:, i]) for i, emb in enumerate(self.second_order)], dim=1)
        square_of_sum = v.sum(dim=1) ** 2
        sum_of_square = (v ** 2).sum(dim=1)
        y_fm = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        hidden = self.mlp(v.reshape(v.size(0), -1))
        return hidden, y_first + y_fm + self.bias.squeeze()
