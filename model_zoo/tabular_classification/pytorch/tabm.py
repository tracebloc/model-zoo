"""TabM (Yandex, 2024). Parameter-efficient MLP ensemble — k weight-shared MLPs with per-member adapters; current SOTA among trainable-from-scratch tabular DL on the rtdl benchmarks. LayerNorm-based → federated-safe."""
import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "MyModel"
license = "MIT"
batch_size = 1024
output_classes = 5
num_feature_points = 50
category = "tabular_classification"


class _EnsembleLinear(nn.Module):
    def __init__(self, in_f, out_f, k):
        super().__init__()
        self.shared = nn.Linear(in_f, out_f)
        # Per-member rank-1 multiplicative adapter (BatchEnsemble-style)
        self.r = nn.Parameter(torch.ones(k, in_f))
        self.s = nn.Parameter(torch.ones(k, out_f))

    def forward(self, x):
        # x: (B, K, in_f)
        x = x * self.r.unsqueeze(0)
        x = self.shared(x)
        return x * self.s.unsqueeze(0)


class MyModel(nn.Module):
    def __init__(self, num_classes=output_classes, num_feature_points=num_feature_points,
                 hidden=128, k=32):
        super().__init__()
        self.k = k
        self.l1 = _EnsembleLinear(num_feature_points, hidden, k)
        self.l2 = _EnsembleLinear(hidden, hidden, k)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)
        self.act = nn.GELU()

    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(1).expand(b, self.k, -1)  # (B, K, F)
        x = self.act(self.norm1(self.l1(x)))
        x = self.act(self.norm2(self.l2(x)))
        logits = self.head(x)  # (B, K, C)
        return logits.mean(dim=1)
