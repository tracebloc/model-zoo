"""TabM regressor (Yandex, 2024). Parameter-efficient MLP ensemble — BatchEnsemble-style multiplicative adapters over a shared MLP; SOTA among from-scratch tabular DL regressors. Federated-friendly (LayerNorm only)."""
import torch
import torch.nn as nn

framework = "pytorch"
model_type = ""
main_class = "MyModel"
license = "MIT"
batch_size = 1024
output_classes = 1
num_feature_points = 17
category = "tabular_regression"


class _EnsembleLinear(nn.Module):
    def __init__(self, in_f, out_f, k):
        super().__init__()
        self.shared = nn.Linear(in_f, out_f)
        self.r = nn.Parameter(torch.ones(k, in_f))
        self.s = nn.Parameter(torch.ones(k, out_f))

    def forward(self, x):
        x = x * self.r.unsqueeze(0)
        x = self.shared(x)
        return x * self.s.unsqueeze(0)


class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points, hidden=128, k=32):
        super().__init__()
        self.k = k
        self.l1 = _EnsembleLinear(num_feature_points, hidden, k)
        self.l2 = _EnsembleLinear(hidden, hidden, k)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        self.act = nn.GELU()

    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(1).expand(b, self.k, -1)
        x = self.act(self.norm1(self.l1(x)))
        x = self.act(self.norm2(self.l2(x)))
        return self.head(x).mean(dim=1)
