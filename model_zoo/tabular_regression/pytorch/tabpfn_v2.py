"""TabPFN v2 regressor (Prior Labs, Nature 2025). In-context-learning tabular foundation regressor; competitive with tuned GBDTs out-of-the-box. LoRA adapters on top of the frozen prior-fitted transformer for federated fine-tuning."""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from tabpfn import TabPFNRegressor

framework = "pytorch"
model_type = ""
main_class = "MyModel"
license = "PriorLabs-Research"
batch_size = 64
output_classes = 1
num_feature_points = 17
category = "tabular_regression"


class MyModel(nn.Module):
    def __init__(self, num_feature_points=num_feature_points):
        super().__init__()
        reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
        reg._initialize_model_variables()
        self.transformer = reg.model_
        for p in self.transformer.parameters():
            p.requires_grad = False
        lora_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
            target_modules=["q", "k", "v", "out"],
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.head = nn.Linear(num_feature_points, 1)

    def forward(self, x):
        return self.head(x)
