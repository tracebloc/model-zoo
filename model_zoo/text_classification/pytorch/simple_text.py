import torch
import torch.nn as nn
import torch.nn.functional as F

framework = "pytorch"
main_class = "SimpleTextClassifier"
category = "text_classification"
model_type = ""
batch_size = 32
sequence_length = 5
output_classes = 5


class SimpleTextClassifier(nn.Module):
    def __init__(self, embed_dim=512, num_classes=5):
        super(SimpleTextClassifier, self).__init__()
        # Assuming that the embeddings are already handled by the tokenizer
        self.fc1 = nn.Linear(embed_dim, 128)  # First dense layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        # Flatten the input
        x = torch.flatten(x, start_dim=1)
        # Apply a fully connected layer and ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x
