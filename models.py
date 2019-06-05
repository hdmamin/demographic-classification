import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def dims(self):
        """Get shape of each layer's weights."""
        return [p.shape for p in self.parameters()]

    def trainable(self):
        """Check which layers are trainable."""
        return [p.requires_grad for p in self.parameters()]

    def layer_stats(self):
        """Check mean and standard deviation of each layer's weights."""
        return [(round(p.data.mean().item(), 3),
                 round(p.data.std().item(), 3))
                for p in self.parameters()]


class CBow(BaseModel):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
