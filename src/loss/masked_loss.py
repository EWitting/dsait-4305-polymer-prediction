import torch

class MaskedLoss(torch.nn.Module):
    """Since the dataset has many missing values, we need to compute the loss only where the label is non-missing.
    This helper class wraps around a loss function and computes the loss only where the label is non-missing."""
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_hat, y):
        mask = ~torch.isnan(y)
        loss = self.loss_fn(y_hat[mask], y[mask])
        return loss