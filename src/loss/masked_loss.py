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


class WeightedMAELoss(torch.nn.Module):
    """
    Weighted Mean Absolute Error (wMAE) loss for polymer property prediction.
    
    Implements the contest evaluation metric:
    - For each property: scaled_MAE_i = mean(|y_hat - y|) / range_i
    - Property weights: w_i = K * sqrt(1/n_i) / sum_j(sqrt(1/n_j))
    - Final score: weighted_average(scaled_MAEs, weights) = sum(w_i * scaled_MAE_i) / sum(w_i)
    
    This weighting scheme:
    1. Scale normalization: Division by range ensures properties with larger ranges don't dominate
    2. Inverse square-root scaling: sqrt(1/n_i) assigns higher weight to rare properties with fewer samples
    3. Weight normalization: Ensures sum of weights across all K properties equals K
    
    Args:
        property_ranges: Tensor of shape (K,) containing the range (max - min) for each property
        num_samples_per_property: Tensor of shape (K,) with counts of available values per property
    """
    def __init__(self, property_ranges, num_samples_per_property):
        super().__init__()
        self.K = len(property_ranges)
        self.register_buffer('property_ranges', property_ranges.float())
        self.register_buffer('num_samples', num_samples_per_property.float())
        self.register_buffer('property_weights', self._compute_property_weights())
        
    def _compute_property_weights(self):
        """Compute the property-specific weights: K * sqrt(1/n_i) / sum_j(sqrt(1/n_j))"""
        # Inverse square-root term: sqrt(1/n_i)
        inv_sqrt_n = torch.sqrt(1.0 / self.num_samples)
        
        # Normalized so sum equals K
        weights = (inv_sqrt_n / torch.sum(inv_sqrt_n)) * self.K
        
        return weights
    
    def forward(self, y_hat, y):
        """
        Compute weighted MAE loss matching the contest metric.
        
        Args:
            y_hat: Predicted values of shape (batch_size, K)
            y: True values of shape (batch_size, K), may contain NaN for missing values
            
        Returns:
            Weighted MAE loss (scalar)
        """
        # Create mask for non-missing values
        mask = ~torch.isnan(y)
        
        # Compute absolute errors
        abs_errors = torch.abs(y_hat - y)
        
        # Compute per-property scaled MAE
        scaled_maes = []
        weights_for_avg = []
        
        for i in range(self.K):
            property_mask = mask[:, i]
            num_valid = property_mask.sum()
            
            if num_valid > 0:
                # Mean absolute error for this property
                mae = abs_errors[property_mask, i].mean()
                # Scale by range: scaling_error = mae / range
                scaled_mae = mae / self.property_ranges[i]
                scaled_maes.append(scaled_mae)
                weights_for_avg.append(self.property_weights[i])
        
        if len(scaled_maes) == 0:
            return torch.tensor(0.0, device=y.device)
        
        # Convert to tensors
        scaled_maes = torch.stack(scaled_maes)
        weights_for_avg = torch.stack(weights_for_avg)
        weighted_score = torch.sum(weights_for_avg * scaled_maes) / torch.sum(weights_for_avg)
        
        return weighted_score