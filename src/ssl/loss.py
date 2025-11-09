import torch
import torch.nn as nn
import torch.nn.functional as F


# Since there were some issues with the lightning_bolts NTXentLoss,
# we implement our own here with some help of LLM
class NTXentLoss(nn.Module):
    """
    Standalone implementation of NTXentLoss (SimCLR loss).
    """

    def __init__(self, temperature=0.1, device=None):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        z_i and z_j are tensors of shape [batch_size, embed_dim]
        """
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        representations = torch.cat([z_j, z_i], dim=0)
        similarity_matrix = self.similarity_f(
            representations.unsqueeze(1), representations.unsqueeze(0)
        )

        # Create positive matches
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        # Get negative matches (all-to-all except self and positive pairs)
        diag = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        diag[batch_size:, :batch_size] = True
        diag[:batch_size, batch_size:] = True

        negatives = similarity_matrix[~diag].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        # Labels: 0-th column is always the positive key
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)

        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)
