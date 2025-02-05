import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class ParticlePickingLoss(nn.Module):
    def __init__(self):
        super(ParticlePickingLoss, self).__init__()

    def forward(self, true_coords_list, pred_tensor):
        batch_size = len(true_coords_list)
        total_loss = 0.0

        for b in range(batch_size):
            # Extract the batch elements
            true_coords_b = true_coords_list[b]
            pred_b = pred_tensor[b]

            # Convert true_coords_b to a tensor and add a dimension for the class
            true_coords_b = torch.tensor(true_coords_b, dtype=torch.float32)
            true_classes_b = torch.ones(true_coords_b.size(0), dtype=torch.float32)
            true_b = torch.cat((true_classes_b.unsqueeze(1), true_coords_b), dim=1)

            # Pad the ground truth vectors
            num_preds = pred_b.size(0)
            num_trues = true_b.size(0)
            if num_trues < num_preds:
                pad_size = num_preds - num_trues
                padding = torch.zeros((pad_size, 3), dtype=torch.float32)
                true_b = torch.cat((true_b, padding), dim=0)

            # Compute the cost matrix
            cost_matrix = self.compute_cost_matrix(true_b, pred_b)

            # Apply Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

            # Compute the loss
            matched_true = true_b[row_ind]
            matched_pred = pred_b[col_ind]

            coord_loss = F.mse_loss(matched_pred[:, 1:], matched_true[:, 1:])
            class_loss = F.binary_cross_entropy_with_logits(matched_pred[:, 0], matched_true[:, 0])

            total_loss += coord_loss + class_loss

        # Average the loss over the batch
        total_loss /= batch_size
        return total_loss

    def compute_cost_matrix(self, true_b, pred_b):
        num_preds = pred_b.size(0)
        num_trues = true_b.size(0)
        cost_matrix = torch.zeros((num_trues, num_preds))

        for i in range(num_trues):
            for j in range(num_preds):
                coord_dist = torch.norm(true_b[i, 1:] - pred_b[j, 1:])
                class_dist = F.binary_cross_entropy_with_logits(pred_b[j, 0], true_b[i, 0])
                cost_matrix[i, j] = coord_dist + class_dist

        return cost_matrix
