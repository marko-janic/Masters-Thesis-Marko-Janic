import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class ParticlePickingLoss(nn.Module):
    def __init__(self):
        super(ParticlePickingLoss, self).__init__()

    def forward(self, true_coords, true_classes, pred_coords, pred_classes):
        # Pad the ground truth vectors
        num_preds = pred_coords.size(0)
        num_trues = true_coords.size(0)
        if num_trues < num_preds:
            pad_size = num_preds - num_trues
            true_coords = F.pad(true_coords, (0, 0, 0, pad_size), value=0)
            true_classes = F.pad(true_classes, (0, pad_size), value=0)

        # Compute the cost matrix
        cost_matrix = self.compute_cost_matrix(true_coords, true_classes, pred_coords, pred_classes)

        # Apply Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

        # Compute the loss
        matched_true_coords = true_coords[row_ind]
        matched_true_classes = true_classes[row_ind]
        matched_pred_coords = pred_coords[col_ind]
        matched_pred_classes = pred_classes[col_ind]

        coord_loss = F.mse_loss(matched_pred_coords, matched_true_coords)
        class_loss = F.binary_cross_entropy_with_logits(matched_pred_classes, matched_true_classes.float())

        total_loss = coord_loss + class_loss
        return total_loss

    def compute_cost_matrix(self, true_coords, true_classes, pred_coords, pred_classes):
        num_preds = pred_coords.size(0)
        num_trues = true_coords.size(0)
        cost_matrix = torch.zeros((num_trues, num_preds))

        for i in range(num_trues):
            for j in range(num_preds):
                coord_dist = torch.norm(true_coords[i] - pred_coords[j])
                class_dist = F.binary_cross_entropy_with_logits(pred_classes[j], true_classes[i].float())
                cost_matrix[i, j] = coord_dist + class_dist

        return cost_matrix
