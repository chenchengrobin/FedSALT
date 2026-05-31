import torch.nn as nn
import torch.nn.functional as F

from src.config import *


class hybrid_loss(nn.Module):
    def __init__(self, temperature=5, use_weight=True):
        super(hybrid_loss, self).__init__()
        self.temperature = temperature
        self.use_weight = use_weight

    def forward(self, logits, targets, scores, is_federated_mask):
        scores = torch.tensor(scores, dtype=torch.float32, device=DEVICE) if not isinstance(scores, torch.Tensor) else scores
    
        is_federated_mask = is_federated_mask.bool()
        is_federated = is_federated_mask
        is_teacher = ~is_federated_mask

        # teacher loss
        teacher_logits = logits[is_teacher]
        teacher_targets = targets[is_teacher]
        teacher_loss = self._computed_teacher_loss(teacher_logits, teacher_targets).mean() if len(teacher_logits) > 0 else torch.tensor(0.0, device=DEVICE)

        # federated loss
        federated_logits = logits[is_federated]
        federated_targets = targets[is_federated]
        federated_scores = scores[is_federated]

        if len(federated_logits) > 0:
            federated_loss = self._computed_federated_loss(federated_logits, federated_targets)
            if self.use_weight:
                federated_loss = (federated_loss * federated_scores).sum()
            else:
                federated_loss = federated_loss.mean()
        else:
            federated_loss = torch.tensor(0.0, device=DEVICE)

        total_loss = teacher_loss + federated_loss
        return total_loss

    @staticmethod
    def _computed_teacher_loss(logit, target):
        loss = torch.nn.functional.cross_entropy(logit, target, reduction='none') if target.dim() == 1 else (
            torch.nn.functional.cross_entropy(logit, target.argmax(dim=1), reduction='none'))
        return loss

    def _computed_federated_loss(self, logit, target):
        if target.dim() == 1:
            loss = torch.nn.functional.cross_entropy(logit, target, reduction='none')
        else:
            logit_prob = torch.log_softmax(logit / self.temperature, dim=1)
            target_prob = torch.softmax(target / self.temperature, dim=1)
            loss = torch.nn.functional.kl_div(logit_prob, target_prob, reduction='none') * (self.temperature ** 2)
            loss = loss.sum(dim=1)
        return loss


def compute_consistency(logit_list: torch.Tensor):
    N, T = logit_list.shape
    device = logit_list.device

    change_count = torch.zeros(N, device=device)
    for t in range(T - 1):
        change = (logit_list[:, t] != logit_list[:, t + 1]).float()
        change_count += change
    sample_consistency = 1.0 - change_count / (T - 1 + 1e-10)

    same_last_two = (logit_list[:, -1] == logit_list[:, -2]).float()
    overall_latest_consistency = same_last_two.mean().item()

    print(sample_consistency.dtype)
    return sample_consistency, overall_latest_consistency
