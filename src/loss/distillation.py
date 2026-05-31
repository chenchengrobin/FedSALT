import torch.nn as nn
import torch.nn.functional as F

from src.config import *
# 蒸馏损失
class distillation_loss(nn.Module):
    def __init__(self, reduction='mean', temperature=1, alpha=1):
        super(distillation_loss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, client_logit, targets, teacher_logit=None):
        client_logit = client_logit.to(DEVICE)
        targets = targets.to(DEVICE)
        if teacher_logit is not None:
            teacher_logit = teacher_logit.to(DEVICE)

        if teacher_logit is not None:
            student_soft = F.log_softmax(client_logit / self.temperature, dim=1)
            teacher_logit = F.softmax(teacher_logit / self.temperature, dim=1)  # 学生软标签 -> 概率+蒸馏温度
            kl_loss = F.kl_div(student_soft, teacher_logit.detach(), reduction='batchmean') * (self.temperature ** 2)
        else:
            kl_loss = 0

        argmax_targets = targets.argmax(dim=1) if targets.dim() > 1 else targets
        ce_loss = F.cross_entropy(client_logit, argmax_targets, reduction=self.reduction)
        if teacher_logit is not None:
            loss = ce_loss * self.alpha + kl_loss * (1 - self.alpha)
        else:
            loss = ce_loss
        return loss