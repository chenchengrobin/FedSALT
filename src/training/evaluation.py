from src.dataset import get_dataloader
from torchmetrics.classification import Accuracy
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall
from torchmetrics.classification import F1Score
from torchmetrics.classification import MatthewsCorrCoef
from torchmetrics.classification import ConfusionMatrix

import torch.nn as nn
import torch.nn.functional as F
from src.config import *

accuracy = Accuracy(task="multiclass", num_classes=CLASS, average="micro").to(DEVICE)
precision = Precision(task="multiclass", num_classes=CLASS, average="micro").to(DEVICE)
recall = Recall(task="multiclass", num_classes=CLASS, average="micro").to(DEVICE)
f1 = F1Score(task="multiclass", num_classes=CLASS, average="micro").to(DEVICE)
mcc = MatthewsCorrCoef(task="multiclass", num_classes=CLASS).to(DEVICE)
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=CLASS).to(DEVICE)


@torch.inference_mode()
def extract_feature(model, dataloader, computed_loss=False):
    model.eval().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    logits, features, targets, loss = [], [], [], 0.0

    for batch in dataloader:
        data, batch_target = batch[:2]
        data = data.to(DEVICE, non_blocking=True)
        batch_target = batch_target.to(DEVICE, non_blocking=True)
        feature, logit = model(data)
        if computed_loss:
            loss += criterion(logit, batch_target)
        logits.append(logit)
        features.append(feature)
        targets.append(batch_target)

    logits = torch.cat(logits, dim=0)
    targets = torch.cat(targets, dim=0)
    features = torch.cat(features, dim=0)

    if computed_loss:
        return logits, targets, features, loss / len(dataloader)

    return logits, targets, features, loss


def evaluating_model_performance(target, predict):
    acc = accuracy(predict, target)
    prec = precision(predict, target)
    rec = recall(predict, target)
    f1s = f1(predict, target)
    test_mcc = mcc(predict, target)
    result = [acc, prec, rec, f1s, test_mcc]
    return result


def evaluate(model, dataset, mode='common'):
    dataloader = get_dataloader(dataset=dataset, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    test_logits, test_targets, test_features, loss = extract_feature(model, dataloader, computed_loss=mode == 'loss')
    test_predict = torch.argmax(test_logits, dim=1)
    if mode == 'extract':
        return test_logits, test_features, test_targets
    test_result = evaluating_model_performance(test_targets, test_predict)
    test_result = torch.tensor(test_result).to(DEVICE)
    if mode == 'simple':
        return test_result
    if mode == 'loss':
        return test_result[0], loss
