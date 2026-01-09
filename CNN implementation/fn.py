import torch
import torch.nn as nn

from metrics import Metrics


def accuracy_fn(pred_logits, target, threshold=0.5, eps=1e-6):
    """eps = small number to prevent zero division problem
    """
    TP, FP, TN, FN = Metrics(pred_logits, target, threshold).get_confusion_components()
    return (TP + TN) / (TP + TN + FP + FN + eps)


def precision_fn(pred_logits, target, threshold=0.5, eps=1e-6):
    TP, FP, _, _ = Metrics(pred_logits, target, threshold).get_confusion_components()
    return TP / (TP + FP + eps)


def recall_fn(pred_logits, target, threshold=0.5, eps=1e-6):
    TP, _, _, FN = Metrics(pred_logits, target, threshold).get_confusion_components()
    return TP / (TP + FN + eps)


def f1_fn(pred_logits, target, threshold=0.5, eps=1e-6):
    precision = precision_fn(pred_logits, target, threshold, eps)
    recall = recall_fn(pred_logits, target, threshold, eps)
    return 2 * precision * recall / (precision + recall + eps)


def dice_fn(pred_logits, target, eps=1e-6):
    probs = torch.sigmoid(pred_logits)
    preds = (probs > 0.5).float()
    target = target.float()

    intersection = torch.sum(preds * target)
    union = torch.sum(preds) + torch.sum(target)
    dice = (2 * intersection + eps) / (union + eps)
    return dice

class DiceCrossEntropyLoss(nn.Module):
    """
    Dice + BCE loss
    alpha = weight of BCE (0.5 =   50% BCE, 50% Dice)
    """
    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target):
        target = target.float()

        # BCE
        bce_loss = self.bce(pred_logits, target)

        # Dice
        probs = torch.sigmoid(pred_logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        dice_score = (2 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice_score.mean()

        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss
