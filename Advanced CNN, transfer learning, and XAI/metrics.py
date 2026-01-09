import pandas as pd
import torch


class Metrics:
    def __init__(self, pred_logits, target, threshold=0.5):
        self.pred_logits = pred_logits
        self.target = target
        self.threshold = threshold
    
    def get_confusion_components(self):
        """
            TP true positive
            FP false positive
            TN true negative
            FN false negative
        """

        # logits to probability (0-1)
        probs = torch.sigmoid(self.pred_logits)

        # True/False → float (1.0/0.0)
        preds = (probs > self.threshold).float()

        # other functions like f1 etc. work with float, so we work with float here too
        target = self.target.float()

        # True Positive: ground truth is 1 and unet marked the pixel as 1 as well
        TP = torch.sum((preds == 1) & (target == 1)).float()

        # False Positive: unet says 1, groundtruth is 0
        FP = torch.sum((preds == 1) & (target == 0)).float()

        # True Negative
        TN = torch.sum((preds == 0) & (target == 0)).float()

        # False Negative
        FN = torch.sum((preds == 0) & (target == 1)).float()

        return TP, FP, TN, FN



    def compute_metrics(self, eps=1e-6):
        TP, FP, TN, FN = self.get_confusion_components()

        accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall    = TP / (TP + FN + eps)
        f1_score  = 2 * precision * recall / (precision + recall + eps)

        metrics = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
        }
        return metrics
    
    def show_metrics_table(self, pred_logits, target, threshold=0.5):
        metrics = self.compute_metrics(pred_logits, target, threshold)
        df = pd.DataFrame([metrics])
        print(df.style.format(precision=4))
        return df