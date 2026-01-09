from time import gmtime, strftime

from matplotlib import pyplot as plt
import torch

from metrics import Metrics

import seaborn as sns



IMG_SIZE = 256

UNET_BASE_CHANNEL_IN = 1; UNET_BASE_CHANNEL_OUT = 1
UNET_BASE = 16

EPOCHS = 120 # early stopping na 23


curr_time = strftime("%y-%m-%d_%H-%M-%S", gmtime())



# output
def outputPlots(hist, out_dir):
    # Loss curves
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["epoch"], hist["train_loss"], label="Train Loss")
    plt.plot(hist["epoch"], hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150); plt.close()

    # Dice curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["epoch"], hist["val_dice"], label="Val Dice")
    plt.xlabel("Epoch"); plt.ylabel("Soft Dice")
    plt.title("Validation Dice")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "dice_curve.png", dpi=150); plt.close()

    # Accuracy curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["epoch"], hist["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=150); plt.close()

    # Precision, Recall, F1 curves
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["epoch"], hist["val_prec"], label="Val Precision")
    plt.plot(hist["epoch"], hist["val_rec"],  label="Val Recall")
    plt.plot(hist["epoch"], hist["val_f1"],   label="Val F1")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.title("Validation Precision / Recall / F1")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "prf1_curves.png", dpi=150); plt.close()

    # IoU curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["epoch"], hist["val_iou"], label="Val IoU")
    plt.xlabel("Epoch"); plt.ylabel("IoU")
    plt.title("Validation IoU")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "iou_curve.png", dpi=150); plt.close()

def _to_int_scalar(x):
    # Works whether x is a Python number or a 0-dim tensor
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)
def confusion_matrix_plot(pred_logits, target, threshold=0.5, save_path=None):
    TP, FP, TN, FN = Metrics(pred_logits, target, threshold).get_confusion_components()
    TP = _to_int_scalar(TP)
    FP = _to_int_scalar(FP)
    TN = _to_int_scalar(TN)
    FN = _to_int_scalar(FN)
    confusion_matrix = torch.tensor([[TN, FP],
                                     [FN, TP]], dtype=torch.int64)

    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix.numpy(), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()
    return confusion_matrix
