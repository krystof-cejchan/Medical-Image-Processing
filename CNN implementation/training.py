import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import CiliaSegDataset
from u_net import UNet
from fn import DiceCrossEntropyLoss, dice_fn
from utils import EPOCHS, confusion_matrix_plot, curr_time, outputPlots

# ---------- setup ----------
_curr_time = curr_time
out_dir = Path(f"./out/test_predictions_{_curr_time}")
out_dir.mkdir(exist_ok=True, parents=True)
test_pred_dir = out_dir / "test_preds"
test_pred_dir.mkdir(exist_ok=True, parents=True)

THR = 0.5  # binarization threshold

# pytorch parameters for unet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = UNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = DiceCrossEntropyLoss(alpha=0.1)

# dataset split
ds = CiliaSegDataset()
train_size = int(0.7 * len(ds))
val_size   = int(0.15 * len(ds))
test_size  = len(ds) - train_size - val_size

train, val, test = random_split(ds, [train_size, val_size, test_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train, batch_size=8, shuffle=True)
val_loader   = DataLoader(val, batch_size=8, shuffle=False)
test_loader  = DataLoader(test, batch_size=8, shuffle=False)

hist = {
    "epoch": [], "train_loss": [], "val_loss": [], "val_dice": [],
    "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": [], "val_iou": []
}

# early stopping parameters
patience = 5
best_val_loss = float("inf")
epochs_no_improve = 0
early_stop = False
best_model_path = out_dir / "best_model.pt"

def _to_bin(t, thr=0.5):
    return (t >= thr).to(torch.int64)

def confusion_from_logits(logits, target, thr=0.5):
    """
    Returns TP, FP, TN, FN as Python ints.
    Works for tensors shaped (N, 1, H, W) or (N, H, W).
    """
    probs = torch.sigmoid(logits)
    pred = _to_bin(probs, thr)
    tgt  = _to_bin(target.float(), 0.5)

    # ensure same shape & int
    pred = pred.view(-1)
    tgt  = tgt.view(-1)

    tp = (pred & tgt).sum().item()
    fp = (pred & (1 - tgt)).sum().item()
    tn = ((1 - pred) & (1 - tgt)).sum().item()
    fn = ((1 - pred) & tgt).sum().item()
    return tp, fp, tn, fn

def metrics_from_confusion(tp, fp, tn, fn, eps=1e-8):
    acc  = (tp + tn) / max(tp + tn + fp + fn, eps)
    prec = tp / max(tp + fp, eps)
    rec  = tp / max(tp + fn, eps)
    f1   = 2 * prec * rec / max(prec + rec, eps)
    iou  = tp / max(tp + fp + fn, eps)
    return acc, prec, rec, f1, iou

def overlay(ax, img2d, mask2d, color, lw=2):
    ax.imshow(img2d, cmap="gray", vmin=0, vmax=1)
    ax.contour(mask2d, levels=[0.5], colors=[color], linewidths=lw)
    ax.axis("off")

def save_triptych(im2d, gt2d, pred2d, save_path):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(im2d, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Input")
    axs[0].axis("off")
    axs[1].imshow(gt2d, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("GT"); axs[1].axis("off")
    axs[2].imshow(pred2d, cmap="gray", vmin=0, vmax=1)
    axs[2].set_title("Pred"); axs[2].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# visualisation image 
vis_img, vis_mask = next(iter(val_loader))
vis_img, vis_mask = vis_img.to(device), vis_mask.to(device)

fig, axes = plt.subplots(1, EPOCHS, figsize=(EPOCHS, 3), constrained_layout=True)
if EPOCHS == 1:
    axes = [axes]

# training loop
for ep in range(EPOCHS):
    model.train()
    conf_matrix_path = out_dir / f"conf_matrix_epoch_{ep+1}.png"

    run_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        opt.zero_grad()
        # forward pass
        logits = model(images)
        # loss computation
        loss = loss_fn(logits, masks)
        # backpropagation
        loss.backward()
        # updating model parameters
        opt.step()
        run_loss += (loss.item() * images.size(0))
        confusion_matrix_plot(logits, masks, THR, conf_matrix_path)
        
    train_loss = run_loss / len(train_loader.dataset)

    # validation: loss, dice, metrics
    model.eval()
    val_loss, dice = 0.0, []
    TP = FP = TN = FN = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            val_loss += (loss.item() * images.size(0))

            dice.append(dice_fn(logits, masks).item())

            # confusion for metrics
            tp, fp, tn, fn = confusion_from_logits(logits, masks, THR)
            TP += tp
            FP += fp
            TN += tn
            FN += fn

    val_loss /= len(val_loader.dataset)
    val_dice = float(np.mean(dice))
    val_acc, val_prec, val_rec, val_f1, val_iou = metrics_from_confusion(TP, FP, TN, FN)

    hist["epoch"].append(ep + 1)
    hist["train_loss"].append(float(train_loss))
    hist["val_loss"].append(float(val_loss))
    hist["val_dice"].append(val_dice)
    hist["val_acc"].append(val_acc)
    hist["val_prec"].append(val_prec)
    hist["val_rec"].append(val_rec)
    hist["val_f1"].append(val_f1)
    hist["val_iou"].append(val_iou)

    print(
        f"Epoch {ep+1:2d} | "
        f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
        f"Dice: {val_dice:.4f} | Acc: {val_acc:.4f} | "
        f"P: {val_prec:.4f} R: {val_rec:.4f} F1: {val_f1:.4f} IoU: {val_iou:.4f}"
    )

    # early stopping implementation
    if val_loss < best_val_loss - 1e-5:
        # validation loss improved; we continue training the model
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"\tValidation loss improved. Model saved.")
    else:
        # model was not improved, but the patience has not been reached yet; we continue training
        epochs_no_improve += 1
        print(f"\tNo improvement for {epochs_no_improve} epochs.")

    if epochs_no_improve >= patience:
        # early stopping; model has not been improved for {patience} times. we stop training
        print(f"\tEarly stopping at epoch {ep+1} (no improvement for {patience} epochs).")
        early_stop = True
        break

    # visualisation of an image throughout the epochs
    with torch.no_grad():
        vis_logits = model(vis_img)
        p = torch.sigmoid(vis_logits)[0, 0].cpu().numpy()
        im = vis_img[0, 0].cpu().numpy()
        g  = vis_mask[0, 0].cpu().numpy()
    overlay(axes[ep], im, g,  "lime", 2)
    overlay(axes[ep], im, p > THR, "red", 1.2)
    axes[ep].set_title(f"Epoch {ep+1}")

fig.savefig(out_dir / "progress_contours.png", dpi=300)
plt.close(fig)

# load the best model, if early stop was applied
if early_stop:
    print(f"Restoring best model from epoch with val_loss={best_val_loss:.4f}")
model.load_state_dict(torch.load(best_model_path))

# evaluation of the model
model.eval()
test_loss, test_dice_list = 0.0, []
TP = FP = TN = FN = 0

with torch.no_grad():
    idx_base = 0
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        test_loss += loss.item() * x.size(0)
        test_dice_list.append(dice_fn(logits, y).item())

        # accumulate confusion
        tp, fp, tn, fn = confusion_from_logits(logits, y, THR)
        TP += tp
        FP += fp
        TN += tn
        FN += fn

        # save result images
        probs = torch.sigmoid(logits)
        preds = (probs >= THR).float()

        for b in range(x.size(0)):
            im2d = x[b, 0].detach().cpu().numpy()
            gt2d = y[b, 0].detach().cpu().numpy()
            pr2d = preds[b, 0].detach().cpu().numpy()
            save_triptych(im2d, gt2d, pr2d, test_pred_dir / f"test_{idx_base+b:05d}.png")
        idx_base += x.size(0)

test_loss /= len(test_loader.dataset)
test_dice = float(np.mean(test_dice_list))
test_acc, test_prec, test_rec, test_f1, test_iou = metrics_from_confusion(TP, FP, TN, FN)

outputPlots(hist, out_dir)