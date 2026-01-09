import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from captum.attr import IntegratedGradients, Saliency, GuidedGradCam
from captum.attr import visualization as viz

from net import Net
from transfer_learning import ResNetClassifier
from data_loader import CiliaClassificationDataset
from utils.params import *

warnings.filterwarnings("ignore")

def safe_visualize(attr, original, method, sign, title, axis, cmap, fig):
    if np.abs(attr).max() == 0 or (sign == "positive" and attr.max() <= 0):
        axis.text(0.5, 0.5, "No Attribution\n(All Zeros)", 
                 horizontalalignment='center', verticalalignment='center')
        axis.set_title(title)
        axis.axis('off')
    else:
        viz.visualize_image_attr(attr, original, method=method, sign=sign,
                                cmap=cmap, show_colorbar=True, 
                                title=title,
                                plt_fig_axis=(fig, axis), use_pyplot=False)

def main():
    output_dir = Path("interpretation_results_all_methods_net")
    output_dir.mkdir(exist_ok=True)

    model = Net().to(DEVICE)
    weights_path = "cilia_classifier_best_net.pth"
    
    if not Path(weights_path).exists():
        raise IOError("pth file does not exist")

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    ig = IntegratedGradients(model)
    saliency = Saliency(model)
    
    if isinstance(model, ResNetClassifier):
        target_layer = model.model.layer4[-1]
    else:
        target_layer = model.conv5
        
    guided_gc = GuidedGradCam(model, target_layer)

    data_dir = Path(DATA_SET_DIR)
    all_files = []
    for cls, sub in enumerate(["original", "inpainted"]):
        class_dir = data_dir / sub
        if class_dir.exists():
            files = sorted(list(class_dir.glob("*.png")))
            for f in files:
                all_files.append((f, cls))
    
    if not all_files:
        return

    dataset = CiliaClassificationDataset(all_files, angles=[0])
    
    print(f"Starting interpretation on {len(dataset)} images...")

    for i in range(len(dataset)):
        input_tensor, label = dataset[i]
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
        input_tensor.requires_grad = True

        original_path, _ = dataset.files[i]
        file_name = original_path.name
        target_class = label.item()
        class_name = "Original" if target_class == 0 else "Inpainted"

        attr_ig = ig.attribute(input_tensor, target=target_class)
        attr_saliency = saliency.attribute(input_tensor, target=target_class, abs=False)
        attr_ggc = guided_gc.attribute(input_tensor, target=target_class)

        original_img = input_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
        ig_img = attr_ig.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
        sal_img = attr_saliency.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
        ggc_img = attr_ggc.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original_img.squeeze(), cmap='gray')
        axes[0].set_title(f"Input: {file_name}\n({class_name})")
        axes[0].axis('off')

        safe_visualize(sal_img, original_img, "heat_map", "absolute_value", 
                      "Saliency", axes[1], "inferno", fig)

        safe_visualize(ig_img, original_img, "heat_map", "positive", 
                      "Integrated Gradients", axes[2], "Reds", fig)

        safe_visualize(ggc_img, original_img, "heat_map", "positive", 
                      "Guided Grad-CAM", axes[3], "viridis", fig)

        plt.tight_layout()
        
        save_path = output_dir / f"compare_{file_name}"
        fig.savefig(save_path)
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images.")

    print("Interpretation complete.")

if __name__ == "__main__":
    main()