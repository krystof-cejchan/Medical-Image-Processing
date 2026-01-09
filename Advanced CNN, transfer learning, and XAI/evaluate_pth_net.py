import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score, accuracy_score

from data_loader import CiliaClassificationDataset
from net import Net
from siamese_net import SiameseNetwork
from transfer_learning import ResNetClassifier
from utils.params import *



def get_test_loader():
    data_dir = Path(DATA_SET_DIR)
    all_files = [] 
    for cls, sub in enumerate(["original", "inpainted"]):
        class_dir = data_dir / sub
        # Seřadíme pro determinističnost před shufflem
        files = sorted(list(class_dir.glob("*.png")))
        for img_path in files:
            all_files.append((img_path, cls))

    if not all_files:
        raise RuntimeError(f"Nenalezena žádná data v {DATA_SET_DIR}")

    # Nastavíme seed pro reprodukovatelnost tohoto testu
    random.seed(42) 
    random.shuffle(all_files)

    total_count = len(all_files)
    train_count = int(total_count * TRAIN_SPLIT)
    val_count = int(total_count * VAL_SPLIT)
    
    # Vezmeme testovací část
    test_files = all_files[train_count + val_count :]
    
    print(f"Testovací sada obsahuje {len(test_files)} obrázků.")
    
    test_dataset = CiliaClassificationDataset(test_files, angles=[0])
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_model(model, model_path, test_loader, device):
    print(f"\n--- Vyhodnocuji model: {model_path} ---")
    
    # Inicializace modelu
    # Zde musíme zvolit správnou architekturu (ResNetClassifier nebo Net)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(e)
        return

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Výpočet metrik
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    iou = jaccard_score(all_labels, all_preds, average='binary') # IoU pro klasifikaci
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Original', 'Inpainted'], 
                yticklabels=['Original', 'Inpainted'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\n{Path(model_path).name}')
    
    save_name = f"cm_{Path(model_path).stem}.png"
    plt.savefig(save_name)
    print(f"Confusion Matrix uložen jako {save_name}")
    plt.close()

def main():

    pth_file = "./cilia_classifier_best_net.pth"
    #model = ResNetClassifier(num_classes=2, input_channels=1, freeze_base=False)
    #model = SiameseNetwork()
    model = Net()

    try:
        test_loader = get_test_loader()
    except Exception as e:
        print(e)
        return

    evaluate_model(model, pth_file, test_loader, DEVICE)

if __name__ == "__main__":
    main()