import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
import random

# Import vlastních modulů (musí být ve stejné složce)
from net import Net
from transfer_learning import ResNetClassifier
from data_loader import CiliaClassificationDataset
from utils.params import *

# ==========================================
# KONFIGURACE - ZDE UPRAV
# ==========================================
# Cesta k natrénovanému modelu (.pth soubor)
#MODEL_PATH = "cilia_classifier_best_resnet.pth" 
MODEL_PATH = "cilia_classifier_best_net.pth"

# Typ architektury: "resnet" nebo "net"
#MODEL_ARCH = "resnet" 
MODEL_ARCH = "net"

# Nastavení seedu pro reprodukovatelnost (pokud jsi ho použil při tréninku)
SEED = 42
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():
    """
    Znovu načte a rozdělí data stejně jako při tréninku, 
    abychom získali správnou TEST sadu.
    """
    data_dir = Path(DATA_SET_DIR)
    all_files = [] 
    for cls, sub in enumerate(["original", "inpainted"]):
        class_dir = data_dir / sub
        # Seřazení je důležité před shuffle pro determinismus
        files = sorted(list(class_dir.glob("*.png")))
        for img_path in files:
            all_files.append((img_path, cls))

    if not all_files:
        raise RuntimeError(f"Nenalezena žádná data ve složce {DATA_SET_DIR}")

    # Náhodné zamíchání (musí odpovídat seedu při tréninku pro přesnou shodu)
    random.shuffle(all_files)

    total_count = len(all_files)
    train_count = int(total_count * TRAIN_SPLIT)
    val_count = int(total_count * VAL_SPLIT)
    
    # Nás zajímá pouze testovací sada (zbytek ignorujeme)
    test_files = all_files[train_count + val_count :]
    
    print(f"Načteno {len(test_files)} souborů pro testování.")
    
    # Pro testování používáme angle=0 (bez augmentace)
    test_dataset = CiliaClassificationDataset(test_files, angles=[0])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_loader, ["Original", "Inpainted"]

def evaluate():
    set_seed(SEED)
    
    # 1. Příprava dat
    test_loader, class_names = load_data()
    
    # 2. Inicializace modelu
    print(f"Načítám architekturu: {MODEL_ARCH}")
    if MODEL_ARCH == "resnet":
        model = ResNetClassifier(num_classes=2, input_channels=1, freeze_base=False)
    elif MODEL_ARCH == "net":
        model = Net()
    else:
        raise ValueError("Neznámý typ architektury. Použij 'resnet' nebo 'net'.")
    
    # 3. Načtení vah
    print(f"Načítám váhy ze souboru: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Soubor {MODEL_PATH} neexistuje.")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 4. Infenerece (sběr predikcí)
    all_preds = []
    all_targets = []
    
    print("Spouštím evaluaci...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # Získání predikované třídy (argmax)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # 5. Výpočet metrik
    # average='binary' předpokládá, že třída 1 (Inpainted) je pozitivní
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # IoU (Jaccard Score) pro binární klasifikaci: TP / (TP + FP + FN)
    iou = jaccard_score(all_targets, all_preds, average='binary')

    # Výpis číselných hodnot
    print("\n" + "="*30)
    print("VÝSLEDKY EVALUACE")
    print("="*30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("-" * 30)

    # 6. Confusion Matrix v procentech
    cm = confusion_matrix(all_targets, all_preds)
    
    # Normalizace na procenta (podle řádků - True Labels)
    # Každý řádek bude dávat v součtu 100 %
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Vykreslení a uložení Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    
    # Přidání znaku % k anotacím (trik pro matplotlib text)
    for t in plt.gca().texts:
        t.set_text(t.get_text() + " %")
        
    plt.xlabel('Predikovaná třída')
    plt.ylabel('Skutečná třída')
    plt.title(f'Confusion Matrix (%) - {MODEL_ARCH}')
    
    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"Graf Confusion Matrix uložen jako '{save_path}'")
    plt.close()

if __name__ == "__main__":
    evaluate()