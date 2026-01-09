import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
import random
import torch.nn.functional as F

# Import vlastních modulů
from siamese_net import SiameseNetwork
from siamese_ds import SiameseDataset
from utils.params import *

# ==========================================
# KONFIGURACE
# ==========================================
MODEL_PATH = "siamese_best.pth"
MARGIN = 1.0
THRESHOLD = MARGIN / 2.0  # Vzdálenost < 0.5 => Stejné (0), Vzdálenost > 0.5 => Různé (1)
SEED = 42
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_test_data():
    """
    Načte data a vybere testovací část. 
    Pokud jsi v tréninku používal 80/20 split, 
    zde vezmeme posledních 20% jako "Test" (nebo validační, pokud jiná neexistuje),
    abychom simulovali evaluaci na datech, která model 'neviděl' v hlavní části tréninku.
    """
    data_dir = Path(DATA_SET_DIR)
    all_files = [] 
    for cls, sub in enumerate(["original", "inpainted"]):
        class_dir = data_dir / sub
        # Seřadíme pro determinismus před zamícháním
        for img_path in sorted(class_dir.glob("*.png")):
            all_files.append((img_path, cls))

    if not all_files:
        raise RuntimeError("Žádná data nenalezena.")

    # Zamícháme stejně jako při tréninku
    random.shuffle(all_files)
    
    # Rozdělení - vezmeme posledních 20% jako testovací sadu
    # (pokud chceš striktně oddělený test set, měl bys upravit i trénovací skript, 
    # aby trénoval např. jen na prvních 70%)
    split_idx = int(len(all_files) * 0.8)
    test_files = all_files[split_idx:]
    
    print(f"Načteno {len(test_files)} párů pro evaluaci.")
    
    test_ds = SiameseDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_loader

def evaluate():
    set_seed(SEED)
    
    # 1. Příprava dat
    test_loader = load_test_data()
    
    # 2. Inicializace modelu
    print("Načítám Siamskou síť...")
    device = DEVICE
    model = SiameseNetwork(input_channels=1, embedding_dim=128).to(device)
    
    # 3. Načtení vah
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Soubor {MODEL_PATH} neexistuje. Nejdříve spusť trénování.")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    all_distances = []
    
    print("Spouštím inferenci...")
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            output1, output2 = model(img1, img2)
            
            # Výpočet euklidovské vzdálenosti
            euclidean_distance = F.pairwise_distance(output1, output2)
            
            # Predikce: 0 (Stejné) pokud dist < threshold, jinak 1 (Různé)
            # Poznámka: Dataset vrací 0 pro stejné, 1 pro různé.
            preds = (euclidean_distance > THRESHOLD).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
            all_distances.extend(euclidean_distance.cpu().numpy())

    # 4. Výpočet metrik
    # Label 0: Stejné (Same)
    # Label 1: Různé (Different)
    
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Precision/Recall/F1 pro třídu "Different" (1)
    precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)

    print("\n" + "="*30)
    print(f"VÝSLEDKY EVALUACE (Threshold={THRESHOLD})")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (Different): {precision:.4f}")
    print(f"Recall (Different):    {recall:.4f}")
    print(f"F1-Score (Different):  {f1:.4f}")
    print("-" * 30)
    
    # Detailní report pro obě třídy
    print("\nDetailní klasifikační report:")
    print(classification_report(all_targets, all_preds, target_names=["Stejné (0)", "Různé (1)"]))

    # 5. Confusion Matrix v procentech
    cm = confusion_matrix(all_targets, all_preds)
    
    # Normalizace na procenta
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Vykreslení
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Purples", 
                xticklabels=["Stejné (Same)", "Různé (Diff)"], 
                yticklabels=["Stejné (Same)", "Různé (Diff)"])
    
    for t in plt.gca().texts:
        t.set_text(t.get_text() + " %")
        
    plt.xlabel('Predikovaná třída')
    plt.ylabel('Skutečná třída')
    plt.title('Confusion Matrix (%) - Siamská síť')
    
    save_path = "siamese_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"Graf Confusion Matrix uložen jako '{save_path}'")
    plt.close()
    
    # Volitelně: Histogram vzdáleností (užitečné pro ladění threshold)
    plt.figure(figsize=(10, 5))
    
    # Rozdělíme vzdálenosti podle skutečného labelu
    dists_same = [d for d, t in zip(all_distances, all_targets) if t == 0]
    dists_diff = [d for d, t in zip(all_distances, all_targets) if t == 1]
    
    plt.hist(dists_same, bins=30, alpha=0.7, label='Stejné páry (Target 0)', color='green')
    plt.hist(dists_diff, bins=30, alpha=0.7, label='Různé páry (Target 1)', color='red')
    plt.axvline(THRESHOLD, color='k', linestyle='dashed', linewidth=2, label=f'Threshold ({THRESHOLD})')
    
    plt.title('Distribuce vzdáleností párů')
    plt.xlabel('Euklidovská vzdálenost')
    plt.ylabel('Počet')
    plt.legend()
    plt.savefig('siamese_distance_dist.png')
    print("Graf distribuce vzdáleností uložen jako 'siamese_distance_dist.png'")
    plt.close()

if __name__ == "__main__":
    evaluate()