import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import random
import matplotlib.pyplot as plt # Import pro grafy
import numpy as np

from siamese_ds import ContrastiveLoss, SiameseDataset
from siamese_net import SiameseNetwork
from utils.params import *

def calculate_accuracy(output1, output2, label, margin):
    threshold = margin / 2.0
    euclidean_distance = F.pairwise_distance(output1, output2)
    
    predictions = (euclidean_distance > threshold).float()
    
    correct = (predictions == label).sum().item()
    return correct

def main():
    data_dir = Path(DATA_SET_DIR)
    all_files = [] 
    for cls, sub in enumerate(["original", "inpainted"]):
        class_dir = data_dir / sub
        for img_path in class_dir.glob("*.png"):
            all_files.append((img_path, cls))

    if not all_files:
        raise RuntimeError("Žádná data nenalezena.")

    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    train_ds = SiameseDataset(train_files)
    val_ds = SiameseDataset(val_files)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    device = DEVICE
    margin = 1.0 # Margin pro Contrastive Loss
    
    model = SiameseNetwork(input_channels=1, embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)


    # Seznamy pro ukládání metrik
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    epochs = 50 
    best_loss = float('inf')

    patience_counter = 0

    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0
        
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # accuracy
            running_correct += calculate_accuracy(out1, out2, label, margin)
            total_train += label.size(0)
            
        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_correct / total_train
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
                val_loss += loss.item()
                
                val_correct += calculate_accuracy(out1, out2, label, margin)
                total_val += label.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / total_val
        
        # Uložení do historie
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.2f}")
        
        if val_loss < best_loss - 1e-5:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "siamese_best.pth")
            print("Loss zlepšena - Model uložen.")
        else:
            patience_counter += 1
            print(f"Validační loss se nezlepšil. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"*** EARLY STOPPING *** Dosažena patience ({PATIENCE}). Konec trénování.")
            break

    plot_metrics(history)

def plot_metrics(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Graf Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Graf Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Grafy uloženy do 'training_metrics.png'")
    plt.show()

if __name__ == "__main__":
    main()