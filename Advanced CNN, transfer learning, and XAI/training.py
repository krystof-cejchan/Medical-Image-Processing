from datetime import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import random
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from data_loader import CiliaClassificationDataset
from metrics import Metrics
from net import Net
from utils.params import *
from transfer_learning import ResNetClassifier



data_dir = Path(DATA_SET_DIR)
all_files = [] 
for cls, sub in enumerate(["original", "inpainted"]):
    class_dir = data_dir / sub
    for img_path in class_dir.glob("*.png"):
        all_files.append((img_path, cls))

if not all_files:
    raise RuntimeError(f"Nenalezena žádná data ve složce {DATA_SET_DIR}")

random.shuffle(all_files)

total_count = len(all_files)
train_count = int(total_count * TRAIN_SPLIT)
val_count = int(total_count * VAL_SPLIT)
test_count = total_count - train_count - val_count 

train_files = all_files[:train_count]
val_files = all_files[train_count : train_count + val_count]
test_files = all_files[train_count + val_count :]

print(f"Celkem unikátních souborů: {total_count}")
print(f"\t Trénovací sada: {len(train_files)} souborů")
print(f"\t Validační sada: {len(val_files)} souborů")
print(f"\t Testovací sada: {len(test_files)} souborů")

train_dataset = CiliaClassificationDataset(train_files, angles=[90, 180, 270])
val_dataset = CiliaClassificationDataset(val_files, angles=[0])
test_dataset = CiliaClassificationDataset(test_files, angles=[0])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = ResNetClassifier(num_classes=2, input_channels=1, freeze_base=False).to(DEVICE)
#model = Net().to(DEVICE)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
history = {
    "train_loss": [],
    "val_loss": [],
    "accuracy": []
}
best_val_loss = float('inf') 
patience_counter = 0
best_model_weights = None

for epoch in range(EPOCHS):
    model.train() 
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    
    model.eval() 
    val_loss = 0.0
    
    # Seznamy pro uložení všech predikcí a labelů v této epoše
    val_logits_list = []
    val_targets_list = []
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += (loss.item() * inputs.size(0))
     
            binary_logits = outputs[:, 1] - outputs[:, 0]
            
            val_logits_list.append(binary_logits)
            val_targets_list.append(labels)

    val_loss = val_loss / len(val_dataset)
    
    all_val_logits = torch.cat(val_logits_list)
    all_val_targets = torch.cat(val_targets_list)
    
    metrics_evaluator = Metrics(all_val_logits, all_val_targets)
    results = metrics_evaluator.compute_metrics()
    
    acc = results["Accuracy"].item() * 100
    prec = results["Precision"].item()
    rec = results["Recall"].item()
    f1 = results["F1-Score"].item()
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["accuracy"].append(acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
          f"Acc: {acc:.2f}% | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    if val_loss < best_val_loss - 1e-5:
        print(f"({best_val_loss:.4f} -\t {val_loss:.4f}). Ukládám model.")
        best_val_loss = val_loss
        patience_counter = 0
        best_model_weights = copy.deepcopy(model.state_dict())
    else:
        patience_counter += 1
        print(f"  \t Validační loss se nezlepšil. Patience: {patience_counter}/{PATIENCE}")
    
    if patience_counter >= PATIENCE:
        print(f"*** EARLY STOPPING *** Dosažena trpělivost {PATIENCE}. Ukončuji trénování.")
        break

epochs_range = range(1, len(history["train_loss"]) + 1)


# Graph 1: Loss (Train vs Validation)
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, history["train_loss"], label='Training Loss')
plt.plot(epochs_range, history["val_loss"], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('graph_loss_net.png')
plt.close()

# Graph 2
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, history["accuracy"], label='Accuracy (%)')
plt.title('accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score (%)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_net.png')
plt.close()

if best_model_weights:
    # Načteme nejlepší model pro finální testování
    model.load_state_dict(best_model_weights)
    model.eval() 

    pred_dir = Path(PREDICTION_DIR)
    dir_original = pred_dir / "predicted_original"
    dir_inpainted = pred_dir / "predicted_inpainted"
    
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
        
    dir_original.mkdir(parents=True, exist_ok=True)
    dir_inpainted.mkdir(parents=True, exist_ok=True)
    
    test_logits_list = []
    test_targets_list = []
    
    file_index = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            
            binary_logits = outputs[:, 1] - outputs[:, 0]
            test_logits_list.append(binary_logits)
            test_targets_list.append(labels)
            
           
            _, predicted = torch.max(outputs.data, 1) 
            
            for i in range(len(labels)):
                pred_label = predicted[i].item()
                
                original_path, _ = test_files[file_index]
                original_filename = original_path.name
                
                if pred_label == 0:
                    dest_dir = dir_original
                else:
                    dest_dir = dir_inpainted
                
                dest_path = dest_dir / original_filename
                shutil.copy(str(original_path), str(dest_path))
                file_index += 1

    all_test_logits = torch.cat(test_logits_list)
    all_test_targets = torch.cat(test_targets_list)
    
    metrics_evaluator = Metrics(all_test_logits, all_test_targets)
    
    final_results = metrics_evaluator.compute_metrics()
    print("\n--- Výsledky na testovací sadě ---")
    for k, v in final_results.items():
        print(f"{k}: {v.item():.4f}")

    torch.save(best_model_weights, "cilia_classifier_best_resnet.pth")
