import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image

class SiameseDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        
        self.grouped_files = {}
        for img_path, label in file_list:
            if label not in self.grouped_files:
                self.grouped_files[label] = []
            self.grouped_files[label].append((img_path, label))

    def __len__(self):
        return len(self.file_list)

    def _load_image(self, path):
        img = Image.open(path).convert("L").resize((256, 256), Image.BILINEAR) #greyscale 256x256
        img = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img)[None, ...] # (1, H, W)
        return img_t

    def __getitem__(self, index):
        img1_path, label1 = self.file_list[index]
        
        # pozitiní/negativní pár (50/50)
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # Pozitivní pár
            class_files = self.grouped_files[label1]
            img2_path, label2 = random.choice(class_files)
            target = torch.tensor([0], dtype=torch.float32) # 0 = stejné
        else:
            # neg. pár
            all_labels = list(self.grouped_files.keys())
            diff_labels = [l for l in all_labels if l != label1]
            
            if not diff_labels:
                 img2_path, label2 = random.choice(self.grouped_files[label1])
                 target = torch.tensor([0], dtype=torch.float32)
            else:
                label2 = random.choice(diff_labels)
                img2_path, label2 = random.choice(self.grouped_files[label2])
                target = torch.tensor([1], dtype=torch.float32) # 1 = různé

        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        return img1, img2, target

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label 0 pro podobné, 1 pro rozdílné
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive