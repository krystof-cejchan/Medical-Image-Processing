import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from torch.utils.data import DataLoader

from siamese_net import SiameseNetwork
from siamese_ds import SiameseDataset
import utils.params as params

def main():
    device = params.DEVICE
    
    # 1. Načtení dat (jen seznam, nepotřebujeme páry, chceme bod po bodu)
    data_dir = Path(params.DATA_SET_DIR)
    all_files = [] 
    for cls, sub in enumerate(["original", "inpainted"]):
        class_dir = data_dir / sub
        for img_path in class_dir.glob("*.png"):
            all_files.append((img_path, cls))

    # Použijeme dataset jen pro načtení (batch loading je rychlejší)
    # Trik: SiameseDataset v __getitem__ vrací páry, my chceme jen jeden.
    # Uděláme si jednoduchý loader jen pro vizualizaci.
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, files): self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self, i):
            path, label = self.files[i]
            # Zkopírováno z siamese_utils logic
            from PIL import Image
            img = Image.open(path).convert("L").resize((256, 256), Image.BILINEAR)
            img = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(img)[None, ...], label

    dataset = SimpleDataset(all_files)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Načtení modelu
    model = SiameseNetwork(input_channels=1, embedding_dim=128).to(device)
    if not Path("siamese_best.pth").exists():
        print("Model 'siamese_best.pth' neexistuje. Spusť trénování.")
        return
    model.load_state_dict(torch.load("siamese_best.pth", map_location=device))
    model.eval()

    # 3. Extrakce embeddingů
    embeddings = []
    labels = []
    
    print("Generuji embeddingy...")
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            # Voláme forward_one, protože chceme jen embedding jednoho obrázku
            emb = model.forward_one(imgs)
            embeddings.append(emb.cpu().numpy())
            labels.append(lbls.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # 4. Redukce dimenzí (PCA a t-SNE)
    print("Počítám PCA...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    print("Počítám t-SNE (to může chvíli trvat)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter_without_progress=1000, init='pca', learning_rate='auto')
    embeddings_tsne = tsne.fit_transform(embeddings)

    # 5. Vykreslení
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Barvy: 0 = Original (Modrá), 1 = Inpainted (Červená)
    colors = ['blue', 'red']
    class_names = ['Original', 'Inpainted']

    # PCA Plot
    for cls in [0, 1]:
        idx = labels == cls
        ax[0].scatter(embeddings_pca[idx, 0], embeddings_pca[idx, 1], 
                      c=colors[cls], label=class_names[cls], alpha=0.6, edgecolors='none')
    ax[0].set_title("PCA Vizualizace Embeddingů")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # t-SNE Plot
    for cls in [0, 1]:
        idx = labels == cls
        ax[1].scatter(embeddings_tsne[idx, 0], embeddings_tsne[idx, 1], 
                      c=colors[cls], label=class_names[cls], alpha=0.6, edgecolors='none')
    ax[1].set_title("t-SNE Vizualizace Embeddingů")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("siamese_embeddings_vis.png")
    plt.show()
    print("Vizualizace uložena do 'siamese_embeddings_vis.png'.")

if __name__ == "__main__":
    main()