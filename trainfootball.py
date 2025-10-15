import os, json, time, random, numpy as np, warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from model import HybridResNetViT  # Uncomment your model import

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = r"C:\Users\T8635\Desktop\project3\Data7classes\images"
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
VAL_SPLIT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Seed
# -----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

# -----------------------------
# Safe loader
# -----------------------------
def safe_image_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        try: img.load()
        except: pass
        try:
            if (img.mode=="P" and "transparency" in img.info) or img.mode in ("RGBA","LA"):
                img_rgba = img.convert("RGBA")
                bg = Image.new("RGB", img_rgba.size, (255,255,255))
                bg.paste(img_rgba, mask=img_rgba.split()[3])
                return bg
            else: return img.convert("RGB")
        except:
            return img.convert("RGB")

# -----------------------------
# Transforms
# -----------------------------
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(0.4,0.4,0.4,0.05),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------------
# Dataset
# -----------------------------
base_ds = ImageFolder(root=DATA_ROOT, transform=train_tf, loader=safe_image_loader)
class_to_idx = base_ds.class_to_idx
idx_to_class = {v:k for k,v in class_to_idx.items()}
num_classes = len(idx_to_class)
with open(os.path.join(WEIGHTS_DIR,"class_index.json"), "w") as f:
    json.dump({"class_to_idx": class_to_idx,"idx_to_class": idx_to_class}, f, indent=2)

targets = np.array([label for _, label in base_ds.samples])
train_indices, val_indices = [], []
for cls_id in range(num_classes):
    cls_idx = np.where(targets==cls_id)[0]
    np.random.shuffle(cls_idx)
    split = int(len(cls_idx)*(1-VAL_SPLIT))
    train_indices.extend(cls_idx[:split])
    val_indices.extend(cls_idx[split:])

train_ds = Subset(base_ds, train_indices)
val_base = ImageFolder(root=DATA_ROOT, transform=val_tf, loader=safe_image_loader)
val_ds = Subset(val_base, val_indices)

# Weighted sampler
class_counts = np.bincount([targets[i] for i in train_indices])
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[targets[i]] for i in train_indices]
train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoaders
num_workers = min(4, os.cpu_count() or 0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

# -----------------------------
# Model, optimizer, loss
# -----------------------------
model = HybridResNetViT(num_classes=num_classes).to(DEVICE)  # Replace with your model
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler()

# -----------------------------
# Training loop with full metrics
# -----------------------------
if __name__ == "__main__":
    best_f1 = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        
        # Train
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        pbar = tqdm(train_loader, desc="Train", ncols=100)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item()*batch_size
            preds = logits.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
        
        train_loss = running_loss/total
        train_acc = correct/total

        # Validate
        model.eval()
        running_loss, total, correct = 0.0, 0, 0
        y_true_all, y_pred_all = [], []
        pbar = tqdm(val_loader, desc="Val  ", ncols=100)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'), torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)
            batch_size = images.size(0)
            running_loss += loss.item()*batch_size
            total += batch_size
            preds = logits.argmax(dim=1)
            correct += (preds==labels).sum().item()
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{correct/total:.4f}"})

        val_loss = running_loss/total
        val_acc = correct/total
        f1_weighted = f1_score(y_true_all, y_pred_all, average='weighted')
        f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_all, y_pred_all)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=[idx_to_class[i] for i in range(num_classes)],
                    yticklabels=[idx_to_class[i] for i in range(num_classes)], cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Epoch {epoch} Confusion Matrix")
        plt.show()

        # Per-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_acc):
            print(f"Class {idx_to_class[i]} Accuracy: {acc:.4f}")

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_true_all, y_pred_all, target_names=[idx_to_class[i] for i in range(num_classes)]))

        scheduler.step()
        print(f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1(w)={f1_weighted:.4f} F1(m)={f1_macro:.4f}")

        # Save
        last_path = os.path.join(WEIGHTS_DIR, "last_hybrid_resnet_vit.pth")
        torch.save(model.state_dict(), last_path)
        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_path = os.path.join(WEIGHTS_DIR, "best_hybrid_resnet_vit.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best F1(w)={best_f1:.4f} saved to {best_path}")

    elapsed = time.time()-start
    print(f"\nTraining complete in {elapsed/60:.1f} min. Best F1(w)={best_f1:.4f}")