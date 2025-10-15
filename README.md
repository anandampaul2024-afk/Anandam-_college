# 📦 Reconstructing the Model File

The model file `best_hybrid_resnet_vit.pth` has been split into smaller parts to comply with GitHub's file size limits.

## 🔧 Step 1: Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

## 🔧 Step 2: Reconstruct the `.pth` file
Run the following command in your terminal:

```bash
cat weights/best_hybrid_resnet_vit_part_* > weights/best_hybrid_resnet_vit.pth
```

This will combine all the parts into the original model file.

## ✅ Done!
You can now use `best_hybrid_resnet_vit.pth` in your PyTorch project as usual.
