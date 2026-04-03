import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import DATA_RAW, CHARTS

def stop_1_load_flowers():
    """STOP 1 — Load Flowers102 Dataset"""
    print("--- STOP 1: LOADING FLOWERS102 ---")
    # Download dataset
    train_ds = datasets.Flowers102(root=DATA_RAW, split='train', download=True)
    val_ds = datasets.Flowers102(root=DATA_RAW, split='val', download=True)
    test_ds = datasets.Flowers102(root=DATA_RAW, split='test', download=True)
    
    print(f"Dataset Loaded Successfully!")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    # Fundamental premise: Transfer learning is essential here because ~8k images 
    # across 102 classes equals only ~80 images per class. 
    # Building from scratch would lead to severe overfitting.
    return train_ds, val_ds, test_ds

def stop_2_transforms():
    """STOP 2 — ImageNet Normalization & Augmentation"""
    print("--- STOP 2: TRANSFORMS & NORMALIZATION ---")
    # ALWAYS use ImageNet normalization for pretrained models
    # It ensures the input pixel distributions match what the model was trained on.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    print(f"Transforms Configured: Resize=224x224, Normalization=ImageNet")
    return train_tf, val_tf

def stop_3_dataloaders(train_tf, val_tf):
    """STOP 3 — DataLoader Setup"""
    print("--- STOP 3: DATALOADERS ---")
    train_ds = datasets.Flowers102(root=DATA_RAW, split='train', transform=train_tf)
    val_ds = datasets.Flowers102(root=DATA_RAW, split='val', transform=val_tf)
    test_ds = datasets.Flowers102(root=DATA_RAW, split='test', transform=val_tf) # Use val_tf for testing
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"DataLoaders Ready: Batch Size=32")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Run without user interaction")
    args = parser.parse_args()
    
    stop_1_load_flowers()
    tr_tf, vl_tf = stop_2_transforms()
    stop_3_dataloaders(tr_tf, vl_tf)
    print("\n[SUCCESS] DATA SETUP COMPLETE")
