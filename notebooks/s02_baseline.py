import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import MODELS
from notebooks.s01_data import stop_2_transforms, stop_3_dataloaders
from notebooks.engine import train_model

def stop_4_load_resnet():
    """STOP 4 — Load Pretrained ResNet-50"""
    print("--- STOP 4: LOADING RESNET-50 ---")
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Architecture Overview:
    # ResNet-50 uses Residual Blocks to solve the vanishing gradient problem.
    # It has ~25.6M parameters.
    return model

def stop_5_freeze_backbone(model):
    """STOP 5 — Layer Freezing"""
    print("--- STOP 5: FREEZING BACKBONE ---")
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102) # 102 classes for Flowers102
    
    print(f"Feature Extraction Mode: Only {sum(p.numel() for p in model.fc.parameters())} params trainable.")
    return model

def run_baseline_training(epochs=10):
    """STOP 6 — Baseline Training"""
    print("--- STOP 6: BASELINE TRAINING ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Data
    tr_tf, vl_tf = stop_2_transforms()
    train_loader, val_loader, _ = stop_3_dataloaders(tr_tf, vl_tf)
    
    # Setup Model
    model = stop_4_load_resnet()
    model = stop_5_freeze_backbone(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Train
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, None, device, num_epochs=epochs)
    
    # Save
    torch.save(model.state_dict(), MODELS / "feature_extract.pkl")
    with open(MODELS / "history_baseline.json", "w") as f:
        json.dump(history, f)
        
    print(f"[SUCCESS] Baseline Model Saved to {MODELS}/feature_extract.pkl")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    args = parser.parse_args()
    
    run_baseline_training(epochs=args.epochs)
