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

def stop_7_selective_unfreeze(model):
    """STOP 7 — Selective Layer Unfreezing"""
    print("--- STOP 7: SELECTIVE UNFREEZING ---")
    # Rule of thumb: Higher layers encode high-level features.
    # We unfreeze layer3 and layer4 for domain adaptation.
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    print(f"Fine-tuning Mode: {sum(p.numel() for p in model.parameters() if p.requires_grad)} params trainable.")
    return model

def stop_8_differential_lr(model):
    """STOP 8 — Differential Learning Rates"""
    print("--- STOP 8: DIFFERENTIAL LR ---")
    # Base layers need small updates, new head needs larger updates.
    # 10x rule: each group gets ~10x smaller LR than the one above it.
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(),     'lr': 1e-3},
    ])
    return optimizer

def run_fine_tuning(epochs=15):
    """STOP 9 — Fine-tuning Training Loop"""
    print("--- STOP 9: FINE-TUNING TRAINING ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Data
    tr_tf, vl_tf = stop_2_transforms()
    train_loader, val_loader, _ = stop_3_dataloaders(tr_tf, vl_tf)
    
    # Step A: Load and Freeze
    from notebooks.s02_baseline import stop_4_load_resnet, stop_5_freeze_backbone
    model = stop_4_load_resnet()
    model = stop_5_freeze_backbone(model)
    
    # Step B: Selectively Unfreeze
    model = stop_7_selective_unfreeze(model)
    model = model.to(device)
    
    # Step C: Setup Differential LR & Scheduler
    optimizer = stop_8_differential_lr(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=epochs)
    
    # Save
    torch.save(model.state_dict(), MODELS / "finetuned.pkl")
    with open(MODELS / "history_finetune.json", "w") as f:
        json.dump(history, f)
        
    print(f"[SUCCESS] Fine-tuned Model Saved to {MODELS}/finetuned.pkl")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    args = parser.parse_args()
    
    run_fine_tuning(epochs=args.epochs)
