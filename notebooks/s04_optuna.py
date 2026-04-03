import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from path_utils import MODELS
from notebooks.s01_data import stop_2_transforms, stop_3_dataloaders
from notebooks.s02_baseline import stop_4_load_resnet, stop_5_freeze_backbone
from notebooks.s03_finetune import stop_7_selective_unfreeze
from notebooks.engine import train_model

def objective(trial):
    """STOP 10 — Optuna Objective Function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Suggest Hyperparameters
    lr_fc = trial.suggest_float("lr_fc", 1e-4, 1e-2, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    dropout_p = trial.suggest_float("dropout", 0.2, 0.5)
    
    # Setup Model
    model = stop_4_load_resnet()
    model = stop_5_freeze_backbone(model)
    # Add dropout to head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(num_ftrs, 102)
    )
    model = stop_7_selective_unfreeze(model)
    model = model.to(device)
    
    # Optimizer with Differential LR
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': lr_backbone / 10},
        {'params': model.layer4.parameters(), 'lr': lr_backbone},
        {'params': model.fc.parameters(),     'lr': lr_fc},
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    # Data
    tr_tf, vl_tf = stop_2_transforms()
    train_loader, val_loader, _ = stop_3_dataloaders(tr_tf, vl_tf)
    
    # Train for fewer epochs for HPT (e.g., 5)
    _, history = train_model(model, train_loader, val_loader, criterion, optimizer, None, device, num_epochs=3)
    
    return max(history["val_acc"])

def run_optuna_search(n_trials=5):
    """STOPS 11 & 12 — Optimization & Final Params"""
    print("--- STOP 11 & 12: OPTUNA BAYESIAN SEARCH ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n[SUCCESS] Best Study Params:")
    print(study.best_params)
    
    # Save best params
    with open(MODELS / "best_params.json", "w") as f:
        json.dump(study.best_params, f)
        
    print(f"HPT Complete. Best accuracy: {study.best_value:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3, help="Number of Optuna trials")
    args = parser.parse_args()
    
    run_optuna_search(n_trials=args.trials)
