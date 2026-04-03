import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from notebooks.s01_data import stop_1_load_flowers, stop_2_transforms, stop_3_dataloaders
from notebooks.s02_baseline import run_baseline_training
from notebooks.s03_finetune import run_fine_tuning
from notebooks.s04_optuna import run_optuna_search

def run_full_pipeline(auto_mode=True):
    """Orchestrate the entire 13-STOP Engineering Workflow"""
    print("\n" + "="*50)
    print("🚀 STARTING TRANSFER LEARNING & FINE-TUNING PIPELINE")
    print("="*50 + "\n")
    
    # -- PHASE 1: DATA --
    print("PHASE 1: DATA INGESTION")
    stop_1_load_flowers()
    tr_tf, vl_tf = stop_2_transforms()
    stop_3_dataloaders(tr_tf, vl_tf)
    
    if auto_mode:
        # -- PHASE 2: BASELINE --
        print("\nPHASE 2: FEATURE EXTRACTION (BASELINE)")
        run_baseline_training(epochs=2)
        
        # -- PHASE 3: FINE-TUNING --
        print("\nPHASE 3: SELECTIVE FINE-TUNING")
        run_fine_tuning(epochs=2)
        
        # -- PHASE 4: OPTUNA --
        print("\nPHASE 4: BAYESIAN HPT")
        run_optuna_search(n_trials=2)
        
        print("\n" + "="*50)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("="*50 + "\n")
    else:
        print("\n[INFO] Auto-training skipped. Run with --auto to trigger training loops.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Run full training loops (auto mode)")
    args = parser.parse_args()
    
    run_full_pipeline(auto_mode=args.auto)
