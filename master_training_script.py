#!/usr/bin/env python3
"""
Master Training Script for HW4P2 - CER < 6% Strategy
All-in-one solution for training, ensembling, and achieving target CER.

Author: Generated for HW4P2 Assignment
Usage:
    python master_training_script.py --mode quick_test
    python master_training_script.py --mode full_pipeline
    python master_training_script.py --mode ensemble_only --run_ids run1,run2,run3
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import List, Optional

def setup_environment():
    """Setup necessary directories and check dependencies."""
    print("🔧 Setting up environment...")

    # Create directories
    dirs = ['experiments', 'checkpoints', 'validation_results', 'optimization_configs', 'averaged_models']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  ✅ Created/verified: {dir_name}/")

    print("✅ Environment setup complete!")

def quick_test_mode():
    """
    Quick test mode: Train one fast model to verify everything works.
    """
    print("\n🚀 Quick Test Mode")
    print("="*50)

    from smart_training_workflow import train_model_variant

    # Train one quick model
    run_id = train_model_variant(
        config_path="optimization_configs/config_quick_wins.yaml",
        epochs=5,
        experiment_name="quick_test_5ep"
    )

    if run_id:
        print(f"\n✅ Quick test successful! Run ID: {run_id}")

        # Quick validation
        from ensemble_validation_strategy import EnsembleValidator
        validator = EnsembleValidator(tokenizer=None)  # Will be created internally

        achieved = validator.quick_validation([run_id], None, target_cer=0.15)  # Lower target for quick test

        if achieved:
            print("🎯 Quick test passed! Ready for full pipeline.")
        else:
            print("⚠️  Quick test CER higher than expected, but infrastructure works.")

        return run_id
    else:
        print("❌ Quick test failed!")
        return None

def full_pipeline_mode():
    """
    Full training pipeline: Train multiple models, ensemble, validate.
    """
    print("\n🎯 Full Training Pipeline Mode")
    print("="*60)

    from smart_training_workflow import smart_training_pipeline

    # Run complete pipeline
    ensemble_model, run_ids = smart_training_pipeline()

    if ensemble_model and len(run_ids) > 0:
        print(f"\n🎉 Full pipeline completed successfully!")
        print(f"Trained models: {len(run_ids)}")
        print(f"Run IDs: {run_ids}")

        # Comprehensive validation
        print(f"\n🔍 Running comprehensive validation...")
        from ensemble_validation_strategy import validate_ensemble_pipeline

        results = validate_ensemble_pipeline(run_ids)

        best_cer = min(r.cer for r in results.values() if r is not None)

        if best_cer <= 0.06:
            print(f"🎯 TARGET ACHIEVED! Best CER: {best_cer:.4f} <= 0.06")
        else:
            print(f"📈 Best CER: {best_cer:.4f} (Target: 0.06)")
            print(f"💡 Consider: More fine-tuning, better beam search params, or additional models")

        return run_ids, best_cer
    else:
        print("❌ Full pipeline failed!")
        return None, None

def ensemble_only_mode(run_ids: List[str]):
    """
    Ensemble only mode: Create ensemble from existing run IDs.
    """
    print(f"\n🤝 Ensemble Only Mode")
    print(f"Run IDs: {run_ids}")
    print("="*50)

    # Test different ensemble methods
    from weight_averaging_ensemble import compare_ensemble_methods
    from ensemble_validation_strategy import EnsembleValidator

    print("🔬 Comparing ensemble methods...")

    # Create dummy validation loader (you'll need to adjust this)
    validator = EnsembleValidator(tokenizer=None)

    # Quick ensemble test
    from ensemble_asr import ensemble_from_wandb

    try:
        ensemble = ensemble_from_wandb(run_ids)
        print("✅ Output averaging ensemble created successfully")

        # Quick validation
        achieved = validator.quick_validation(run_ids, None, target_cer=0.06)

        if achieved:
            print("🎯 TARGET ACHIEVED with ensemble!")
        else:
            print("📈 Ensemble created, consider beam search tuning")

            # Beam search tuning
            from beam_search_tuning import quick_beam_search_sweep
            print("🔍 Running beam search parameter tuning...")

            tuning_results = quick_beam_search_sweep(run_ids, fast_mode=True)

            if tuning_results:
                best_config = min(tuning_results.items(), key=lambda x: x[1]['best_cer'] if x[1] else float('inf'))
                print(f"🏆 Best beam config: {best_config[1]['best_params']}")
                print(f"   Best CER: {best_config[1]['best_cer']:.4f}")

        return True

    except Exception as e:
        print(f"❌ Ensemble creation failed: {e}")
        return False

def resume_training_mode(run_id: str, additional_epochs: int = 10):
    """
    Resume training mode: Continue training from a specific run.
    """
    print(f"\n🔄 Resume Training Mode")
    print(f"Run ID: {run_id}")
    print(f"Additional epochs: {additional_epochs}")
    print("="*50)

    from resume_from_wandb import resume_and_continue_training

    try:
        model, trainer = resume_and_continue_training(
            run_id=run_id,
            train_loader=None,  # Will be created internally
            val_loader=None,    # Will be created internally
            num_additional_epochs=additional_epochs,
            config_override={
                'optimizer.lr': 0.0005,  # Lower LR for fine-tuning
                'Name': f"resumed_{run_id}_{additional_epochs}ep"
            }
        )

        new_run_id = trainer.wandb_run.id if trainer.use_wandb else None

        if new_run_id:
            print(f"✅ Resume training successful! New run ID: {new_run_id}")
            return new_run_id
        else:
            print("⚠️  Training completed but no wandb run ID")
            return None

    except Exception as e:
        print(f"❌ Resume training failed: {e}")
        return None

def interactive_mode():
    """
    Interactive mode: Let user choose what to do.
    """
    print("\n🎮 Interactive Mode")
    print("="*40)

    while True:
        print("\nChoose an option:")
        print("1. Quick test (5 epochs)")
        print("2. Train single model (specify config)")
        print("3. Full pipeline (all variants + ensemble)")
        print("4. Create ensemble from run IDs")
        print("5. Resume training from run ID")
        print("6. Beam search tuning")
        print("7. Exit")

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == '1':
            quick_test_mode()
        elif choice == '2':
            config_path = input("Config path: ").strip()
            epochs = int(input("Epochs (default 25): ") or 25)

            from smart_training_workflow import train_model_variant
            run_id = train_model_variant(config_path, epochs)
            print(f"Run ID: {run_id}")

        elif choice == '3':
            full_pipeline_mode()
        elif choice == '4':
            run_ids_str = input("Enter run IDs (comma separated): ").strip()
            run_ids = [rid.strip() for rid in run_ids_str.split(',')]
            ensemble_only_mode(run_ids)
        elif choice == '5':
            run_id = input("Run ID to resume: ").strip()
            epochs = int(input("Additional epochs (default 10): ") or 10)
            resume_training_mode(run_id, epochs)
        elif choice == '6':
            run_ids_str = input("Enter run IDs for tuning (comma separated): ").strip()
            run_ids = [rid.strip() for rid in run_ids_str.split(',')]

            from beam_search_tuning import quick_beam_search_sweep
            quick_beam_search_sweep(run_ids)
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Master Training Script for HW4P2")
    parser.add_argument('--mode', choices=['quick_test', 'full_pipeline', 'ensemble_only', 'resume', 'interactive'],
                       default='interactive', help='Training mode')
    parser.add_argument('--run_ids', type=str, help='Comma-separated run IDs for ensemble mode')
    parser.add_argument('--run_id', type=str, help='Single run ID for resume mode')
    parser.add_argument('--epochs', type=int, default=10, help='Additional epochs for resume mode')

    args = parser.parse_args()

    print("🎯 HW4P2 Master Training Script - CER < 6% Strategy")
    print("="*60)

    # Setup environment
    setup_environment()

    # Run based on mode
    if args.mode == 'quick_test':
        quick_test_mode()
    elif args.mode == 'full_pipeline':
        full_pipeline_mode()
    elif args.mode == 'ensemble_only':
        if not args.run_ids:
            print("❌ Error: --run_ids required for ensemble_only mode")
            sys.exit(1)
        run_ids = [rid.strip() for rid in args.run_ids.split(',')]
        ensemble_only_mode(run_ids)
    elif args.mode == 'resume':
        if not args.run_id:
            print("❌ Error: --run_id required for resume mode")
            sys.exit(1)
        resume_training_mode(args.run_id, args.epochs)
    elif args.mode == 'interactive':
        interactive_mode()

    print("\n🎉 Script completed!")

if __name__ == "__main__":
    main()