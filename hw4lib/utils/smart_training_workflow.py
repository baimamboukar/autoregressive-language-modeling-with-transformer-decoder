# Smart Training Workflow for HW4P2 - CER < 6% Strategy
import os
import yaml
import wandb
from pathlib import Path
from typing import List, Dict, Optional

def train_model_variant(config_path: str, epochs: int = 25, experiment_name: str = None) -> str:
    """
    Train a single model variant and return the wandb run ID.

    Args:
        config_path: Path to the config YAML file
        epochs: Number of epochs to train
        experiment_name: Optional custom experiment name

    Returns:
        wandb_run_id: The run ID for this training session
    """
    from hw4lib.trainers.asr_trainer import ASRTrainer
    from hw4lib.data.asr_dataset import ASRDataset
    from hw4lib.model.transformers import EncoderDecoderTransformer
    from hw4lib.data.tokenizer import H4Tokenizer
    import torch
    from torch.utils.data import DataLoader

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set experiment name if provided
    if experiment_name:
        config['Name'] = experiment_name

    print(f"\n🚀 Training: {config['Name']}")
    print(f"Config: {config_path}")
    print(f"Epochs: {epochs}")

    # Create tokenizer
    token_type = config['tokenization']['token_type']
    token_map = config['tokenization']['token_map']
    tokenizer = H4Tokenizer(token_map, token_type)

    # Create datasets
    train_dataset = ASRDataset(
        partition=config['data']['train_partition'],
        config=config['data'],
        tokenizer=tokenizer,
        isTrainPartition=True,
        global_stats=None
    )

    # Get global stats from training dataset for validation
    global_stats = None
    if config['data'].get('norm') == 'global_mvn':
        global_stats = (train_dataset.global_mean, train_dataset.global_std)

    val_dataset = ASRDataset(
        partition=config['data']['val_partition'],
        config=config['data'],
        tokenizer=tokenizer,
        isTrainPartition=False,
        global_stats=global_stats
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['NUM_WORKERS'],
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['NUM_WORKERS'],
        collate_fn=val_dataset.collate_fn
    )

    # Create model
    model_config = config['model'].copy()
    model_config['num_classes'] = tokenizer.get_vocab_size()
    model_config['max_len'] = 1024

    model = EncoderDecoderTransformer(**model_config)

    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    assert total_params < 30_000_000, f"Model too large: {total_params:,} > 30M"

    # Create trainer
    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        config_file=config_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Train
    trainer.train(epochs)

    # Return wandb run ID
    return trainer.wandb_run.id if trainer.use_wandb else None


def train_multiple_variants(
    config_paths: List[str],
    epochs_per_variant: int = 25,
    sequential: bool = True
) -> List[str]:
    """
    Train multiple model variants.

    Args:
        config_paths: List of config file paths
        epochs_per_variant: Epochs to train each variant
        sequential: If True, train sequentially. If False, return configs for parallel training

    Returns:
        List of wandb run IDs
    """
    run_ids = []

    if sequential:
        for i, config_path in enumerate(config_paths):
            print(f"\n{'='*60}")
            print(f"Training Variant {i+1}/{len(config_paths)}")
            print(f"{'='*60}")

            run_id = train_model_variant(
                config_path=config_path,
                epochs=epochs_per_variant,
                experiment_name=f"{Path(config_path).stem}_epoch_{epochs_per_variant}"
            )

            if run_id:
                run_ids.append(run_id)
                print(f"✅ Completed: {run_id}")
            else:
                print(f"❌ Failed: {config_path}")

    else:
        print("📋 Configs ready for parallel training:")
        for config_path in config_paths:
            print(f"  - {config_path}")
        print("Run each config manually in separate sessions/GPUs")
        return config_paths

    return run_ids


def finetune_best_models(
    run_ids: List[str],
    additional_epochs: int = 10,
    finetune_lr: float = 0.0005
) -> List[str]:
    """
    Fine-tune the best performing models with lower learning rate.

    Args:
        run_ids: List of wandb run IDs to fine-tune
        additional_epochs: Number of additional epochs
        finetune_lr: Lower learning rate for fine-tuning

    Returns:
        List of fine-tuned model run IDs
    """
    from resume_from_wandb import resume_and_continue_training

    finetuned_run_ids = []

    for i, run_id in enumerate(run_ids):
        print(f"\n🔧 Fine-tuning model {i+1}/{len(run_ids)}: {run_id}")

        try:
            # Resume and continue training with lower LR
            model, trainer = resume_and_continue_training(
                run_id=run_id,
                train_loader=None,  # Will be created by the function
                val_loader=None,    # Will be created by the function
                num_additional_epochs=additional_epochs,
                config_override={
                    'optimizer.lr': finetune_lr,
                    'scheduler.cosine.T_max': additional_epochs,
                    'Name': f"finetuned_{run_id}_{additional_epochs}ep"
                }
            )

            finetuned_run_id = trainer.wandb_run.id if trainer.use_wandb else None
            if finetuned_run_id:
                finetuned_run_ids.append(finetuned_run_id)
                print(f"✅ Fine-tuned: {finetuned_run_id}")

        except Exception as e:
            print(f"❌ Failed to fine-tune {run_id}: {e}")
            continue

    return finetuned_run_ids


def smart_training_pipeline():
    """
    Complete smart training pipeline for CER < 6%.
    """
    print("🎯 Smart Training Pipeline for CER < 6%")
    print("="*50)

    # Step 1: Define all variant configs
    variant_configs = [
        "config_variant_1_balanced.yaml",
        "config_variant_2_wider.yaml",
        "config_variant_3_deeper.yaml",
        "config_variant_4_less_reduction.yaml",
        "config_variant_5_small_vocab.yaml"
    ]

    # Check if configs exist
    existing_configs = []
    for config in variant_configs:
        if os.path.exists(config):
            existing_configs.append(config)
        else:
            print(f"⚠️  Config not found: {config}")

    print(f"📂 Found {len(existing_configs)} configs to train")

    # Step 2: Train base variants
    print("\n🚀 Step 1: Training base variants...")
    base_run_ids = train_multiple_variants(
        config_paths=existing_configs,
        epochs_per_variant=25,
        sequential=True
    )

    print(f"\n✅ Base training completed: {len(base_run_ids)} models")
    for i, run_id in enumerate(base_run_ids):
        print(f"  {i+1}. {run_id}")

    # Step 3: Fine-tune best models
    print(f"\n🔧 Step 2: Fine-tuning top models...")
    finetuned_run_ids = finetune_best_models(
        run_ids=base_run_ids,
        additional_epochs=10,
        finetune_lr=0.0005
    )

    print(f"\n✅ Fine-tuning completed: {len(finetuned_run_ids)} models")

    # Step 4: Combine all run IDs for ensemble
    all_run_ids = base_run_ids + finetuned_run_ids

    print(f"\n🎯 Pipeline Complete!")
    print(f"Total models trained: {len(all_run_ids)}")
    print(f"Ready for ensemble: {all_run_ids}")

    # Save run IDs for ensemble
    with open("trained_models_run_ids.txt", "w") as f:
        f.write("# Trained Model Run IDs for Ensemble\n")
        f.write("# Base models:\n")
        for run_id in base_run_ids:
            f.write(f"{run_id}\n")
        f.write("\n# Fine-tuned models:\n")
        for run_id in finetuned_run_ids:
            f.write(f"{run_id}\n")

    print(f"💾 Run IDs saved to: trained_models_run_ids.txt")

    # Step 5: Create ensemble
    print(f"\n🤝 Creating ensemble...")
    from ensemble_asr import ensemble_from_wandb

    try:
        ensemble_model = ensemble_from_wandb(
            run_ids=all_run_ids,
            beam_width=15,
            length_penalty=1.0,
            temperature=0.9
        )

        print(f"✅ Ensemble created successfully!")
        print(f"📈 Ready for validation and final testing")

        return ensemble_model, all_run_ids

    except Exception as e:
        print(f"❌ Ensemble creation failed: {e}")
        print(f"💡 Manual ensemble creation:")
        print(f"   from ensemble_asr import ensemble_from_wandb")
        print(f"   ensemble = ensemble_from_wandb({all_run_ids})")

        return None, all_run_ids


# Convenience functions
def quick_train_single(config_name: str, epochs: int = 25) -> str:
    """Quick train a single configuration."""
    return train_model_variant(f"config_{config_name}.yaml", epochs)


def parallel_training_setup():
    """Print commands for parallel training on multiple GPUs."""
    configs = [
        "config_variant_1_balanced.yaml",
        "config_variant_2_wider.yaml",
        "config_variant_3_deeper.yaml",
        "config_variant_4_less_reduction.yaml",
        "config_variant_5_small_vocab.yaml"
    ]

    print("🚀 Parallel Training Setup")
    print("Run these commands on separate GPUs/sessions:")
    print("="*50)

    for i, config in enumerate(configs):
        print(f"# GPU {i}")
        print(f"CUDA_VISIBLE_DEVICES={i} python -c \"")
        print(f"from smart_training_workflow import train_model_variant")
        print(f"train_model_variant('{config}', epochs=25)\"")
        print()


if __name__ == "__main__":
    # Run the complete smart training pipeline
    ensemble_model, run_ids = smart_training_pipeline()

    if ensemble_model:
        print("🎉 All done! Your ensemble is ready for CER < 6%")
    else:
        print("⚠️  Please create ensemble manually with the run IDs above")