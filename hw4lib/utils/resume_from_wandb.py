# Resume Training from Wandb for HW4P2
def resume_from_wandb(run_id: str, config_override: dict = None):
    """
    Resume training from a specific wandb run for ASR models.
    Downloads the latest checkpoint and continues training from where it stopped.

    Args:
        run_id: Wandb run ID to resume from
        config_override: Optional dict to override specific config values (e.g., epochs, lr)

    Returns:
        model: Loaded EncoderDecoderTransformer model
        optimizer: Optimizer with loaded state
        scheduler: Scheduler with loaded state
        start_epoch: Epoch to start training from
        config: Full configuration (loaded + overrides)
        tokenizer: H4Tokenizer instance
    """
    import wandb
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
    import yaml
    import os
    from hw4lib.model.transformers import EncoderDecoderTransformer
    from hw4lib.data.tokenizer import H4Tokenizer

    print(f"\n=== Resuming from Wandb Run: {run_id} ===")

    # Initialize wandb API
    api = wandb.Api()
    run = api.run(f"HW4P2/{run_id}")

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    try:
        # Download files from wandb
        print("Downloading files from wandb...")

        # Try to find the latest checkpoint
        checkpoint_files = [f for f in run.files() if f.name.endswith('.pth')]

        if not checkpoint_files:
            raise ValueError("No checkpoint files found in run")

        # Sort by name to get the latest (assuming naming like checkpoint_epoch_20.pth)
        checkpoint_files.sort(key=lambda x: x.name)

        # Try to get the last checkpoint or best model
        checkpoint_file = None
        for name in ['checkpoint-last-epoch.pth', 'best_model.pth', checkpoint_files[-1].name]:
            try:
                file = run.file(name)
                if file:
                    checkpoint_file = file
                    break
            except:
                continue

        if not checkpoint_file:
            raise ValueError("Could not find suitable checkpoint file")

        # Download checkpoint
        local_checkpoint_path = f"checkpoints/{checkpoint_file.name}"
        checkpoint_file.download(root="checkpoints", replace=True)
        print(f"Downloaded checkpoint: {checkpoint_file.name}")

        # Download config
        config_file = run.file('config.yaml')
        config_file.download(root="checkpoints", replace=True)

        # Load config
        with open("checkpoints/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        # Apply config overrides if provided
        if config_override:
            print(f"Applying config overrides: {config_override}")
            for key, value in config_override.items():
                # Handle nested keys like 'training.epochs'
                keys = key.split('.')
                target = config
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value

        # Create tokenizer
        token_type = config['tokenization']['token_type']
        token_map = config['tokenization']['token_map']
        tokenizer = H4Tokenizer(token_map, token_type)
        print(f"Tokenizer loaded: {token_type} ({tokenizer.vocab_size} tokens)")

        # Create model
        model_config = config['model'].copy()
        model_config['num_classes'] = tokenizer.vocab_size
        model_config['max_len'] = 1024  # Default max length

        print(f"Creating model: d_model={model_config['d_model']}, "
              f"enc_layers={model_config['num_encoder_layers']}, "
              f"dec_layers={model_config['num_decoder_layers']}")

        model = EncoderDecoderTransformer(**model_config)

        # Load checkpoint
        checkpoint = torch.load(local_checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")

        # Get epoch information
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_cer = checkpoint.get('best_cer', float('inf'))

        print(f"Resuming from epoch {start_epoch}")
        print(f"Previous best CER: {best_cer:.4f}")

        # Create optimizer
        optimizer_config = config['optimizer']
        if optimizer_config['name'].lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay'],
                betas=optimizer_config['adamw']['betas'],
                eps=optimizer_config['adamw']['eps'],
                amsgrad=optimizer_config['adamw']['amsgrad']
            )
        elif optimizer_config['name'].lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        else:
            print("Warning: No optimizer state found, using fresh optimizer")

        # Create scheduler
        scheduler = None
        scheduler_config = config['scheduler']

        if scheduler_config['name'].lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['cosine']['T_max'],
                eta_min=scheduler_config['cosine']['eta_min'],
                last_epoch=scheduler_config['cosine'].get('last_epoch', -1)
            )
        elif scheduler_config['name'].lower() == 'exponential':
            scheduler = ExponentialLR(
                optimizer,
                gamma=scheduler_config['exponential']['gamma'],
                last_epoch=scheduler_config['exponential'].get('last_epoch', -1)
            )

        # Load scheduler state if available
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to GPU")

        # Print summary
        print("\n=== Resume Summary ===")
        print(f"Run ID: {run_id}")
        print(f"Model: {config['Name']}")
        print(f"Starting from epoch: {start_epoch}")
        print(f"Previous best CER: {best_cer:.4f}")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

        return model, optimizer, scheduler, start_epoch, config, tokenizer

    except Exception as e:
        print(f"\nError resuming from wandb: {e}")
        print("You may need to start training from scratch")
        raise


def resume_and_continue_training(
    run_id: str,
    train_loader,
    val_loader,
    num_additional_epochs: int = 10,
    config_override: dict = None
):
    """
    Resume from wandb and continue training for additional epochs.

    Args:
        run_id: Wandb run ID to resume from
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_additional_epochs: Number of additional epochs to train
        config_override: Optional config overrides

    Returns:
        model: Trained model
        trainer: ASRTrainer instance
    """
    from hw4lib.trainers.asr_trainer import ASRTrainer
    import torch
    import wandb

    # Resume from wandb
    model, optimizer, scheduler, start_epoch, config, tokenizer = resume_from_wandb(
        run_id,
        config_override
    )

    # Calculate total epochs
    total_epochs = start_epoch + num_additional_epochs

    print(f"\n=== Continuing Training ===")
    print(f"Training from epoch {start_epoch} to {total_epochs}")

    # Initialize new wandb run for continued training
    wandb.init(
        project="HW4P2",
        name=f"{config['Name']}_resumed_epoch_{start_epoch}",
        config=config,
        resume="allow"  # Allow resuming but don't require it
    )

    # Create trainer
    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        exp_dir=f"experiments/{config['Name']}_resumed"
    )

    # Set starting epoch
    trainer.current_epoch = start_epoch

    # Train for additional epochs
    print(f"\nTraining for {num_additional_epochs} additional epochs...")
    for epoch in range(start_epoch, total_epochs):
        trainer.current_epoch = epoch

        # Train epoch
        train_loss, train_cer = trainer.train_epoch()

        # Validate epoch
        val_loss, val_cer = trainer.valid_epoch()

        print(f"Epoch {epoch}/{total_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, CER: {train_cer:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, CER: {val_cer:.4f}")

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_cer': train_cer,
            'val_loss': val_loss,
            'val_cer': val_cer,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Step scheduler
        if scheduler:
            scheduler.step()

        # Save checkpoint
        if val_cer < trainer.best_cer:
            trainer.best_cer = val_cer
            trainer.save_checkpoint(epoch, val_cer, is_best=True)
            print(f"  New best model! CER: {val_cer:.4f}")

    print(f"\nTraining completed! Final best CER: {trainer.best_cer:.4f}")

    # Save final model
    trainer.save_checkpoint(total_epochs - 1, trainer.best_cer, is_best=False)

    wandb.finish()

    return model, trainer