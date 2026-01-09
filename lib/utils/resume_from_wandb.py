# Resume Training from Wandb for P2
def list_wandb_run_files(run_id: str):
    """List all files available in a wandb run."""
    import wandb

    api = wandb.Api()
    run = api.run(f"idlf25/P2/{run_id}")

    print(f"Files in run {run_id} ({run.name}) - State: {run.state}")
    print("="*60)
    files = run.files()

    # Group files by directory
    from collections import defaultdict
    file_structure = defaultdict(list)

    for f in files:
        parts = f.name.split('/')
        if len(parts) > 1:
            directory = '/'.join(parts[:-1])
            filename = parts[-1]
            file_structure[directory].append(filename)
        else:
            file_structure['[root]'].append(f.name)

    # Print organized structure
    for directory in sorted(file_structure.keys()):
        print(f"\n{directory}:")
        for filename in sorted(file_structure[directory]):
            print(f"  - {filename}")

    return [f.name for f in files]


def resume_from_wandb(run_id: str, config_override: dict = None, checkpoint_type: str = "best"):
    """
    Resume training from a specific wandb run for ASR models.
    Downloads the latest checkpoint and continues training from where it stopped.

    Args:
        run_id: Wandb run ID to resume from
        config_override: Optional dict to override specific config values (e.g., epochs, lr)
        checkpoint_type: "best" or "last" to specify which checkpoint to load

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
    from lib.model.transformers import EncoderDecoderTransformer
    from lib.data.tokenizer import H4Tokenizer

    print(f"\nResuming from Wandb Run: {run_id}")

    # Initialize wandb API
    api = wandb.Api()
    run = api.run(f"idlf25/P2/{run_id}")

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

        # Try to get the last checkpoint or best model based on checkpoint_type
        checkpoint_file = None

        # Prioritize based on checkpoint_type
        if checkpoint_type == "best":
            possible_names = [
                'checkpoints/checkpoint-best-metric-model.pth',
                'checkpoints/best_model.pth',
                'best_model.pth',
                'checkpoints/checkpoint-last-epoch-model.pth',  # Fallback to last if best not found
            ]
        else:  # checkpoint_type == "last"
            possible_names = [
                'checkpoints/checkpoint-last-epoch-model.pth',
                'checkpoints/last_model.pth',
                'checkpoint-last-epoch.pth',
                'checkpoints/checkpoint-best-metric-model.pth',  # Fallback to best if last not found
            ]

        # Add any found checkpoint files to the list
        if checkpoint_files:
            possible_names.extend([f.name for f in checkpoint_files])

        for name in possible_names:
            try:
                file = run.file(name)
                if file:
                    checkpoint_file = file
                    print(f"Found checkpoint at: {name}")
                    break
            except:
                continue

        if not checkpoint_file:
            raise ValueError("Could not find suitable checkpoint file")

        # Download checkpoint
        local_checkpoint_path = f"checkpoints/{checkpoint_file.name}"
        checkpoint_file.download(root="checkpoints", replace=True)
        print(f"Downloaded checkpoint: {checkpoint_file.name}")

        # Download config - try different possible paths
        config_file = None
        config_paths = [
            'run.2/config.yaml',
            'run.1/config.yaml',
            'experiments/run.1/config.yaml',
            'experiments/run.2/config.yaml',
            'config.yaml'
        ]

        for path in config_paths:
            try:
                config_file = run.file(path)
                if config_file:
                    print(f"Found config at: {path}")
                    break
            except:
                continue

        if not config_file:
            raise ValueError("Could not find config.yaml file")

        # Download config to checkpoints directory
        config_file.download(root="checkpoints", replace=True)

        # Load config - construct the actual path where it was downloaded
        config_local_path = os.path.join("checkpoints", config_file.name)
        with open(config_local_path, 'r') as f:
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

        # Load checkpoint first to inspect the model architecture
        checkpoint = torch.load(local_checkpoint_path, map_location='cpu')

        # Extract the correct max_len from the saved positional encoding shape
        if 'model_state_dict' in checkpoint:
            # Look for positional encoding tensor to get the correct max_len
            pe_key = 'positional_encoding.pe'
            if pe_key in checkpoint['model_state_dict']:
                pe_shape = checkpoint['model_state_dict'][pe_key].shape
                actual_max_len = pe_shape[1]  # Shape is [1, max_len, d_model]
                print(f"Detected max_len from checkpoint: {actual_max_len}")
            else:
                actual_max_len = 1024  # Fallback to default
                print(f"Could not detect max_len, using default: {actual_max_len}")
        else:
            actual_max_len = 1024

        # Create model with correct architecture
        model_config = config['model'].copy()
        model_config['num_classes'] = tokenizer.vocab_size
        model_config['max_len'] = actual_max_len  # Use the correct max_len from checkpoint

        print(f"Creating model: d_model={model_config['d_model']}, "
              f"enc_layers={model_config['num_encoder_layers']}, "
              f"dec_layers={model_config['num_decoder_layers']}, "
              f"max_len={model_config['max_len']}")

        model = EncoderDecoderTransformer(**model_config)

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
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")
            except ValueError as e:
                if "different number of parameter groups" in str(e):
                    print(f"Warning: Could not load optimizer state - {e}")
                    print("Using fresh optimizer state instead")
                else:
                    raise
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
    config_override: dict = None,
    checkpoint_type: str = "best"
):
    """
    Resume from wandb and continue training for additional epochs.

    Args:
        run_id: Wandb run ID to resume from
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_additional_epochs: Number of additional epochs to train
        config_override: Optional config overrides
        checkpoint_type: "best" or "last" to specify which checkpoint to load

    Returns:
        model: Trained model
        trainer: ASRTrainer instance
    """
    from lib.trainers.asr_trainer import ASRTrainer
    import torch
    import wandb

    # Resume from wandb
    model, optimizer, scheduler, start_epoch, config, tokenizer = resume_from_wandb(
        run_id,
        config_override,
        checkpoint_type
    )

    # Calculate total epochs
    total_epochs = start_epoch + num_additional_epochs

    print("\n=== Continuing Training ===")
    print(f"Training from epoch {start_epoch} to {total_epochs}")

    # Initialize new wandb run for continued training
    wandb.init(
        project="P2",
        name=f"{config['Name']}_resumed_epoch_{start_epoch}",
        config=config,
        resume="allow"  # Allow resuming but don't require it
    )

    # Create trainer
    trainer = ASRTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        run_name=f"{config['Name']}_resumed_epoch_{start_epoch}",
        config_file="config.yaml",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Set optimizer and scheduler
    trainer.optimizer = optimizer

    # Create fresh scheduler if needed
    if scheduler is None:
        from lib.utils import create_scheduler
        trainer.scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_config=config['scheduler'],
            train_loader=train_loader,
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
        )
        print("Created fresh scheduler")
    else:
        trainer.scheduler = scheduler
        print("Using loaded scheduler")

    # Set starting epoch
    trainer.current_epoch = start_epoch

    # Train for additional epochs
    print(f"\nTraining for {num_additional_epochs} additional epochs...")

    # Use the trainer's train method with dataloaders
    trainer.train(train_loader, val_loader, num_additional_epochs)

    print("\nTraining completed!")

    wandb.finish()

    return model, trainer